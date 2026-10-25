"""Stage 11 pilot calculations. No upstream artifacts are changed.

11A is the matched cohort. 11B is a discrete-time population risk set:
one focal aircraft observed at the start of a 30 s interval contributes 30 s.
An actionable clearance in (tick, tick+30] is assigned to that pre-event row.
Both arms use the same nearest horizontal neighbour at the tick.  This is an
event intensity conditional on an observable snapshot, not conflict risk.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.stats import chi2

BASE = Path(__file__).resolve().parents[2] / "data/processed/scat"
OUT = BASE / "11_intervention_geometry"
WEEK = "20161015_20161021"
PAIR_FIELDS = ["window_id", "other_flight_key", "relative_seconds", "time_epoch",
               "horizontal_separation_nm", "vertical_separation_ft",
               "horizontal_closing_rate_nm_min", "vertical_closing_rate_fpm",
               "cpa_nm", "tcpa_min", "predicted_min_vertical_sep_ft",
               "focal_latitude", "focal_longitude", "focal_altitude_ft"]
STATE_FIELDS = ["flight_key", "archive_week", "time_epoch", "latitude", "longitude",
                "altitude_ft", "ground_speed_kt", "track_deg", "vertical_rate_fpm",
                "source_archive", "plot_index"]


def snapshot_tick(epoch: np.ndarray, step: int) -> np.ndarray:
    """Earliest regular grid time that does not precede a native state."""
    return (np.ceil(np.asarray(epoch,dtype=float)/step)*step).astype("int64")


def event_tick(epoch: np.ndarray, step: int) -> np.ndarray:
    """Risk-set start strictly before an event, including events on a tick."""
    return snapshot_tick(epoch,step)-step


def save(frame: pd.DataFrame, relative: str) -> None:
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False, compression="zstd")
    else:
        frame.to_csv(path, index=False)


def _week_groups(parquet: pq.ParquetFile, week: str):
    col = parquet.schema.names.index("archive_week")
    for i in range(parquet.num_row_groups):
        s = parquet.metadata.row_group(i).column(col).statistics
        if s is None or s.min <= week <= s.max:
            yield i


def extract_matched_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use Stage10B backward-as-of anchors; scan Stage05 for fixed-pair traces."""
    cached_a = OUT / "tables/11a_anchor_pairs.parquet"
    cached_t = OUT / "tables/11a_fixed_pair_timeline.parquet"
    if cached_a.exists() and cached_t.exists():
        old=pd.read_parquet(cached_a)
        if "anchor_source" in old.columns:
            return old, pd.read_parquet(cached_t)
    cohort = pd.read_parquet(BASE / "10_dynamic_rule_cruise/focal_dynamic_states.parquet")
    assert len(cohort) == 8196 and cohort.window_id.is_unique
    if (cohort.archive_week != WEEK).any() or (cohort.anchor_relative_seconds > 0).any():
        raise ValueError("Unexpected matched cohort scope or pre-event anchor")
    snapshot=pd.read_parquet(BASE/"10_dynamic_rule_cruise/snapshot_reconstruction.parquet",
        columns=["window_id","flight_key","latitude","longitude","altitude_ft","ground_speed_ms",
                 "track_deg","vertical_rate_fpm","is_focal","source_timestamp","anchor_timestamp","age_s"])
    if (snapshot.source_timestamp>snapshot.anchor_timestamp+1e-6).any() or snapshot.age_s.gt(15+1e-6).any():
        raise ValueError("Stage10B snapshot violates backward-as-of provenance")
    if snapshot.groupby("window_id").is_focal.sum().ne(1).any():
        raise ValueError("Stage10B focal identity is not unique")
    anchor_rows=[]
    for window,g in snapshot.groupby("window_id",sort=False):
        f=g.loc[g.is_focal].iloc[0]
        peers=g.loc[~g.is_focal].copy()
        row={"window_id":window,"time_epoch":float(f.source_timestamp),
             "focal_latitude":float(f.latitude),"focal_longitude":float(f.longitude),
             "focal_altitude_ft":float(f.altitude_ft),"focal_age_s":float(f.age_s),
             "other_flight_key":None,"other_source_epoch":np.nan,"other_age_s":np.nan,
             "horizontal_separation_nm":np.nan,"vertical_separation_ft":np.nan,
             "horizontal_closing_rate_nm_min":np.nan,"vertical_closing_rate_fpm":np.nan,
             "cpa_nm":np.nan,"tcpa_min":np.nan,"predicted_min_vertical_sep_ft":np.nan,
             "anchor_source":"STAGE10B_BACKWARD_ASOF_NATIVE"}
        if len(peers):
            lat0=np.deg2rad(float(f.latitude));lon0=np.deg2rad(float(f.longitude))
            lat=np.deg2rad(peers.latitude.to_numpy(float));lon=np.deg2rad(peers.longitude.to_numpy(float))
            arg=np.sin((lat-lat0)/2)**2+np.cos(lat0)*np.cos(lat)*np.sin((lon-lon0)/2)**2
            distance=2*3440.065*np.arcsin(np.sqrt(np.clip(arg,0,1)))
            peers["distance_nm"]=distance
            n=peers.sort_values(["distance_nm","flight_key"],kind="mergesort").iloc[0]
            dx=(float(n.longitude)-float(f.longitude))*60*math.cos(math.radians((float(n.latitude)+float(f.latitude))/2))
            dy=(float(n.latitude)-float(f.latitude))*60
            fa=math.radians(float(f.track_deg));na=math.radians(float(n.track_deg))
            fs=float(f.ground_speed_ms)*1.94384449;ns=float(n.ground_speed_ms)*1.94384449
            rvx=(ns*math.sin(na)-fs*math.sin(fa))/60
            rvy=(ns*math.cos(na)-fs*math.cos(fa))/60
            rv2=rvx*rvx+rvy*rvy
            tcpa=-(dx*rvx+dy*rvy)/rv2 if rv2>1e-12 else np.nan
            cpa=math.hypot(dx+rvx*max(tcpa,0),dy+rvy*max(tcpa,0)) if np.isfinite(tcpa) else np.nan
            signed_v=float(n.altitude_ft)-float(f.altitude_ft)
            dvr=float(n.vertical_rate_fpm)-float(f.vertical_rate_fpm)
            h=float(n.distance_nm)
            row.update({"other_flight_key":str(n.flight_key),"other_source_epoch":float(n.source_timestamp),
                        "other_age_s":float(n.age_s),"horizontal_separation_nm":h,
                        "vertical_separation_ft":abs(signed_v),
                        "horizontal_closing_rate_nm_min":-(dx*rvx+dy*rvy)/h if h>0 else np.nan,
                        "vertical_closing_rate_fpm":-np.sign(signed_v)*dvr,
                        "cpa_nm":cpa,"tcpa_min":tcpa,
                        "predicted_min_vertical_sep_ft":abs(signed_v+dvr*max(tcpa,0)) if np.isfinite(tcpa) else np.nan})
        anchor_rows.append(row)
    anchor=pd.DataFrame(anchor_rows)
    ids=set(cohort.window_id.astype(str))
    if set(anchor.window_id.astype(str))!=ids:
        raise ValueError("Matched snapshot coverage differs from cohort")
    source = pq.ParquetFile(BASE / "05_features/primary_interaction_features_20161015_20161021.parquet")
    anchor = anchor.merge(cohort[["window_id", "ATC_INTERVENTION", "clearance_id", "focal_flight_key",
                                  "event_epoch", "focal_dynamic_state", "matched_pair_id", "anchor_relative_seconds"]],
                          on="window_id", validate="one_to_one")
    anchor["relative_seconds"]=anchor.time_epoch-anchor.event_epoch
    if (anchor.time_epoch > anchor.event_epoch + 1e-6).any() or (anchor.other_source_epoch.dropna()>anchor.loc[anchor.other_source_epoch.notna(),"event_epoch"]+1e-6).any():
        raise ValueError("Future state in matched anchor")
    events = pd.read_parquet(BASE / "03_clearances/scat_clearances.parquet",
                             columns=["clearance_id", "clearance_type"])
    anchor = anchor.merge(events, on="clearance_id", how="left", validate="many_to_one")
    anchor.loc[anchor.ATC_INTERVENTION.eq(0), "clearance_type"] = "CONTROL"
    anchor["pair_quality"] = np.where(anchor.tcpa_min.between(1/60, 30) & anchor.cpa_nm.ge(0), "FUTURE_GE_1S_LE_30MIN", "AUDIT_ONLY")
    save(anchor, "tables/11a_anchor_pairs.parquet")
    selected = dict(zip(anchor.loc[anchor.other_flight_key.notna(),"window_id"].astype(str),
                        anchor.loc[anchor.other_flight_key.notna(),"other_flight_key"].astype(str)))
    timeline = {}
    for i in range(source.num_row_groups):
        f = source.read_row_group(i, columns=PAIR_FIELDS).to_pandas()
        f = f.loc[f.window_id.isin(selected) & f.relative_seconds.between(-300, 300)].copy()
        if f.empty:
            continue
        f = f.loc[f.other_flight_key.astype(str).eq(f.window_id.map(selected).astype(str))].copy()
        if f.empty:
            continue
        f["relative_bin_s"] = (np.rint(f.relative_seconds.to_numpy(float) / 30) * 30).astype(int)
        f["bin_error_s"] = (f.relative_seconds - f.relative_bin_s).abs()
        f.sort_values(["window_id", "relative_bin_s", "bin_error_s", "time_epoch"], inplace=True)
        for row in f.drop_duplicates(["window_id", "relative_bin_s"]).to_dict("records"):
            key = (str(row["window_id"]), int(row["relative_bin_s"]))
            old = timeline.get(key)
            if old is None or (row["bin_error_s"], row["time_epoch"]) < (old["bin_error_s"], old["time_epoch"]):
                timeline[key] = row
    trace = pd.DataFrame(timeline.values())
    # The Stage05 common 5 s bucket may include a neighbour native state after
    # the authorization for rows nearest t=0.  Replace that entire bin with
    # the independently validated backward-as-of anchor from Stage10B.
    trace=trace.loc[trace.relative_bin_s.ne(0)].copy()
    safe_zero=anchor.loc[anchor.other_flight_key.notna(),PAIR_FIELDS].copy()
    safe_zero["relative_bin_s"]=0
    safe_zero["bin_error_s"]=safe_zero.relative_seconds.abs()
    trace=pd.concat([trace,safe_zero],ignore_index=True)
    trace = trace.merge(anchor[["window_id", "ATC_INTERVENTION", "clearance_type", "focal_dynamic_state",
                                "focal_flight_key", "event_epoch"]], on="window_id", validate="many_to_one")
    if trace.duplicated(["window_id", "relative_bin_s"]).any():
        raise ValueError("Repeated trace bin")
    save(trace, "tables/11a_fixed_pair_timeline.parquet")
    return anchor, trace


def _snapshot_geometry(f: pd.DataFrame) -> pd.DataFrame:
    """Nearest horizontal neighbour in each observed tick, including all altitudes."""
    rows = []
    earth_nm = 3440.065
    for tick, g in f.groupby("tick", sort=False):
        if len(g) < 2:
            continue
        lat = np.deg2rad(g.latitude.to_numpy(float))
        lon = np.deg2rad(g.longitude.to_numpy(float))
        xyz = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        tree = cKDTree(xyz)
        chord, idx = tree.query(xyz, k=2)
        neighbour = idx[:, 1]
        own = g.reset_index(drop=True)
        n = own.iloc[neighbour].reset_index(drop=True)
        keep = own.altitude_ft.between(29000, 41000).to_numpy()
        if not keep.any():
            continue
        a = own.loc[keep].reset_index(drop=True)
        b = n.loc[keep].reset_index(drop=True)
        h = 2*earth_nm*np.arcsin(np.clip(chord[keep, 1]/2, 0, 1))
        meanlat = np.deg2rad((a.latitude.to_numpy(float)+b.latitude.to_numpy(float))/2)
        dx = (b.longitude.to_numpy(float)-a.longitude.to_numpy(float))*60*np.cos(meanlat)
        dy = (b.latitude.to_numpy(float)-a.latitude.to_numpy(float))*60
        av = np.deg2rad(a.track_deg.to_numpy(float)); bv = np.deg2rad(b.track_deg.to_numpy(float))
        va = a.ground_speed_kt.to_numpy(float); vb = b.ground_speed_kt.to_numpy(float)
        rvx = (vb*np.sin(bv)-va*np.sin(av))/60
        rvy = (vb*np.cos(bv)-va*np.cos(av))/60
        rv2 = rvx**2+rvy**2
        tcpa = np.divide(-(dx*rvx+dy*rvy), rv2, out=np.full(len(a), np.nan), where=rv2>1e-12)
        cpa = np.hypot(dx+rvx*np.maximum(tcpa, 0),dy+rvy*np.maximum(tcpa, 0))
        signed_v = b.altitude_ft.to_numpy(float)-a.altitude_ft.to_numpy(float)
        dvr = b.vertical_rate_fpm.to_numpy(float)-a.vertical_rate_fpm.to_numpy(float)
        out = pd.DataFrame({"tick":tick, "flight_key":a.flight_key.to_numpy(),
                            "neighbour_key":b.flight_key.to_numpy(), "focal_source_epoch":a.time_epoch.to_numpy(float),
                            "neighbour_source_epoch":b.time_epoch.to_numpy(float),
                            "focal_age_s":tick-a.time_epoch.to_numpy(float), "neighbour_age_s":tick-b.time_epoch.to_numpy(float),
                            "horizontal_nm":h, "vertical_ft":abs(signed_v),
                            "horizontal_closing_nm_min":np.divide(-(dx*rvx+dy*rvy),np.hypot(dx,dy),out=np.full(len(a),np.nan),where=np.hypot(dx,dy)>0),
                            "vertical_closing_fpm":-np.sign(signed_v)*dvr,
                            "cpa_nm":cpa,"tcpa_min":tcpa,
                            "vertical_at_cpa_ft":np.abs(signed_v+dvr*np.maximum(tcpa,0)),
                            "source_archive":a.source_archive.to_numpy(), "source_plot_index":a.plot_index.to_numpy()})
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_snapshots(step: int = 30, max_age: float = 15.0) -> tuple[pd.DataFrame, dict]:
    """One last native state per aircraft at grid tick, source time <= tick."""
    source = pq.ParquetFile(BASE / "02_traffic/scat_states.parquet")
    parts = []
    n_week = 0
    for i in _week_groups(source, WEEK):
        f = source.read_row_group(i, columns=STATE_FIELDS).to_pandas()
        f = f.loc[f.archive_week.eq(WEEK)]
        n_week += len(f)
        f = f.loc[f.time_epoch.notna() & f.latitude.between(-90,90) & f.longitude.between(-180,180)
                  & f.altitude_ft.between(-1000,60000) & f.flight_key.notna()].copy()
        if f.empty:
            continue
        f["tick"] = snapshot_tick(f.time_epoch.to_numpy(float),step)
        f = f.loc[(f.tick-f.time_epoch).between(0,max_age)]
        if len(f):
            parts.append(f)
    states = pd.concat(parts, ignore_index=True)
    states.sort_values(["tick","flight_key","time_epoch","plot_index"], inplace=True, kind="mergesort")
    states.drop_duplicates(["tick","flight_key"], keep="last", inplace=True)
    states.reset_index(drop=True,inplace=True)
    diag = {"pilot_native_rows_scanned":n_week, "snapshots_all_altitudes":len(states),
            "grid_seconds":step,"max_state_age_seconds":max_age,
            "focal_cruise_snapshots":int(states.altitude_ft.between(29000,41000).sum()),
            "unique_focal_flights":int(states.loc[states.altitude_ft.between(29000,41000),"flight_key"].nunique())}
    return states, diag


def build_population(step: int = 30, max_age: float = 15.0) -> tuple[pd.DataFrame, dict]:
    cached = OUT / "tables/11b_population_risk_set.parquet"
    diagnostic = OUT / "diagnostics/11b_population_audit.json"
    if step == 30 and max_age == 15 and cached.exists() and diagnostic.exists():
        old_diag=json.loads(diagnostic.read_text())
        if "event_duplicates_same_flight_tick_eligible" in old_diag:
            return pd.read_parquet(cached), old_diag
    states, diag = _load_snapshots(step,max_age)
    geom = _snapshot_geometry(states)
    focal = states.loc[states.altitude_ft.between(29000,41000),
                       ["tick","flight_key","altitude_ft","time_epoch","source_archive","plot_index"]].copy()
    focal.rename(columns={"time_epoch":"focal_native_epoch","source_archive":"focal_native_archive",
                          "plot_index":"focal_native_plot_index"},inplace=True)
    risk = focal.merge(geom,on=["tick","flight_key"],how="left",validate="one_to_one")
    risk["focal_source_epoch"] = risk.focal_source_epoch.fillna(risk.focal_native_epoch)
    risk["source_archive"] = risk.source_archive.fillna(risk.focal_native_archive)
    risk["source_plot_index"] = risk.source_plot_index.fillna(risk.focal_native_plot_index)
    risk["focal_age_s"] = risk.focal_age_s.fillna(risk.tick-risk.focal_native_epoch)
    risk["exposure_hours"] = step/3600
    risk["has_neighbour"] = risk.neighbour_key.notna()
    events = pd.read_parquet(BASE / "03_clearances/scat_clearances.parquet",
                             columns=["clearance_id","flight_key","archive_week","clearance_epoch","clearance_type","ATC_INTERVENTION"],
                             filters=[("archive_week","==",WEEK)])
    events = events.loc[events.ATC_INTERVENTION.eq(1)].copy()
    events["tick"] = event_tick(events.clearance_epoch.to_numpy(float),step)
    event_risk = events.merge(risk[["tick","flight_key","altitude_ft","neighbour_key","horizontal_nm","vertical_ft"]],
                              on=["tick","flight_key"],how="left",validate="many_to_one",indicator=True)
    diag.update({"pilot_actionable_events":len(events),
                 "events_with_eligible_cruise_snapshot":int(event_risk._merge.eq("both").sum()),
                 "events_with_eligible_neighbour":int(event_risk.neighbour_key.notna().sum()),
                 "focal_snapshots":len(risk),"focal_hours":float(risk.exposure_hours.sum()),
                 "focal_snapshots_with_neighbour":int(risk.has_neighbour.sum()),
                 "neighbour_hours":float(risk.loc[risk.has_neighbour,"exposure_hours"].sum()),
                 "event_duplicates_same_flight_tick_all_altitudes":int(events.duplicated(["flight_key","tick"]).sum()),
                 "event_duplicates_same_flight_tick_eligible":int(event_risk.loc[event_risk._merge.eq("both")].duplicated(["flight_key","tick"]).sum())})
    event_risk = event_risk.loc[event_risk._merge.eq("both")].drop(columns="_merge")
    event_risk["event_count"] = 1
    counts = event_risk.groupby(["tick","flight_key"],as_index=False).event_count.sum()
    risk = risk.merge(counts,on=["tick","flight_key"],how="left",validate="one_to_one")
    risk["event_count"] = risk.event_count.fillna(0).astype(int)
    for t in ("LEVEL","HEADING","SPEED"):
        x = event_risk.loc[event_risk.clearance_type.eq(t)].groupby(["tick","flight_key"]).size().rename("n")
        risk[f"event_{t.lower()}"] = pd.MultiIndex.from_frame(risk[["tick","flight_key"]]).map(x).fillna(0).astype(int)
    if int(risk.event_count.sum()) != len(event_risk):
        raise ValueError("Population event totals mismatch")
    if (risk.focal_source_epoch > risk.tick+1e-6).any() or (risk.neighbour_source_epoch.dropna() > risk.loc[risk.neighbour_source_epoch.notna(),"tick"]+1e-6).any():
        raise ValueError("Future state in population snapshot")
    if ((event_risk.clearance_epoch <= event_risk.tick) | (event_risk.clearance_epoch > event_risk.tick+step)).any():
        raise ValueError("Event interval misassigned")
    if step == 30 and max_age == 15:
        save(risk,"tables/11b_population_risk_set.parquet")
        save(event_risk,"tables/11b_eligible_events.parquet")
        diagnostic.parent.mkdir(parents=True,exist_ok=True)
        diagnostic.write_text(json.dumps(diag,indent=2)+"\n")
    return risk,diag


def binned_rate(frame: pd.DataFrame, x: str, edges: np.ndarray, y: str | None = None,
                yedges: np.ndarray | None = None, event_col: str = "event_count") -> pd.DataFrame:
    """All bins, including zero event bins; out-of-range counts stay in diagnostics."""
    valid = frame[x].notna() & np.isfinite(frame[x])
    if y:
        valid &= frame[y].notna() & np.isfinite(frame[y])
    f = frame.loc[valid].copy()
    f["xb"] = pd.cut(f[x],edges,right=False,include_lowest=True)
    if y:
        f["yb"] = pd.cut(f[y],yedges,right=False,include_lowest=True)
        group = f.groupby(["xb","yb"],observed=False)
        keys=["xb","yb"]
    else:
        group = f.groupby("xb",observed=False)
        keys=["xb"]
    table = group.agg(exposure_hours=("exposure_hours","sum"),events=(event_col,"sum"),
                      snapshots=("flight_key","size"),flights=("flight_key","nunique")).reset_index()
    # Influence-function uncertainty clustered by focal flight.  The Garwood
    # count interval is retained as an envelope where counts are very sparse.
    cluster=f.groupby(keys+["flight_key"],observed=True).agg(
        cluster_hours=("exposure_hours","sum"),cluster_events=(event_col,"sum")).reset_index()
    totals=table[keys+["events","exposure_hours"]].copy()
    totals["raw_rate"]=np.divide(totals.events,totals.exposure_hours,
                                  out=np.zeros(len(totals)),where=totals.exposure_hours.gt(0))
    cluster=cluster.merge(totals[keys+["raw_rate"]],on=keys,how="left",validate="many_to_one")
    cluster["influence_sq"]=(cluster.cluster_events-cluster.raw_rate*cluster.cluster_hours)**2
    influence=cluster.groupby(keys,observed=True).influence_sq.sum().reset_index()
    table=table.merge(influence,on=keys,how="left",validate="one_to_one")
    table["x_low"] = table.xb.map(lambda z: z.left).astype(float)
    table["x_high"] = table.xb.map(lambda z: z.right).astype(float)
    if y:
        table["y_low"] = table.yb.map(lambda z: z.left).astype(float)
        table["y_high"] = table.yb.map(lambda z: z.right).astype(float)
        table.drop(columns=["xb","yb"],inplace=True)
    else:
        table.drop(columns="xb",inplace=True)
    e = table.events.to_numpy(float); d = table.exposure_hours.to_numpy(float)
    table["rate_per_1000h"] = np.divide(1000*e,d,out=np.full(len(e),np.nan),where=d>0)
    table["ci_low_per_1000h"] = np.divide(1000*0.5*chi2.ppf(.025,2*e),d,out=np.full(len(e),np.nan),where=(d>0)&(e>0))
    table.loc[(d>0)&(e==0),"ci_low_per_1000h"] = 0
    table["ci_high_per_1000h"] = np.divide(1000*0.5*chi2.ppf(.975,2*(e+1)),d,out=np.full(len(e),np.nan),where=d>0)
    fcount=table.flights.to_numpy(float)
    correction=np.divide(fcount,fcount-1,out=np.ones(len(fcount)),where=fcount>1)
    se=np.divide(1000*np.sqrt(table.influence_sq.fillna(0).to_numpy(float)*correction),d,
                 out=np.full(len(d),np.nan),where=d>0)
    table["flight_cluster_se_per_1000h"]=se
    table["ci_low_per_1000h"]=np.minimum(table.ci_low_per_1000h,np.maximum(0,table.rate_per_1000h-1.96*se))
    table["ci_high_per_1000h"]=np.maximum(table.ci_high_per_1000h,table.rate_per_1000h+1.96*se)
    table.drop(columns="influence_sq",inplace=True)
    table["supported"] = (d>=10)&(e>=3)&(table.flights>=20)
    return table


def build_rates(risk: pd.DataFrame) -> dict:
    have = risk.loc[risk.has_neighbour].copy()
    output = {}
    for hn,vf in [(1,100),(2,250),(2,500),(5,500)]:
        name=f"h{hn}_v{vf}"
        t=binned_rate(have,"horizontal_nm",np.arange(0,50+hn,hn),"vertical_ft",np.arange(0,6000+vf,vf))
        save(t,f"tables/11b_rate_map_{name}.csv");output[name]=t
    for hn in (1,2,5):
        t=binned_rate(have,"horizontal_nm",np.arange(0,50+hn,hn))
        save(t,f"tables/11b_horizontal_rate_{hn}nm.csv");output[f"horizontal_{hn}"]=t
    for vf in (100,250,500):
        t=binned_rate(have,"vertical_ft",np.arange(0,6000+vf,vf))
        save(t,f"tables/11b_vertical_rate_{vf}ft.csv");output[f"vertical_{vf}"]=t
    for tname in ("level","heading","speed"):
        t=binned_rate(have,"horizontal_nm",np.arange(0,52,2),event_col=f"event_{tname}")
        save(t,f"tables/11b_horizontal_rate_{tname}.csv");output[tname]=t
    have["motion"] = np.where(have.horizontal_closing_nm_min>0,"CONVERGING","DIVERGING_OR_TANGENTIAL")
    for motion,g in have.groupby("motion"):
        t=binned_rate(g,"horizontal_nm",np.arange(0,52,2))
        save(t,f"tables/11b_horizontal_rate_{motion.lower()}.csv");output[motion]=t
    quality = have.tcpa_min.between(1/60,30) & have.cpa_nm.between(0,50) & have.vertical_at_cpa_ft.notna()
    pred=have.loc[quality].copy()
    t=binned_rate(pred,"tcpa_min",np.arange(0,31,1),"cpa_nm",np.arange(0,51,2))
    save(t,"tables/11b_cpa_tcpa_rate.csv");output["cpa_tcpa"]=t
    for field,edges in (("cpa_nm",np.arange(0,51,2)),("tcpa_min",np.arange(0,31,1))):
        t=binned_rate(pred,field,edges)
        save(t,f"tables/11b_{field}_rate.csv");output[field]=t
    audit={"nearest_horizontal_not_in_0_50_nm":int((have.horizontal_nm>=50).sum()),
           "nearest_vertical_not_in_0_6000_ft":int((have.vertical_ft>=6000).sum()),
           "events_nearest_horizontal_not_in_0_50_nm":int(have.loc[have.horizontal_nm>=50,"event_count"].sum()),
           "events_nearest_vertical_not_in_0_6000_ft":int(have.loc[have.vertical_ft>=6000,"event_count"].sum()),
           "cpa_tcpa_quality_snapshots":len(pred),"cpa_tcpa_quality_events":int(pred.event_count.sum()),
           "subsecond_positive_tcpa_snapshots":int(have.tcpa_min.between(0,1/60,inclusive="left").sum()),
           "negative_tcpa_snapshots":int(have.tcpa_min.lt(0).sum()),
           "missing_tcpa_snapshots":int(have.tcpa_min.isna().sum())}
    (OUT/"diagnostics/11b_rate_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    return output


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--matched-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    a,t=extract_matched_pairs()
    print(f"11A: {len(a)} anchor pairs; {len(t)} fixed-pair time rows",flush=True)
    if not args.matched_only:
        risk,diag=build_population()
        build_rates(risk)
        print("11B:",json.dumps(diag),flush=True)


if __name__=="__main__":
    main()
