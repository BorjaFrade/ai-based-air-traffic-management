"""Stage 11D: pre-event population snapshots and alternative pair selection.

All five criteria use the same focal ticks, native states, 15 s age limit and
100 NM candidate universe. Output is descriptive authorization intensity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage11_core import BASE, OUT as STAGE11, STATE_FIELDS, _week_groups, event_tick, snapshot_tick

OUT = BASE / "11d_validation"
PILOT = "20161015_20161021"
REPLICATION = "20161112_20161118"
PARAMS = dict(grid_seconds=30, max_age_seconds=15, cruise_min_ft=29000,
              cruise_max_ft=41000, candidate_radius_nm=100,
              future_cpa_min_seconds=1, future_cpa_max_minutes=30,
              level_tolerance_fpm=200, horizontal_bin_nm=2,
              vertical_bin_ft=250, exposure_hours_per_tick=30/3600)
CRITERIA = ("NEAREST_HORIZONTAL", "MINIMUM_FUTURE_CPA", "MINIMUM_3D_DISTANCE",
            "MAXIMUM_CLOSING", "NEAREST_VERTICAL")
FIELDS = ("week", "tick", "flight_key", "criterion", "neighbour_key",
          "horizontal_nm", "vertical_ft", "cpa_nm", "tcpa_min", "closing_nm_min",
          "focal_vrate_fpm", "neighbour_vrate_fpm", "focal_source_epoch",
          "neighbour_source_epoch", "event_count", "event_level", "event_heading",
          "event_speed", "exposure_hours")
OUTPUT_SCHEMA=pa.schema([(name, pa.int64() if name in ("tick","event_count","event_level","event_heading","event_speed")
                          else pa.string() if name in ("week","flight_key","criterion","neighbour_key")
                          else pa.float64()) for name in FIELDS])


def vertical_status(rate: np.ndarray, tolerance: float = 200) -> np.ndarray:
    v = np.asarray(rate, dtype=float)
    return np.select([~np.isfinite(v), v > tolerance, v < -tolerance],
                     ["UNKNOWN", "CLIMBING", "DESCENDING"], default="LEVEL_FLIGHT")


def pair_group(focal: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
    f = np.asarray(focal, dtype=str); n = np.asarray(neighbor, dtype=str)
    out = np.full(len(f), "UNKNOWN", dtype="U26")
    known = (f != "UNKNOWN") & (n != "UNKNOWN")
    both_level = known & (f == "LEVEL_FLIGHT") & (n == "LEVEL_FLIGHT")
    out[both_level] = "BOTH_LEVEL"
    for status, label in (("CLIMBING", "LEVEL_AND_CLIMB"), ("DESCENDING", "LEVEL_AND_DESCEND")):
        out[known & (((f == "LEVEL_FLIGHT") & (n == status)) |
                     ((n == "LEVEL_FLIGHT") & (f == status)))] = label
    out[known & (f == "CLIMBING") & (n == "CLIMBING")] = "BOTH_CLIMB"
    out[known & (f == "DESCENDING") & (n == "DESCENDING")] = "BOTH_DESCEND"
    out[known & (((f == "CLIMBING") & (n == "DESCENDING")) |
                 ((f == "DESCENDING") & (n == "CLIMBING")))] = "OPPOSITE_MOTION"
    return out


def load_snapshots(week: str) -> tuple[pd.DataFrame, dict]:
    source = pq.ParquetFile(BASE / "02_traffic/scat_states.parquet")
    parts=[]; native=0; scanned=0
    for i in _week_groups(source, week):
        scanned += 1
        f=source.read_row_group(i, columns=STATE_FIELDS).to_pandas()
        f=f.loc[f.archive_week.eq(week)]
        native += len(f)
        f=f.loc[f.time_epoch.notna() & f.latitude.between(-90,90) &
                f.longitude.between(-180,180) & f.altitude_ft.between(-1000,60000) &
                f.flight_key.notna()].copy()
        if f.empty: continue
        f["tick"]=snapshot_tick(f.time_epoch.to_numpy(float), PARAMS["grid_seconds"])
        f=f.loc[(f.tick-f.time_epoch).between(0,PARAMS["max_age_seconds"])]
        if len(f): parts.append(f)
    states=pd.concat(parts,ignore_index=True)
    states.sort_values(["tick","flight_key","time_epoch","plot_index"],kind="mergesort",inplace=True)
    duplicates=int(states.duplicated(["tick","flight_key"]).sum())
    states.drop_duplicates(["tick","flight_key"],keep="last",inplace=True)
    states.reset_index(drop=True,inplace=True)
    audit=dict(week=week, scanned_row_groups=scanned, native_rows=native,
               snapshots=len(states), duplicate_native_tick_flight_rows=duplicates,
               focal_snapshots=int(states.altitude_ft.between(29000,41000).sum()))
    return states,audit


def make_risk(states: pd.DataFrame, week: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if week == PILOT:
        risk=pd.read_parquet(STAGE11/"tables/11b_population_risk_set.parquet",
            columns=["tick","flight_key","event_count","event_level","event_heading","event_speed","exposure_hours"])
        native=states.loc[states.altitude_ft.between(29000,41000),["tick","flight_key"]]
        if len(risk)!=len(native) or not pd.MultiIndex.from_frame(risk[["tick","flight_key"]]).equals(pd.MultiIndex.from_frame(native[["tick","flight_key"]])):
            # Same keys are enough; Stage 11 and current state sort order can differ.
            if set(map(tuple,risk[["tick","flight_key"]].to_numpy())) != set(map(tuple,native.to_numpy())):
                raise ValueError("Pilot focal population differs from frozen Stage 11B")
    else:
        risk=states.loc[states.altitude_ft.between(29000,41000),["tick","flight_key"]].copy()
        risk["exposure_hours"]=PARAMS["exposure_hours_per_tick"]
    events=pd.read_parquet(BASE/"03_clearances/scat_clearances.parquet",
        columns=["clearance_id","flight_key","archive_week","clearance_epoch","clearance_type","ATC_INTERVENTION"])
    events=events.loc[events.archive_week.eq(week) & events.ATC_INTERVENTION.eq(1)].copy()
    events["tick"]=event_tick(events.clearance_epoch.to_numpy(float),30)
    events=events.merge(risk[["tick","flight_key"]],on=["tick","flight_key"],how="inner",validate="many_to_one")
    if week != PILOT:
        counts=events.groupby(["tick","flight_key"]).size().rename("event_count")
        risk["event_count"]=pd.MultiIndex.from_frame(risk[["tick","flight_key"]]).map(counts).fillna(0).astype(int)
        for t in ("LEVEL","HEADING","SPEED"):
            c=events.loc[events.clearance_type.eq(t)].groupby(["tick","flight_key"]).size()
            risk[f"event_{t.lower()}"]=pd.MultiIndex.from_frame(risk[["tick","flight_key"]]).map(c).fillna(0).astype(int)
    if risk.event_count.sum()!=len(events):
        raise ValueError("Event assignment differs from eligible risk set")
    if ((events.clearance_epoch <= events.tick) | (events.clearance_epoch > events.tick+30)).any():
        raise ValueError("Event uses future or wrong interval")
    audit=dict(week=week,risk_ticks=len(risk),focal_hours=float(risk.exposure_hours.sum()),
               eligible_events=len(events),event_ticks=int(risk.event_count.gt(0).sum()),
               multiple_events_same_tick=int((risk.event_count-1).clip(lower=0).sum()),
               flights=int(risk.flight_key.nunique()))
    return risk,events,audit


def select_neighbors(f: pd.DataFrame, focal_keys: np.ndarray) -> dict[str,dict[str,np.ndarray]]:
    """Vectorized all-pair geometry at one tick; tie break by sorted key."""
    f=f.sort_values("flight_key",kind="mergesort").reset_index(drop=True)
    n=len(f)
    if n<2: return {}
    keys=f.flight_key.astype(str).to_numpy()
    index=pd.Index(keys).get_indexer(focal_keys.astype(str))
    if (index<0).any(): raise ValueError("Focal aircraft missing from native tick")
    lat=np.deg2rad(f.latitude.to_numpy(float)); lon=np.deg2rad(f.longitude.to_numpy(float))
    dlat=lat[None,:]-lat[:,None]; dlon=lon[None,:]-lon[:,None]
    arg=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    h=2*3440.065*np.arcsin(np.sqrt(np.clip(arg,0,1)))
    alt=f.altitude_ft.to_numpy(float); vertical=np.abs(alt[None,:]-alt[:,None])
    dx=(f.longitude.to_numpy(float)[None,:]-f.longitude.to_numpy(float)[:,None])*60*np.cos((lat[None,:]+lat[:,None])/2)
    dy=(f.latitude.to_numpy(float)[None,:]-f.latitude.to_numpy(float)[:,None])*60
    track=np.deg2rad(f.track_deg.to_numpy(float)); speed=f.ground_speed_kt.to_numpy(float)
    vx=speed*np.sin(track)/60; vy=speed*np.cos(track)/60
    rvx=vx[None,:]-vx[:,None]; rvy=vy[None,:]-vy[:,None]
    dot=dx*rvx+dy*rvy; v2=rvx**2+rvy**2
    tcpa=np.divide(-dot,v2,out=np.full_like(dot,np.nan),where=v2>1e-12)
    cpa=np.hypot(dx+rvx*np.maximum(tcpa,0),dy+rvy*np.maximum(tcpa,0))
    closing=np.divide(-dot,np.hypot(dx,dy),out=np.full_like(dot,np.nan),where=np.hypot(dx,dy)>1e-9)
    valid=np.isfinite(h)&(h<=PARAMS["candidate_radius_nm"])
    np.fill_diagonal(valid,False)
    scores={
      "NEAREST_HORIZONTAL":(h,valid),
      "MINIMUM_FUTURE_CPA":(cpa,valid & np.isfinite(cpa)&(tcpa>=1/60)&(tcpa<=30)),
      "MINIMUM_3D_DISTANCE":(np.hypot(h,vertical/6076.12),valid),
      "MAXIMUM_CLOSING":(-closing,valid & np.isfinite(closing)&(closing>0)),
      "NEAREST_VERTICAL":(vertical+1e-6*h,valid),
    }
    out={}
    for criterion,(score,ok) in scores.items():
        sub=np.where(ok[index],score[index],np.inf)
        j=np.argmin(sub,axis=1); selected=np.isfinite(sub[np.arange(len(index)),j])
        j=np.where(selected,j,-1)
        def take(matrix):
            x=np.full(len(index),np.nan)
            x[selected]=matrix[index[selected],j[selected]]
            return x
        out[criterion]=dict(neighbour_key=np.where(selected,keys[np.maximum(j,0)],None),
                            horizontal_nm=take(h),vertical_ft=take(vertical),
                            cpa_nm=take(cpa),tcpa_min=take(tcpa),closing_nm_min=take(closing),
                            neighbour_vrate_fpm=np.where(selected,f.vertical_rate_fpm.to_numpy(float)[np.maximum(j,0)],np.nan),
                            neighbour_source_epoch=np.where(selected,f.time_epoch.to_numpy(float)[np.maximum(j,0)],np.nan),
                            focal_vrate_fpm=f.vertical_rate_fpm.to_numpy(float)[index],
                            focal_source_epoch=f.time_epoch.to_numpy(float)[index])
    return out


def build_week(week: str, force: bool=False) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ("figures/01_vertical_periodicity","figures/02_neighbour_selection",
              "figures/03_clearance_recurrence","figures/04_week_reproducibility",
              "figures/05_episode_audit","tables","diagnostics","examples","reports"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    path=OUT/"tables"/f"{week}_criterion_snapshots.parquet"
    audit_path=OUT/"diagnostics"/f"{week}_coverage.json"
    if path.exists() and audit_path.exists() and not force:
        print(f"CACHED {path}",flush=True);return
    if path.exists(): path.unlink()
    states,source_audit=load_snapshots(week)
    risk,events,risk_audit=make_risk(states,week)
    events.to_parquet(OUT/"tables"/f"{week}_eligible_events.parquet",index=False,compression="zstd")
    risk.to_parquet(OUT/"tables"/f"{week}_risk_base.parquet",index=False,compression="zstd")
    risk=risk.sort_values(["tick","flight_key"],kind="mergesort")
    state_groups={int(t):g for t,g in states.groupby("tick",sort=False)}
    writer=None; pieces=[]; selected_counts={c:0 for c in CRITERIA}; done=0
    try:
        for tick,g in risk.groupby("tick",sort=False):
            s=state_groups.get(int(tick))
            if s is None: raise ValueError("Risk tick absent from states")
            choices=select_neighbors(s,g.flight_key.to_numpy())
            for criterion in CRITERIA:
                z=choices.get(criterion)
                if z is None:
                    z={k:(np.full(len(g),None,dtype=object) if k=="neighbour_key" else np.full(len(g),np.nan))
                       for k in FIELDS if k not in ("week","tick","flight_key","criterion","event_count","event_level","event_heading","event_speed","exposure_hours")}
                selected_counts[criterion]+=int(pd.notna(z["neighbour_key"]).sum())
                frame=g[["tick","flight_key","event_count","event_level","event_heading","event_speed","exposure_hours"]].copy()
                frame["week"]=week;frame["criterion"]=criterion
                for key,value in z.items(): frame[key]=value
                pieces.append(frame[list(FIELDS)])
            done+=1
            if done%250==0:
                batch=pa.Table.from_pandas(pd.concat(pieces,ignore_index=True),preserve_index=False).cast(OUTPUT_SCHEMA)
                if writer is None: writer=pq.ParquetWriter(path,OUTPUT_SCHEMA,compression="zstd")
                writer.write_table(batch);pieces=[]
                if done%2500==0: print(week,"ticks",done,flush=True)
        if pieces:
            batch=pa.Table.from_pandas(pd.concat(pieces,ignore_index=True),preserve_index=False).cast(OUTPUT_SCHEMA)
            if writer is None: writer=pq.ParquetWriter(path,OUTPUT_SCHEMA,compression="zstd")
            writer.write_table(batch)
    finally:
        if writer is not None: writer.close()
    audit={**source_audit,**risk_audit,"grid_ticks":done,"selected_rows":selected_counts,"parameters":PARAMS}
    (OUT/"diagnostics"/f"{week}_coverage.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("week",choices=[PILOT,REPLICATION]);parser.add_argument("--force",action="store_true")
    args=parser.parse_args();build_week(args.week,args.force)
