"""Tabular tests for Stage 11D. Run after stage11d_core for both weeks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from stage11d_core import OUT, PILOT, REPLICATION, CRITERIA, PARAMS, pair_group, vertical_status


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT/"tables"/name,index=False)


def rates(f: pd.DataFrame, keys: list[str], event_col: str="event_count", cluster: bool=True) -> pd.DataFrame:
    valid=f.loc[f[keys].notna().all(axis=1)].copy()
    g=valid.groupby(keys,observed=True,dropna=False)
    t=g.agg(events=(event_col,"sum"),exposure_hours=("exposure_hours","sum"),
            snapshots=("flight_key","size"),flights=("flight_key","nunique")).reset_index()
    e=t.events.to_numpy(float);d=t.exposure_hours.to_numpy(float)
    t["rate_per_1000h"]=np.divide(1000*e,d,out=np.full(len(e),np.nan),where=d>0)
    lo=np.where(e>0,.5*chi2.ppf(.025,2*e),0)
    hi=.5*chi2.ppf(.975,2*(e+1))
    t["garwood_low_per_1000h"]=np.divide(1000*lo,d,out=np.full(len(e),np.nan),where=d>0)
    t["garwood_high_per_1000h"]=np.divide(1000*hi,d,out=np.full(len(e),np.nan),where=d>0)
    if cluster and len(t):
        by=valid.groupby(keys+["flight_key"],observed=True,dropna=False).agg(
            cluster_events=(event_col,"sum"),cluster_hours=("exposure_hours","sum")).reset_index()
        by=by.merge(t[keys+["rate_per_1000h"]],on=keys,validate="many_to_one")
        by["sq"]=(by.cluster_events-by.rate_per_1000h*by.cluster_hours/1000)**2
        var=by.groupby(keys,observed=True).sq.sum().rename("influence_sq").reset_index()
        t=t.merge(var,on=keys,how="left",validate="one_to_one")
        correction=np.where(t.flights>1,t.flights/np.maximum(t.flights-1,1),1)
        t["flight_cluster_se_per_1000h"]=1000*np.sqrt(t.influence_sq.fillna(0)*correction)/t.exposure_hours
        t["ci_low_per_1000h"]=np.minimum(t.garwood_low_per_1000h,
             np.maximum(0,t.rate_per_1000h-1.96*t.flight_cluster_se_per_1000h))
        t["ci_high_per_1000h"]=np.maximum(t.garwood_high_per_1000h,
             t.rate_per_1000h+1.96*t.flight_cluster_se_per_1000h)
        t.drop(columns="influence_sq",inplace=True)
    else:
        t["flight_cluster_se_per_1000h"]=np.nan
        t["ci_low_per_1000h"]=t.garwood_low_per_1000h
        t["ci_high_per_1000h"]=t.garwood_high_per_1000h
    t["supported"]=(t.exposure_hours>=10)&(t.events>=3)&(t.flights>=20)
    return t


def bins(f: pd.DataFrame, width_h: int=2, width_v: int=250) -> pd.DataFrame:
    f=f.copy()
    f["h_bin_nm"]=(np.floor(f.horizontal_nm/width_h)*width_h).where(f.horizontal_nm.between(0,50,inclusive="left"))
    f["v_bin_ft"]=(np.floor(f.vertical_ft/width_v)*width_v).where(f.vertical_ft.between(0,6000,inclusive="left"))
    return f


def analyze_week(week: str) -> None:
    f=pd.read_parquet(OUT/"tables"/f"{week}_criterion_snapshots.parquet")
    if f.duplicated(["tick","flight_key","criterion"]).any(): raise ValueError("Duplicate criterion focal tick")
    if len(f)!=len(CRITERIA)*f.loc[f.criterion.eq(CRITERIA[0])].shape[0]: raise ValueError("Unequal risk sets")
    totals=f.groupby("criterion").agg(snapshots=("flight_key","size"),hours=("exposure_hours","sum"),
           events=("event_count","sum"),selected=("neighbour_key","count")).reset_index()
    if totals.snapshots.nunique()!=1 or totals.events.nunique()!=1 or totals.hours.nunique()!=1:
        raise ValueError("Different focal exposure/events across criteria")
    write(totals,f"{week}_selection_coverage.csv")
    f["focal_status"]=vertical_status(f.focal_vrate_fpm.to_numpy(float))
    f["neighbour_status"]=vertical_status(f.neighbour_vrate_fpm.to_numpy(float))
    f["pair_group"]=pair_group(f.focal_status.to_numpy(),f.neighbour_status.to_numpy())
    f["motion_group"]=np.where(f.pair_group.eq("BOTH_LEVEL"),"BOTH_LEVEL",
                         np.where(f.pair_group.eq("UNKNOWN"),"UNKNOWN","MOVING"))
    f=bins(f)
    have=f.loc[f.neighbour_key.notna()].copy()
    write(rates(have,["criterion","h_bin_nm"]),f"{week}_horizontal_rates.csv")
    write(rates(have,["criterion","v_bin_ft"]),f"{week}_vertical_rates.csv")
    write(rates(have,["criterion","h_bin_nm","v_bin_ft"],cluster=False),f"{week}_rate_maps.csv")
    selected=have.loc[have.criterion.eq("NEAREST_HORIZONTAL")].copy()
    write(rates(selected,["pair_group","v_bin_ft"]),f"{week}_state_vertical_rates.csv")
    write(rates(selected,["motion_group","v_bin_ft"]),f"{week}_motion_vertical_rates.csv")
    write(rates(selected,["motion_group","h_bin_nm","v_bin_ft"],cluster=False),f"{week}_motion_rate_maps.csv")
    for typ in ("level","heading","speed"):
        t=rates(selected,["motion_group","v_bin_ft"],event_col=f"event_{typ}")
        t["clearance_type"]=typ.upper()
        if typ=="level": by_type=[t]
        else: by_type.append(t)
    write(pd.concat(by_type,ignore_index=True),f"{week}_type_motion_vertical_rates.csv")
    for tolerance in (100,200,300):
        sf=vertical_status(selected.focal_vrate_fpm.to_numpy(float),tolerance)
        sn=vertical_status(selected.neighbour_vrate_fpm.to_numpy(float),tolerance)
        g=selected.copy();g["pair_group"]=pair_group(sf,sn)
        t=g.groupby("pair_group").agg(snapshots=("flight_key","size"),events=("event_count","sum"),
                hours=("exposure_hours","sum")).reset_index()
        t["tolerance_fpm"]=tolerance
        if tolerance==100: tolerances=[t]
        else:tolerances.append(t)
    write(pd.concat(tolerances,ignore_index=True),f"{week}_level_tolerance_sensitivity.csv")
    for width in (100,250,500):
        x=selected.copy();x["v_bin_ft"]=(np.floor(x.vertical_ft/width)*width).where(x.vertical_ft.between(0,6000,inclusive="left"))
        t=rates(x,["v_bin_ft"]);t["bin_width_ft"]=width
        if width==100: widths=[t]
        else:widths.append(t)
    write(pd.concat(widths,ignore_index=True),f"{week}_vertical_bin_sensitivity.csv")
    phase=selected.loc[selected.vertical_ft.between(0,6000,inclusive="left")].copy()
    mod=phase.vertical_ft.to_numpy(float)%1000
    phase["phase_group"]=np.where(np.minimum(mod,1000-mod)<=125,"NEAR_1000_LEVEL",
                             np.where(np.abs(mod-500)<=125,"NEAR_500_OFFSET","OTHER"))
    write(rates(phase,["motion_group","phase_group"]),f"{week}_periodicity_decomposition.csv")
    summary=selected.groupby("pair_group").agg(snapshots=("flight_key","size"),events=("event_count","sum"),
        hours=("exposure_hours","sum"),flights=("flight_key","nunique")).reset_index()
    summary["rate_per_1000h"]=1000*summary.events/summary.hours
    write(summary,f"{week}_pair_group_summary.csv")
    if week==PILOT:
        p=f.pivot(index=["tick","flight_key"],columns="criterion",values="neighbour_key")
        matrix=[]
        for a in CRITERIA:
            for b in CRITERIA:
                same=p[a].notna()&p[b].notna()&p[a].eq(p[b])
                both=p[a].notna()&p[b].notna()
                matrix.append(dict(criterion_a=a,criterion_b=b,same=int(same.sum()),both_selected=int(both.sum()),
                                   concordance=float(same.sum()/both.sum()) if both.sum() else np.nan))
        write(pd.DataFrame(matrix),"neighbor_concordance.csv")
        a=f.loc[f.criterion.eq("NEAREST_HORIZONTAL"),["tick","flight_key","neighbour_key","horizontal_nm","vertical_ft","cpa_nm","tcpa_min","closing_nm_min","event_count"]]
        b=f.loc[f.criterion.eq("MINIMUM_FUTURE_CPA"),["tick","flight_key","neighbour_key","horizontal_nm","vertical_ft","cpa_nm","tcpa_min","closing_nm_min"]]
        d=a.merge(b,on=["tick","flight_key"],suffixes=("_nearest","_future"),validate="one_to_one")
        d=d.loc[d.neighbour_key_nearest.notna() & d.neighbour_key_future.notna()].copy()
        for c in ("horizontal_nm","vertical_ft","cpa_nm","tcpa_min","closing_nm_min"):
            d[f"delta_{c}"]=d[f"{c}_future"]-d[f"{c}_nearest"]
        d["same_neighbor"]=d.neighbour_key_nearest.eq(d.neighbour_key_future)
        d.to_parquet(OUT/"tables"/"pilot_nearest_vs_future_cpa.parquet",index=False,compression="zstd")
        d.loc[~d.same_neighbor].sort_values("delta_cpa_nm").head(200).to_csv(OUT/"examples"/"cpa_selection_discrepancies.csv",index=False)
    print(f"Analyzed {week}: {len(f)} criterion rows",flush=True)


def analyze_recurrence(week: str=PILOT) -> None:
    e=pd.read_parquet(OUT/"tables"/f"{week}_eligible_events.parquet")
    e=e.sort_values(["flight_key","clearance_epoch","clearance_id"],kind="mergesort").reset_index(drop=True)
    e["previous_gap_s"]=e.groupby("flight_key").clearance_epoch.diff()
    flights=e.groupby("flight_key").agg(events=("clearance_id","size"),level=("clearance_type",lambda x:int(x.eq("LEVEL").sum())),
                 heading=("clearance_type",lambda x:int(x.eq("HEADING").sum())),speed=("clearance_type",lambda x:int(x.eq("SPEED").sum()))).reset_index()
    write(flights,f"{week}_events_per_flight.csv")
    write(e[["clearance_id","flight_key","clearance_epoch","clearance_type","tick","previous_gap_s"]],f"{week}_event_gaps.csv")
    recurrence=[]
    for window in (30,60,120,300):
        counts=np.empty(len(e),dtype=int)
        for _,idx in e.groupby("flight_key",sort=False).indices.items():
            times=e.clearance_epoch.to_numpy(float)[idx]
            counts[idx]=np.arange(len(idx))-np.searchsorted(times,times-window,side="left")+1
        recurrence.append(dict(window_seconds=window,events_with_prior=int((counts>1).sum()),
                fraction_with_prior=float((counts>1).mean()),max_in_window=int(counts.max()),
                mean_in_window=float(counts.mean())))
    write(pd.DataFrame(recurrence),f"{week}_recurrence_windows.csv")
    episodes=[]
    for gap in (30,60,120,300):
        start=e.previous_gap_s.isna()|e.previous_gap_s.gt(gap)
        episodes.append(dict(gap_seconds=gap,episodes=int(start.sum()),events=len(e),
             events_per_episode=float(len(e)/start.sum()),fraction_events_continuing=float((~start).mean())))
        e[f"episode_start_{gap}s"]=start.astype(int)
    write(pd.DataFrame(episodes),f"{week}_clearance_episode_sensitivity.csv")
    e.to_parquet(OUT/"tables"/f"{week}_events_with_episodes.parquet",index=False,compression="zstd")
    risk=pd.read_parquet(OUT/"tables"/f"{week}_criterion_snapshots.parquet",
          columns=["tick","flight_key","criterion","horizontal_nm","vertical_ft","event_count",
                   "exposure_hours","focal_source_epoch"])
    risk=risk.loc[risk.criterion.eq("NEAREST_HORIZONTAL")].copy()
    # Archive-week labels are not exact UTC cutoffs. Censor only the final 30 s
    # interval if its outcome extends past the actual last observed source time.
    end_epoch=max(float(risk.focal_source_epoch.max()),float(e.clearance_epoch.max()))
    complete=risk.tick+30<=end_epoch
    pd.DataFrame([dict(week=week,last_observed_epoch=end_epoch,
        candidate_ticks=len(risk),right_censored_ticks=int((~complete).sum()),
        complete_30s_ticks=int(complete.sum()),
        eligible_events_on_right_censored_ticks=int(risk.loc[~complete,"event_count"].sum()))]).to_csv(
          OUT/"tables"/f"{week}_binary_horizon_censoring.csv",index=False)
    risk=risk.loc[complete].copy()
    starts=e.loc[e.episode_start_120s.eq(1)].groupby(["tick","flight_key"]).size().rename("episode_count")
    risk["episode_count"]=pd.MultiIndex.from_frame(risk[["tick","flight_key"]]).map(starts).fillna(0).astype(int)
    risk["binary_event_tick"]=risk.event_count.gt(0).astype(int)
    risk=bins(risk)
    for metric in ("event_count","episode_count","binary_event_tick"):
        for dim in ("h_bin_nm","v_bin_ft"):
            t=rates(risk,[dim],event_col=metric);t["metric"]=metric;t["dimension"]=dim
            if metric=="event_count" and dim=="h_bin_nm": curves=[t]
            else:curves.append(t)
    write(pd.concat(curves,ignore_index=True),f"{week}_event_episode_binary_curves.csv")
    for metric in ("event_count","episode_count","binary_event_tick"):
        t=rates(risk,["h_bin_nm","v_bin_ft"],event_col=metric,cluster=False)
        t["metric"]=metric
        if metric=="event_count": maps=[t]
        else:maps.append(t)
    write(pd.concat(maps,ignore_index=True),f"{week}_event_episode_binary_maps.csv")
    print(f"Recurrence {week}: {len(e)} eligible events",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["week","recurrence"]);p.add_argument("week",choices=[PILOT,REPLICATION]);a=p.parse_args()
    if a.action=="week":analyze_week(a.week)
    else:analyze_recurrence(a.week)
