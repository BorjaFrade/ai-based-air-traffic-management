"""Partition pair traces by identity and derive observation-contiguous encounters."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage12_audit_inputs import OUT,WEEKS
from stage12_build_encounters import CONFIG

BUCKETS=32
COLS=["pair_id","tick","a_key","b_key","a_flight_key","b_flight_key",
      "a_source_epoch","b_source_epoch","a_age_s","b_age_s","a_lat","a_lon","b_lat","b_lon",
      "a_alt_ft","b_alt_ft","a_speed_kt","b_speed_kt","a_track_deg","b_track_deg",
      "a_vrate_fpm","b_vrate_fpm","a_phase","b_phase","a_sector","b_sector",
      "horizontal_nm","signed_vertical_ft","vertical_ft","relative_east_nm","relative_north_nm",
      "relative_vx_nm_min","relative_vy_nm_min","relative_speed_kt","closing_nm_min",
      "crossing_angle_deg","tcpa_raw_min","cpa_unbounded_nm",
      "concurrent_neighbors_a","concurrent_neighbors_b"]


def partition(week:str,force:bool=False):
    source=OUT/"tables"/f"{week}_candidate_pairs_150nm.parquet"
    root=OUT/"episodes"/f"{week}_pair_buckets"
    root.mkdir(parents=True,exist_ok=True)
    ready=root/"_READY.json"
    if ready.exists() and not force:
        print("CACHED",root,flush=True);return
    writers={};counts=np.zeros(BUCKETS,dtype=np.int64)
    q=pq.ParquetFile(source)
    schema=pa.schema([q.schema_arrow.field(k) for k in COLS])
    try:
        for bi,batch in enumerate(q.iter_batches(batch_size=100_000,columns=COLS)):
            t=pa.Table.from_batches([batch]).cast(schema)
            ids=t.column("pair_id").to_pandas()
            bucket=(pd.util.hash_pandas_object(ids,index=False).to_numpy(dtype=np.uint64)%BUCKETS).astype(int)
            for b in np.unique(bucket):
                part=t.take(pa.array(np.flatnonzero(bucket==b)))
                if b not in writers:writers[b]=pq.ParquetWriter(root/f"bucket_{b:02d}.parquet",schema,compression="zstd")
                writers[b].write_table(part);counts[b]+=part.num_rows
            if bi%25==0:print(week,"partition batches",bi,"rows",int(counts.sum()),flush=True)
    finally:
        for writer in writers.values():writer.close()
    ready.write_text(json.dumps({"week":week,"buckets":BUCKETS,"rows":int(counts.sum()),
                                 "bucket_rows":counts.tolist(),"columns":COLS},indent=2)+"\n")


def segment_flags(d:pd.DataFrame,gap_s:int=60)->np.ndarray:
    same=d.pair_id.eq(d.pair_id.shift())
    gap=d.tick.diff().fillna(np.inf).to_numpy(float)
    jump=d.horizontal_nm.diff().abs().fillna(0).to_numpy(float)
    impossible=same.to_numpy()&(jump>30*np.maximum(gap,30)/30)
    return (~same.to_numpy())|(gap>gap_s)|impossible


def _episode_table(d:pd.DataFrame)->pd.DataFrame:
    g=d.groupby("episode_id",sort=False)
    first=g.first();last=g.last()
    agg=g.agg(n_samples=("tick","size"),min_horizontal_nm=("horizontal_nm","min"),
       max_horizontal_nm=("horizontal_nm","max"),min_vertical_ft=("vertical_ft","min"),
       median_horizontal_nm=("horizontal_nm","median"),median_vertical_ft=("vertical_ft","median"),
       median_closing_nm_min=("closing_nm_min","median"),
       median_crossing_angle_deg=("crossing_angle_deg","median"),
       max_concurrent_neighbors_a=("concurrent_neighbors_a","max"),
       max_concurrent_neighbors_b=("concurrent_neighbors_b","max"),
       max_state_age_s=("a_age_s","max"))
    agg["max_state_age_s"]=np.maximum(agg.max_state_age_s,g.b_age_s.max())
    imin=g.horizontal_nm.idxmin();ivmin=g.vertical_ft.idxmin()
    min_h=d.loc[imin.to_numpy(),["episode_id","tick","vertical_ft","horizontal_nm"]].set_index("episode_id")
    min_v=d.loc[ivmin.to_numpy(),["episode_id","tick","horizontal_nm"]].set_index("episode_id")
    out=agg.copy()
    for k in ("pair_id","a_key","b_key","a_flight_key","b_flight_key","a_phase","b_phase","a_sector","b_sector"):
        out[k]=first[k]
    for k in ("tick","horizontal_nm","vertical_ft","signed_vertical_ft","a_lat","a_lon","b_lat","b_lon","a_alt_ft","b_alt_ft",
              "a_speed_kt","b_speed_kt","a_track_deg","b_track_deg","a_vrate_fpm","b_vrate_fpm",
              "relative_east_nm","relative_north_nm","relative_vx_nm_min","relative_vy_nm_min",
              "tcpa_raw_min","cpa_unbounded_nm","closing_nm_min","crossing_angle_deg"):
        out[f"start_{k}"]=first[k]
    for k in ("tick","horizontal_nm","vertical_ft","signed_vertical_ft","a_alt_ft","b_alt_ft",
              "a_speed_kt","b_speed_kt","a_track_deg","b_track_deg"):
        out[f"end_{k}"]=last[k]
    out["time_min_horizontal_tick"]=min_h.tick
    out["vertical_at_min_horizontal_ft"]=min_h.vertical_ft
    out["time_min_vertical_tick"]=min_v.tick
    out["horizontal_at_min_vertical_nm"]=min_v.horizontal_nm
    out["observed_duration_s"]=out.end_tick-out.start_tick
    out["exposure_duration_s"]=out.observed_duration_s+30
    out["a_altitude_change_ft"]=out.end_a_alt_ft-out.start_a_alt_ft
    out["b_altitude_change_ft"]=out.end_b_alt_ft-out.start_b_alt_ft
    out["a_speed_change_kt"]=out.end_a_speed_kt-out.start_a_speed_kt
    out["b_speed_change_kt"]=out.end_b_speed_kt-out.start_b_speed_kt
    def angle_change(x,y):return ((x-y+180)%360)-180
    out["a_track_change_deg"]=angle_change(out.end_a_track_deg,out.start_a_track_deg)
    out["b_track_change_deg"]=angle_change(out.end_b_track_deg,out.start_b_track_deg)
    d["converging"]=d.closing_nm_min.gt(0).astype(int)
    d["diverging"]=d.closing_nm_min.lt(0).astype(int)
    out["converging_observed_s"]=g.converging.sum()*30
    out["diverging_observed_s"]=g.diverging.sum()*30
    out["fraction_converging"]=g.converging.mean()
    delta=d.groupby("episode_id").horizontal_nm.diff()
    trend=np.sign(delta)
    prior=trend.groupby(d.episode_id).shift()
    d["trend_switch"]=(trend.ne(0)&prior.ne(0)&trend.ne(prior)&prior.notna()).astype(int)
    out["trend_switches"]=g.trend_switch.sum()
    out["multiple_neighbours"]=((out.max_concurrent_neighbors_a>1)|(out.max_concurrent_neighbors_b>1))
    out["possible_radius_truncation_start"]=out.start_horizontal_nm>=95
    out["possible_radius_truncation_end"]=out.end_horizontal_nm>=95
    out["quality_for_structure"]=(out.n_samples>=4)&(out.observed_duration_s>=90)&out.max_state_age_s.le(15)
    return out.reset_index()


def segment_week(week:str,force:bool=False):
    root=OUT/"episodes"/f"{week}_pair_buckets"
    trace_root=OUT/"episodes"/f"{week}_episode_traces"
    trace_root.mkdir(parents=True,exist_ok=True)
    episode_root=OUT/"episodes"/f"{week}_episode_features"
    episode_root.mkdir(parents=True,exist_ok=True)
    done=OUT/"diagnostics"/f"{week}_segmentation_audit.json"
    if done.exists() and not force:
        print("CACHED",done,flush=True);return
    sensitivity=[];totals={"candidate_rows":0,"primary_rows":0,"episodes":0,
                          "physical_jump_splits":0,"duplicate_pair_ticks":0}
    for b in range(BUCKETS):
        path=root/f"bucket_{b:02d}.parquet"
        if not path.exists():continue
        d=pd.read_parquet(path)
        d.sort_values(["pair_id","tick"],kind="mergesort",inplace=True)
        d.reset_index(drop=True,inplace=True)
        totals["candidate_rows"]+=len(d)
        totals["duplicate_pair_ticks"]+=int(d.duplicated(["pair_id","tick"]).sum())
        for radius in (50,100,150):
            x=d.loc[d.horizontal_nm.le(radius)]
            for gap in (45,60,90):
                flags=segment_flags(x,gap)
                sensitivity.append(dict(week=week,bucket=b,radius_nm=radius,gap_s=gap,
                    observations=len(x),episodes=int(flags.sum()),unique_pairs=int(x.pair_id.nunique())))
        p=d.loc[d.horizontal_nm.le(100)].copy().reset_index(drop=True)
        if p.empty:continue
        flags=segment_flags(p,60)
        gap=p.tick.diff().fillna(np.inf).to_numpy(float)
        jump=p.horizontal_nm.diff().abs().fillna(0).to_numpy(float)
        same=p.pair_id.eq(p.pair_id.shift()).to_numpy()
        totals["physical_jump_splits"]+=int((same&(gap<=60)&(jump>30*np.maximum(gap,30)/30)).sum())
        seg=np.cumsum(flags)
        p["episode_id"]=[f"{week}:B{b:02d}E{x:07d}" for x in seg]
        episodes=_episode_table(p)
        p.to_parquet(trace_root/f"bucket_{b:02d}.parquet",index=False,compression="zstd")
        episodes.to_parquet(episode_root/f"bucket_{b:02d}.parquet",index=False,compression="zstd")
        totals["primary_rows"]+=len(p);totals["episodes"]+=len(episodes)
        print(week,"bucket",b,"primary rows",len(p),"episodes",len(episodes),flush=True)
    if totals["duplicate_pair_ticks"]:raise ValueError("A/B duplicate at same tick")
    sens=pd.DataFrame(sensitivity).groupby(["week","radius_nm","gap_s"],as_index=False).agg(
        observations=("observations","sum"),episodes=("episodes","sum"),unique_pairs=("unique_pairs","sum"))
    sens.to_csv(OUT/"tables"/f"{week}_radius_gap_sensitivity.csv",index=False)
    totals["week"]=week;totals["parameters"]=CONFIG
    done.write_text(json.dumps(totals,indent=2)+"\n")
    print(json.dumps(totals,indent=2),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("action",choices=["partition","segment"])
    q.add_argument("week",choices=WEEKS);q.add_argument("--force",action="store_true")
    a=q.parse_args()
    if a.action=="partition":partition(a.week,a.force)
    else:segment_week(a.week,a.force)
