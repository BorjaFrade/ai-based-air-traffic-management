"""Classify observed pair entries/exits versus lost coverage and radius censoring."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from stage12_audit_inputs import OUT,WEEKS
from stage12_compute_features import BUCKETS


def haversine_nm(lat_a,lon_a,lat_b,lon_b):
    x=np.deg2rad(np.asarray(lat_a,float));y=np.deg2rad(np.asarray(lat_b,float))
    dl=y-x;do=np.deg2rad(np.asarray(lon_b,float)-np.asarray(lon_a,float))
    a=np.sin(dl/2)**2+np.cos(x)*np.cos(y)*np.sin(do/2)**2
    return 2*3440.065*np.arcsin(np.sqrt(np.clip(a,0,1)))


def classify_boundary(coobserved:np.ndarray,distance:np.ndarray,at_edge:np.ndarray,side:str)->np.ndarray:
    out=np.full(len(coobserved),"OBSERVATION_CENSORED",dtype="U26")
    out[at_edge]="DATA_EDGE_CENSORED"
    out[coobserved & (distance>100)]="OBSERVED_RADIUS_CROSSING"
    out[coobserved & (distance<=100)]="PHYSICAL_OR_GRID_BREAK"
    return out


def audit_week(week:str):
    states=pd.read_parquet(OUT/"tables"/f"{week}_snapshots_30s.parquet",
             columns=["tick","entity_key","latitude","longitude"])
    assert not states.duplicated(["tick","entity_key"]).any()
    idx=states.set_index(["tick","entity_key"])[["latitude","longitude"]]
    first_tick=int(states.tick.min());last_tick=int(states.tick.max())
    out=[]
    for bucket in range(BUCKETS):
        path=OUT/"episodes"/f"{week}_episode_features"/f"bucket_{bucket:02d}.parquet"
        if not path.exists():continue
        e=pd.read_parquet(path)
        for label,direction in (("before",-1),("after",1)):
            anchor=(e.start_tick if label=="before" else e.end_tick).to_numpy(int)
            e[f"{label}_coobserved"]=False
            e[f"{label}_horizontal_nm"]=np.nan
            e[f"{label}_offset_s"]=np.nan
            for offset in (30,60):
                base=anchor+direction*offset
                sampled={}
                for side in ("a","b"):
                    keys=pd.MultiIndex.from_arrays([base,e[f"{side}_key"].astype(str)])
                    sample=idx.reindex(keys)
                    sampled[f"{side}_lat"]=sample.latitude.to_numpy(float)
                    sampled[f"{side}_lon"]=sample.longitude.to_numpy(float)
                distance=haversine_nm(sampled["a_lat"],sampled["a_lon"],sampled["b_lat"],sampled["b_lon"])
                fresh=(~e[f"{label}_coobserved"].to_numpy())&np.isfinite(distance)
                e.loc[fresh,f"{label}_coobserved"]=True
                e.loc[fresh,f"{label}_horizontal_nm"]=distance[fresh]
                e.loc[fresh,f"{label}_offset_s"]=offset
        e["start_boundary"]=classify_boundary(e.before_coobserved.to_numpy(),e.before_horizontal_nm.to_numpy(),
                                               e.start_tick.le(first_tick+30).to_numpy(),"start")
        e["end_boundary"]=classify_boundary(e.after_coobserved.to_numpy(),e.after_horizontal_nm.to_numpy(),
                                             e.end_tick.ge(last_tick-30).to_numpy(),"end")
        e["fully_bracketed_at_100nm"]=(e.start_boundary.eq("OBSERVED_RADIUS_CROSSING")&
                                         e.end_boundary.eq("OBSERVED_RADIUS_CROSSING"))
        e["temporally_censored"]=(e.start_boundary.isin(["OBSERVATION_CENSORED","DATA_EDGE_CENSORED"])|
                                    e.end_boundary.isin(["OBSERVATION_CENSORED","DATA_EDGE_CENSORED"]))
        e["radius_truncated"]=(e.start_boundary.eq("OBSERVED_RADIUS_CROSSING")|
                                e.end_boundary.eq("OBSERVED_RADIUS_CROSSING"))
        out.append(e)
        print(week,"boundary bucket",bucket,"episodes",len(e),flush=True)
    episodes=pd.concat(out,ignore_index=True)
    episodes.to_parquet(OUT/"episodes"/f"{week}_episode_summary.parquet",index=False,compression="zstd")
    rows=dict(week=week,episodes=len(episodes),fully_bracketed_at_100nm=int(episodes.fully_bracketed_at_100nm.sum()),
        temporally_censored=int(episodes.temporally_censored.sum()),radius_truncated=int(episodes.radius_truncated.sum()),
        start_boundary=episodes.start_boundary.value_counts().to_dict(),
        end_boundary=episodes.end_boundary.value_counts().to_dict(),
        quality_for_structure=int(episodes.quality_for_structure.sum()),
        caveat="Radius-bracketed is complete only within 100 NM scope, not a complete physical interaction at unlimited distance.")
    (OUT/"diagnostics"/f"{week}_boundary_audit.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("week",choices=WEEKS);a=q.parse_args();audit_week(a.week)
