"""Discover all co-observed unordered aircraft pairs, independent of ATC labels.

The spatial index searches to 150 NM so 50/100/150 NM can be compared from
one immutable native-state snapshot pass. The primary analysis uses 100 NM.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from stage11_core import BASE, _week_groups, snapshot_tick
from stage12_audit_inputs import OUT, WEEKS

CONFIG={"grid_seconds":30,"max_state_age_s":15,"primary_radius_nm":100,
        "sensitivity_radii_nm":[50,150],"episode_gap_s":60,"gap_sensitivity_s":[45,60,90],
        "airborne_only":True,"vertical_filter":None,"earth_radius_nm":3440.065,
        "no_interpolation":True,"constant_velocity_model":"Stage05 relative east/north velocity and CPA equations"}
STATE_COLUMNS=["flight_key","flight_id","archive_week","source_archive","source_member",
   "plot_index","time_epoch","latitude","longitude","altitude_ft","ground_speed_kt",
   "track_deg","vertical_rate_fpm","on_ground","flight_phase","active_centre_id","callsign"]
PAIR_FIELDS=["week","tick","pair_id","a_key","b_key","a_flight_key","b_flight_key",
   "a_source_archive","b_source_archive","a_source_member","b_source_member",
   "a_source_epoch","b_source_epoch","a_age_s","b_age_s","a_plot_index","b_plot_index",
   "a_lat","a_lon","b_lat","b_lon","a_alt_ft","b_alt_ft","a_speed_kt","b_speed_kt",
   "a_track_deg","b_track_deg","a_vrate_fpm","b_vrate_fpm","a_phase","b_phase",
   "a_sector","b_sector","horizontal_nm","signed_vertical_ft","vertical_ft",
   "relative_east_nm","relative_north_nm","relative_vx_nm_min","relative_vy_nm_min",
   "relative_speed_kt","closing_nm_min","crossing_angle_deg","tcpa_raw_min",
   "cpa_unbounded_nm","concurrent_neighbors_a","concurrent_neighbors_b"]
STRING_FIELDS={"week","pair_id","a_key","b_key","a_flight_key","b_flight_key",
 "a_source_archive","b_source_archive","a_source_member","b_source_member",
 "a_phase","b_phase","a_sector","b_sector"}
INT_FIELDS={"tick","a_plot_index","b_plot_index","concurrent_neighbors_a","concurrent_neighbors_b"}
SCHEMA=pa.schema([(k,pa.string() if k in STRING_FIELDS else pa.int64() if k in INT_FIELDS else pa.float64()) for k in PAIR_FIELDS])


def canonical_pair(week:str,a:str,b:str)->tuple[str,str,str]:
    if not a or not b or a==b:raise ValueError("A pair needs two distinct stable identities")
    x,y=sorted((a,b))
    return x,y,f"{week}::{len(x)}:{x}|{len(y)}:{y}"


def load_snapshots(week:str,step:int=30,max_age:float=15)->tuple[pd.DataFrame,dict]:
    source=pq.ParquetFile(BASE/"02_traffic/scat_states.parquet")
    parts=[];native=0;invalid=0;ground=0
    for i in _week_groups(source,week):
        f=source.read_row_group(i,columns=STATE_COLUMNS).to_pandas()
        f=f.loc[f.archive_week.eq(week)]
        native+=len(f)
        valid=(f.flight_key.notna()&f.source_member.notna()&f.time_epoch.notna()&
               f.latitude.between(-90,90)&f.longitude.between(-180,180)&
               f.altitude_ft.between(-1000,60000))
        invalid+=int((~valid).sum());f=f.loc[valid].copy()
        ground_mask=f.on_ground.fillna(False).astype(bool)
        ground+=int(ground_mask.sum())
        if CONFIG["airborne_only"]:f=f.loc[~ground_mask].copy()
        if f.empty:continue
        f["tick"]=snapshot_tick(f.time_epoch.to_numpy(float),step)
        f=f.loc[(f.tick-f.time_epoch).between(0,max_age)]
        if len(f):parts.append(f)
    states=pd.concat(parts,ignore_index=True)
    identity=states.groupby("flight_key",sort=False).agg(members=("source_member","nunique"),
             archives=("source_archive","nunique"),flight_ids=("flight_id","nunique")).reset_index()
    colliding=identity.loc[(identity.members>1)|(identity.archives>1)|(identity.flight_ids>1),"flight_key"]
    if len(colliding):
        # Same archive flight ID appears in multiple source records. Keep them distinct.
        states["entity_key"]=np.where(states.flight_key.isin(colliding),
          states.flight_key.astype(str)+"::"+states.source_member.astype(str),states.flight_key.astype(str))
    else:states["entity_key"]=states.flight_key.astype(str)
    states.sort_values(["tick","entity_key","time_epoch","plot_index"],kind="mergesort",inplace=True)
    duplicate=int(states.duplicated(["tick","entity_key"]).sum())
    states.drop_duplicates(["tick","entity_key"],keep="last",inplace=True)
    states.reset_index(drop=True,inplace=True)
    if states.duplicated(["tick","entity_key"]).any():raise ValueError("Snapshot identity collision")
    if (states.time_epoch>states.tick+1e-6).any():raise ValueError("Future state")
    audit=dict(week=week,grid_seconds=step,native_rows=native,invalid_native_rows=invalid,
        ground_rows_excluded=ground,snapshots=len(states),identity_collisions=len(colliding),
        duplicate_native_rows_at_same_tick=duplicate,unique_entities=int(states.entity_key.nunique()),
        ticks=int(states.tick.nunique()),max_source_age_s=float((states.tick-states.time_epoch).max()))
    return states,audit


def pair_geometry(g:pd.DataFrame,tick:int,week:str,radius_nm:float=150)->pd.DataFrame:
    """KD-tree candidate search on unit sphere; exact great-circle filtering."""
    if len(g)<2:return pd.DataFrame(columns=PAIR_FIELDS)
    g=g.sort_values("entity_key",kind="mergesort").reset_index(drop=True)
    lat=np.deg2rad(g.latitude.to_numpy(float));lon=np.deg2rad(g.longitude.to_numpy(float))
    xyz=np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    chord=2*np.sin(radius_nm/(2*CONFIG["earth_radius_nm"]))
    ij=cKDTree(xyz).query_pairs(chord,output_type="ndarray")
    if not len(ij):return pd.DataFrame(columns=PAIR_FIELDS)
    i,j=ij[:,0],ij[:,1]
    arg=np.sin((lat[j]-lat[i])/2)**2+np.cos(lat[i])*np.cos(lat[j])*np.sin((lon[j]-lon[i])/2)**2
    h=2*CONFIG["earth_radius_nm"]*np.arcsin(np.sqrt(np.clip(arg,0,1)))
    keep=h<=radius_nm+1e-7;i=i[keep];j=j[keep];h=h[keep]
    if not len(i):return pd.DataFrame(columns=PAIR_FIELDS)
    a=g.iloc[i].reset_index(drop=True);b=g.iloc[j].reset_index(drop=True)
    dx=(b.longitude.to_numpy(float)-a.longitude.to_numpy(float))*60*np.cos((lat[i]+lat[j])/2)
    dy=(b.latitude.to_numpy(float)-a.latitude.to_numpy(float))*60
    ta=np.deg2rad(a.track_deg.to_numpy(float));tb=np.deg2rad(b.track_deg.to_numpy(float))
    sa=a.ground_speed_kt.to_numpy(float);sb=b.ground_speed_kt.to_numpy(float)
    rvx=(sb*np.sin(tb)-sa*np.sin(ta))/60;rvy=(sb*np.cos(tb)-sa*np.cos(ta))/60
    rv2=rvx**2+rvy**2;dot=dx*rvx+dy*rvy
    tcpa=np.divide(-dot,rv2,out=np.full(len(i),np.nan),where=rv2>1e-12)
    cpa=np.hypot(dx+rvx*tcpa,dy+rvy*tcpa)
    closing=np.divide(-dot,np.hypot(dx,dy),out=np.full(len(i),np.nan),where=np.hypot(dx,dy)>1e-9)
    signed=b.altitude_ft.to_numpy(float)-a.altitude_ft.to_numpy(float)
    keys_a=a.entity_key.astype(str).to_numpy();keys_b=b.entity_key.astype(str).to_numpy()
    pairid=np.array([canonical_pair(week,x,y)[2] for x,y in zip(keys_a,keys_b)],dtype=object)
    counts=np.bincount(np.r_[i,j],minlength=len(g))
    out=pd.DataFrame({"week":week,"tick":int(tick),"pair_id":pairid,"a_key":keys_a,"b_key":keys_b,
      "a_flight_key":a.flight_key.to_numpy(),"b_flight_key":b.flight_key.to_numpy(),
      "a_source_archive":a.source_archive.to_numpy(),"b_source_archive":b.source_archive.to_numpy(),
      "a_source_member":a.source_member.to_numpy(),"b_source_member":b.source_member.to_numpy(),
      "a_source_epoch":a.time_epoch.to_numpy(float),"b_source_epoch":b.time_epoch.to_numpy(float),
      "a_age_s":tick-a.time_epoch.to_numpy(float),"b_age_s":tick-b.time_epoch.to_numpy(float),
      "a_plot_index":a.plot_index.to_numpy(int),"b_plot_index":b.plot_index.to_numpy(int),
      "a_lat":a.latitude.to_numpy(float),"a_lon":a.longitude.to_numpy(float),
      "b_lat":b.latitude.to_numpy(float),"b_lon":b.longitude.to_numpy(float),
      "a_alt_ft":a.altitude_ft.to_numpy(float),"b_alt_ft":b.altitude_ft.to_numpy(float),
      "a_speed_kt":sa,"b_speed_kt":sb,"a_track_deg":a.track_deg.to_numpy(float),
      "b_track_deg":b.track_deg.to_numpy(float),"a_vrate_fpm":a.vertical_rate_fpm.to_numpy(float),
      "b_vrate_fpm":b.vertical_rate_fpm.to_numpy(float),"a_phase":a.flight_phase.to_numpy(),
      "b_phase":b.flight_phase.to_numpy(),"a_sector":a.active_centre_id.to_numpy(),
      "b_sector":b.active_centre_id.to_numpy(),"horizontal_nm":h,"signed_vertical_ft":signed,
      "vertical_ft":np.abs(signed),"relative_east_nm":dx,"relative_north_nm":dy,
      "relative_vx_nm_min":rvx,"relative_vy_nm_min":rvy,"relative_speed_kt":60*np.sqrt(rv2),
      "closing_nm_min":closing,"crossing_angle_deg":np.abs(((b.track_deg.to_numpy(float)-a.track_deg.to_numpy(float)+180)%360)-180),
      "tcpa_raw_min":tcpa,"cpa_unbounded_nm":cpa,
      "concurrent_neighbors_a":counts[i].astype(int),"concurrent_neighbors_b":counts[j].astype(int)})
    return out[list(PAIR_FIELDS)]


def build_week(week:str,force:bool=False):
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"tables"/f"{week}_candidate_pairs_150nm.parquet"
    snapshots_path=OUT/"tables"/f"{week}_snapshots_30s.parquet"
    if path.exists() and snapshots_path.exists() and not force:
        print("CACHED",path,flush=True);return
    if path.exists():path.unlink()
    states,audit=load_snapshots(week)
    states.to_parquet(snapshots_path,index=False,compression="zstd")
    writer=None;parts=[];ticks=0;pair_rows=0;within={50:0,100:0,150:0};max_aircraft=0
    try:
        for tick,g in states.groupby("tick",sort=False):
            max_aircraft=max(max_aircraft,len(g))
            p=pair_geometry(g,int(tick),week,150)
            if len(p):
                pair_rows+=len(p)
                for radius in within:within[radius]+=int(p.horizontal_nm.le(radius).sum())
                parts.append(p)
            ticks+=1
            if ticks%250==0 and parts:
                batch=pa.Table.from_pandas(pd.concat(parts,ignore_index=True),preserve_index=False).cast(SCHEMA)
                if writer is None:writer=pq.ParquetWriter(path,SCHEMA,compression="zstd")
                writer.write_table(batch);parts=[]
            if ticks%2500==0:print(week,"ticks",ticks,"pairs<=150",pair_rows,flush=True)
        if parts:
            batch=pa.Table.from_pandas(pd.concat(parts,ignore_index=True),preserve_index=False).cast(SCHEMA)
            if writer is None:writer=pq.ParquetWriter(path,SCHEMA,compression="zstd")
            writer.write_table(batch)
    finally:
        if writer is not None:writer.close()
    audit.update(candidate_rows_within_nm=within,max_aircraft_per_tick=max_aircraft,
                 elapsed_seconds=round(time.monotonic()-start,2),
                 peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,2),
                 parameters=CONFIG)
    (OUT/"diagnostics"/f"{week}_pair_build_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("week",choices=WEEKS);q.add_argument("--force",action="store_true")
    a=q.parse_args();build_week(a.week,a.force)
