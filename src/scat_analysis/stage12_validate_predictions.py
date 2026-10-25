"""As-of constant-velocity predictions and retrospective observed-path checks."""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from stage12_audit_inputs import OUT,WEEKS
from stage12_compute_features import BUCKETS

HORIZONS_MIN=(2,5,10,15,30)


def forecast(dx,dy,rvx,rvy,signed_v,dvr,tcpa,horizon_min):
    """Closest *future* approach within horizon; negative TCPA is invalid."""
    arrays=np.broadcast_arrays(dx,dy,rvx,rvy,signed_v,dvr,tcpa)
    dx,dy,rvx,rvy,signed_v,dvr,tcpa=[np.asarray(x,dtype=float) for x in arrays]
    valid=np.isfinite(dx)&np.isfinite(dy)&np.isfinite(rvx)&np.isfinite(rvy)&np.isfinite(tcpa)&(tcpa>=1/60)
    t=np.minimum(tcpa,horizon_min)
    cpa=np.where(valid,np.hypot(dx+rvx*t,dy+rvy*t),np.nan)
    v=np.where(valid,np.abs(signed_v+dvr*t),np.nan)
    return valid,t,cpa,v


def validate_week(week:str):
    root=OUT/"episodes"/f"{week}_episode_traces"
    features=OUT/"episodes"/f"{week}_episode_features"
    output=OUT/"tables"/f"{week}_prediction_validation"
    output.mkdir(parents=True,exist_ok=True)
    for bucket in range(BUCKETS):
        ep_path=features/f"bucket_{bucket:02d}.parquet"
        trace_path=root/f"bucket_{bucket:02d}.parquet"
        if not ep_path.exists():continue
        ep=pd.read_parquet(ep_path)
        trace=pd.read_parquet(trace_path,columns=["episode_id","tick","horizontal_nm","vertical_ft",
            "relative_east_nm","relative_north_nm","a_lat","a_lon","b_lat","b_lon","a_alt_ft","b_alt_ft"])
        trace.sort_values(["episode_id","tick"],inplace=True)
        trace["observed_min_h_to_tick"]=trace.groupby("episode_id").horizontal_nm.cummin()
        previous=trace.groupby("episode_id").observed_min_h_to_tick.shift()
        new_min=previous.isna()|trace.horizontal_nm.lt(previous)
        trace["observed_tmin_tick_to_tick"]=trace.tick.where(new_min).groupby(trace.episode_id).ffill()
        target=trace.rename(columns={c:"observed_"+c for c in trace.columns if c not in ("episode_id","tick")})
        parts=[]
        for horizon in HORIZONS_MIN:
            p=ep[["episode_id","pair_id","start_tick","start_relative_east_nm","start_relative_north_nm",
                  "start_relative_vx_nm_min","start_relative_vy_nm_min","start_signed_vertical_ft",
                  "start_a_vrate_fpm","start_b_vrate_fpm","start_tcpa_raw_min",
                  "start_a_lat","start_a_lon","start_b_lat","start_b_lon",
                  "start_a_alt_ft","start_b_alt_ft","start_a_speed_kt","start_b_speed_kt",
                  "start_a_track_deg","start_b_track_deg"]].copy()
            p["horizon_min"]=horizon;p["target_tick"]=p.start_tick+60*horizon
            dvr=p.start_b_vrate_fpm-p.start_a_vrate_fpm
            valid,t,cpa,v=forecast(p.start_relative_east_nm,p.start_relative_north_nm,
                p.start_relative_vx_nm_min,p.start_relative_vy_nm_min,p.start_signed_vertical_ft,
                dvr,p.start_tcpa_raw_min,horizon)
            p["future_cpa_valid"]=valid
            p["predicted_tcpa_within_horizon_min"]=np.where(valid,t,np.nan)
            p["predicted_cpa_nm"]=cpa
            p["predicted_vertical_at_cpa_ft"]=v
            p["predicted_end_dx_nm"]=p.start_relative_east_nm+p.start_relative_vx_nm_min*horizon
            p["predicted_end_dy_nm"]=p.start_relative_north_nm+p.start_relative_vy_nm_min*horizon
            p["predicted_end_h_nm"]=np.hypot(p.predicted_end_dx_nm,p.predicted_end_dy_nm)
            p["predicted_end_signed_v_ft"]=p.start_signed_vertical_ft+dvr*horizon
            p=p.merge(target,left_on=["episode_id","target_tick"],right_on=["episode_id","tick"],
                      how="left",validate="one_to_one").drop(columns="tick")
            p["future_observed"]=p.observed_horizontal_nm.notna()
            p["validation_status"]=np.where(~p.future_observed,"INSUFFICIENT_FUTURE_TRACE",
                                          np.where(~p.future_cpa_valid,"INVALID_CPA_KINEMATICS","VALIDATED"))
            p["relative_position_error_nm"]=np.hypot(p.predicted_end_dx_nm-p.observed_relative_east_nm,
                                                       p.predicted_end_dy_nm-p.observed_relative_north_nm)
            p["horizontal_end_error_nm"]=p.predicted_end_h_nm-p.observed_horizontal_nm
            p["vertical_end_error_ft"]=np.abs(p.predicted_end_signed_v_ft)-p.observed_vertical_ft
            p["cpa_error_nm"]=p.predicted_cpa_nm-p.observed_observed_min_h_to_tick
            p["tcpa_error_min"]=p.predicted_tcpa_within_horizon_min-(p.observed_observed_tmin_tick_to_tick-p.start_tick)/60
            for side in ("a","b"):
                track=np.deg2rad(p[f"start_{side}_track_deg"])
                speed=p[f"start_{side}_speed_kt"]*horizon/60
                px=speed*np.sin(track);py=speed*np.cos(track)
                meanlat=np.deg2rad((p[f"start_{side}_lat"]+p[f"observed_{side}_lat"])/2)
                ox=(p[f"observed_{side}_lon"]-p[f"start_{side}_lon"])*60*np.cos(meanlat)
                oy=(p[f"observed_{side}_lat"]-p[f"start_{side}_lat"])*60
                p[f"{side}_position_error_nm"]=np.hypot(px-ox,py-oy)
            keep=["episode_id","pair_id","start_tick","horizon_min","target_tick",
                  "future_cpa_valid","future_observed","validation_status",
                  "predicted_tcpa_within_horizon_min","predicted_cpa_nm","predicted_vertical_at_cpa_ft",
                  "predicted_end_h_nm","observed_horizontal_nm","predicted_end_signed_v_ft",
                  "observed_vertical_ft","observed_observed_min_h_to_tick",
                  "observed_observed_tmin_tick_to_tick","a_position_error_nm","b_position_error_nm",
                  "relative_position_error_nm","horizontal_end_error_nm","vertical_end_error_ft",
                  "cpa_error_nm","tcpa_error_min"]
            parts.append(p[keep])
        pd.concat(parts,ignore_index=True).to_parquet(output/f"bucket_{bucket:02d}.parquet",index=False,compression="zstd")
        print(week,"prediction bucket",bucket,"episodes",len(ep),flush=True)
    result=pd.read_parquet(output,columns=["episode_id","horizon_min","validation_status",
        "relative_position_error_nm","horizontal_end_error_nm","vertical_end_error_ft",
        "cpa_error_nm","tcpa_error_min"])
    summary=result.groupby(["horizon_min","validation_status"]).agg(n=("episode_id","size"),
        median_relative_position_error_nm=("relative_position_error_nm","median"),
        median_abs_horizontal_end_error_nm=("horizontal_end_error_nm",lambda x:x.abs().median()),
        median_abs_vertical_end_error_ft=("vertical_end_error_ft",lambda x:x.abs().median()),
        median_abs_cpa_error_nm=("cpa_error_nm",lambda x:x.abs().median()),
        median_abs_tcpa_error_min=("tcpa_error_min",lambda x:x.abs().median())).reset_index()
    summary.to_csv(OUT/"tables"/f"{week}_prediction_error_summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("week",choices=WEEKS);a=q.parse_args();validate_week(a.week)
