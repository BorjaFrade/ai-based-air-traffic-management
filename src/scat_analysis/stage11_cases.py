"""Traceable, observed Stage 11 case gallery for the pilot matched cohort."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage11_core import BASE, OUT, WEEK, _week_groups, save


def choose_cases(a: pd.DataFrame) -> pd.DataFrame:
    choices=[
        ("converging_intervention",a.ATC_INTERVENTION.eq(1)&a.horizontal_closing_rate_nm_min.gt(0)),
        ("diverging_intervention",a.ATC_INTERVENTION.eq(1)&a.horizontal_closing_rate_nm_min.lt(0)),
        ("large_initial_separation",a.ATC_INTERVENTION.eq(1)&a.horizontal_separation_nm.ge(35)),
        ("predicted_approach",a.ATC_INTERVENTION.eq(1)&a.tcpa_min.between(1/60,10)&a.cpa_nm.lt(5)),
        ("level_authorization",a.ATC_INTERVENTION.eq(1)&a.clearance_type.eq("LEVEL")),
        ("heading_authorization",a.ATC_INTERVENTION.eq(1)&a.clearance_type.eq("HEADING")),
        ("speed_authorization",a.ATC_INTERVENTION.eq(1)&a.clearance_type.eq("SPEED")),
        ("green_intervention",a.ATC_INTERVENTION.eq(1)&a.focal_dynamic_state.eq("GREEN")),
        ("red_control_anchor",a.ATC_INTERVENTION.eq(0)&a.focal_dynamic_state.eq("RED"))]
    selected=[];used=set()
    for label,mask in choices:
        part=a.loc[mask&~a.window_id.isin(used)].copy()
        if part.empty:
            continue
        part["quality_rank"]=(part.relative_seconds.abs()+np.where(part.tcpa_min.between(1/60,30),0,1000))
        row=part.sort_values(["quality_rank","window_id"]).iloc[0].drop(labels="quality_rank").to_dict()
        row["selection_category"]=label
        selected.append(row);used.add(row["window_id"])
    return pd.DataFrame(selected)


def load_tracks(cases:pd.DataFrame)->pd.DataFrame:
    ids=set(cases.focal_flight_key.astype(str))|set(cases.other_flight_key.astype(str))
    bounds=(cases.event_epoch.min()-300,cases.event_epoch.max()+300)
    p=pq.ParquetFile(BASE/"02_traffic/scat_states.parquet")
    cols=["flight_key","archive_week","time_epoch","latitude","longitude","altitude_ft","track_deg","source_archive","plot_index"]
    parts=[]
    for i in _week_groups(p,WEEK):
        f=p.read_row_group(i,columns=cols).to_pandas()
        f=f.loc[f.archive_week.eq(WEEK)&f.flight_key.isin(ids)&f.time_epoch.between(*bounds)]
        if len(f):parts.append(f)
    source=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)
    tracks=[]
    for row in cases.itertuples():
        q=source.loc[source.flight_key.isin([row.focal_flight_key,row.other_flight_key]) &
                     source.time_epoch.between(row.event_epoch-300,row.event_epoch+300)].copy()
        q["case_id"]=row.selection_category
        q["relative_seconds"]=q.time_epoch-row.event_epoch
        q["aircraft_role"]=np.where(q.flight_key.eq(row.focal_flight_key),"focal","fixed_neighbour")
        tracks.append(q)
    return pd.concat(tracks,ignore_index=True)


def draw_case(row,t:pd.DataFrame,tr:pd.DataFrame):
    fig,ax=plt.subplots(2,2,figsize=(12.5,8.3),layout="constrained")
    lat0=float(tr.latitude.median());lon0=float(tr.longitude.median())
    for role,color in [("focal","#c35025"),("fixed_neighbour","#1e6c98")]:
        s=tr.loc[tr.aircraft_role.eq(role)].sort_values("relative_seconds")
        if s.empty:continue
        x=(s.longitude.to_numpy(float)-lon0)*60*np.cos(np.deg2rad(lat0));y=(s.latitude.to_numpy(float)-lat0)*60
        ax[0,0].plot(x,y,color=color,label=f"{role}: {s.flight_key.iloc[0]}")
        pre=s.loc[s.relative_seconds.le(0)]
        if len(pre):
            xx=(pre.longitude.iloc[-1]-lon0)*60*np.cos(np.deg2rad(lat0));yy=(pre.latitude.iloc[-1]-lat0)*60
            ax[0,0].scatter([xx],[yy],color=color,marker="*",s=100,zorder=3)
        if len(x)>1:
            k=max(0,len(x)//2-1)
            ax[0,0].annotate("",xy=(x[k+1],y[k+1]),xytext=(x[k],y[k]),arrowprops={"arrowstyle":"->","color":color,"lw":2})
    ax[0,0].set(xlabel="East–west local offset (NM)",ylabel="North–south local offset (NM)",title="Observed aircraft tracks; star = last pre-anchor state")
    ax[0,0].legend(fontsize=7);ax[0,0].axis("equal");ax[0,0].grid(alpha=.2)
    for field,color,label in [("horizontal_separation_nm","#c35025","Horizontal NM"),("vertical_separation_ft","#1e6c98","Vertical ft")]:
        axis=ax[0,1] if field.startswith("horizontal") else ax[0,1].twinx()
        axis.plot(t.relative_seconds,t[field],color=color,label=label)
        axis.set_ylabel(label,color=color)
    ax[0,1].axvline(0,color="black",ls="--",lw=.8)
    ax[0,1].set(xlabel="Relative time (s)",title="Same-pair observed separation")
    ax[0,1].grid(alpha=.2)
    valid=t.tcpa_min.between(1/60,30)
    ax[1,0].plot(t.loc[valid,"relative_seconds"],t.loc[valid,"cpa_nm"],color="#854a98",marker=".",ms=3)
    ax[1,0].axvline(0,color="black",ls="--",lw=.8)
    ax[1,0].set(xlabel="Relative time (s)",ylabel="Future horizontal CPA (NM)",title="Same-pair kinematic prediction; TCPA ≥1 s")
    ax[1,0].grid(alpha=.2)
    ax[1,1].axis("off")
    role="ATC clearance" if row.ATC_INTERVENTION==1 else "Control pseudoanchor: no clearance at t=0"
    details=(f"{row.selection_category}\n{role}\nType: {row.clearance_type} | Detector: {row.focal_dynamic_state}\n"
             f"Focal: {row.focal_flight_key}\nNeighbour: {row.other_flight_key}\n"
             f"Anchor H: {row.horizontal_separation_nm:.2f} NM; V: {row.vertical_separation_ft:.0f} ft\n"
             f"CPA: {row.cpa_nm:.2f} NM; TCPA: {row.tcpa_min:.2f} min\n"
             f"Stage05 pair bins: {len(t)} / 21; native state plots: {len(tr)}\n"
             f"Pair selected at t={row.relative_seconds:.1f} s; postevent association only.\n"
             "Not a confirmed conflict.")
    ax[1,1].text(0,.98,details,va="top",fontsize=10,linespacing=1.5)
    fig.suptitle(f"SCAT Stage 11 case — {row.selection_category.replace('_',' ').title()}",fontsize=15)
    return fig


def main():
    a=pd.read_parquet(OUT/"tables/11a_anchor_pairs.parquet")
    trace=pd.read_parquet(OUT/"tables/11a_fixed_pair_timeline.parquet")
    cases=choose_cases(a)
    save(cases,"examples/selected_cases.csv")
    tracks=load_tracks(cases)
    save(tracks,"examples/selected_native_tracks.parquet")
    directory=OUT/"figures/09_case_studies";directory.mkdir(parents=True,exist_ok=True)
    for row in cases.itertuples(index=False):
        t=trace.loc[trace.window_id.eq(row.window_id)].sort_values("relative_seconds")
        tr=tracks.loc[tracks.case_id.eq(row.selection_category)]
        if t.empty or tr.empty:continue
        fig=draw_case(row,t,tr)
        name=row.selection_category
        fig.savefig(directory/f"{name}.png",dpi=300,bbox_inches="tight")
        fig.savefig(directory/f"{name}.svg",bbox_inches="tight")
        plt.close(fig)
        t.to_csv(directory/f"{name}_pair_data.csv",index=False)
        tr.to_csv(directory/f"{name}_native_tracks.csv",index=False)
        (directory/f"{name}.json").write_text(json.dumps({"selection":name,"week":WEEK,"window_id":row.window_id,
             "focal_flight_key":row.focal_flight_key,"neighbour_flight_key":row.other_flight_key,
             "event_epoch":row.event_epoch,"clearance_type":row.clearance_type,"detector_state":row.focal_dynamic_state,
             "pair_rows":len(t),"native_plots":len(tr),"source":"Stage02 native states; Stage05 fixed pair features"},indent=2)+"\n")
    print(f"Case gallery: {len(cases)} selected, {len(tracks)} native plots")


if __name__=="__main__":main()
