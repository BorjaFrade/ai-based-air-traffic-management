#!/usr/bin/env python3
"""Reproduce CEAS figures 1-7 from the frozen SCAT Stage 19-21 outputs.

Run from the repository root: python3 paper/figure_scripts/generate.py --figure all
No Stage 1-21 result is written or modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
CAL = ROOT / "data/processed/scat/19_calib_selection/tables"
S20 = ROOT / "results/stage20"
S21 = ROOT / "results/stage21"
COLOURS = {"verde": "#2f7d59", "amarillo": "#d2a527", "naranja": "#d16b27", "rojo": "#b53e3e"}
ORDER = ["verde", "amarillo", "naranja", "rojo"]
EN = {"verde": "Green", "amarillo": "Yellow", "naranja": "Orange", "rojo": "Red"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def check(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


def fig1():
    fig, ax = plt.subplots(figsize=(6.8, 3.75))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis("off")
    obs, calc = "#e4edf6", "#eee9f5"
    def box(x, y, w, h, text, fill, edge="#4c6176"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                           facecolor=fill, edgecolor=edge, linewidth=1)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>", mutation_scale=10,
                                     color="#536779", linewidth=1))
    box(.2,4.25,1.7,.65,"SCAT archives\n13 sampled weeks",obs)
    box(2.4,4.25,2.65,.65,"Surveillance + flight plans\nobserved records",obs)
    box(5.6,4.25,2.7,.65,"Extracted clearances\nobserved records",obs)
    arrow(1.98,4.58,2.3,4.58);arrow(5.13,4.58,5.5,4.58)
    box(.25,2.7,2.55,.85,"Route prediction\nand colour calibration",calc)
    box(3.72,2.7,2.55,.85,"Order attribution\nand encounter grouping",calc)
    box(7.15,2.7,2.55,.85,"Observed-track\nscreening",calc)
    arrow(1.2,4.15,1.2,3.68);arrow(6.9,4.15,5.0,3.68)
    arrow(2.9,3.12,3.61,3.12);arrow(6.37,3.12,7.04,3.12)
    box(.25,1.42,2.55,.48,"9 calibration / 4 test weeks",calc)
    box(3.72,1.42,2.55,.48,"Pair-instants → encounters",calc)
    box(7.15,1.42,2.55,.48,"Observed 20-min follow-up",calc)
    arrow(1.52,2.6,1.52,2.02);arrow(5,2.6,5,2.02);arrow(8.42,2.6,8.42,2.02)
    box(3.48,.35,3.15,.55,"Audited rates, multiplicity\nand coverage",calc)
    arrow(5,1.28,5,1.02);arrow(8.42,1.28,6.75,.68)
    ax.text(.2,5.16,"Blue: observed SCAT data     Lavender: calculated predictions, links and summaries",fontsize=7)
    ax.text(6.8,.4,"M0–M5 comparison",fontsize=7,color="#555")
    save(fig,"fig01_workflow")


def fig2():
    heat = pd.read_csv(CAL/"heat_vs_green_R.csv", index_col=0)
    check(heat.shape==(9,4),"Stage 19 heat-map shape")
    # Panel a: geometry with every class region shaded in its colour
    fig,a=plt.subplots(figsize=(3.5,3.4))
    L=2.55
    a.add_patch(Rectangle((0,0),L,L,facecolor="#bfe0cd",edgecolor="none"))
    a.add_patch(Rectangle((0,0),2,2,facecolor="#f4dd8c",edgecolor="none"))
    a.add_patch(Rectangle((0,0),1.2,1.2,facecolor="#bfe0cd",edgecolor="none"))
    a.add_patch(Rectangle((0,0),1,1,facecolor="#e08a45",edgecolor="none"))
    a.add_patch(Rectangle((0,0),1,1,facecolor="none",edgecolor=COLOURS["rojo"],hatch="////",linewidth=0))
    for level,style in [(2,"-"),(1.2,"--"),(1,":")]:
        a.plot([0,level,level],[level,level,0],style,color="#263746",lw=1.3)
    a.text(1.02,2.04,"m = 2",fontsize=7,ha="right",va="bottom")
    a.text(1.22,1.24,"m = 1.2",fontsize=7,va="bottom")
    a.text(.5,.5,"m < 1\nRed (τ < 5 min)\nOrange (τ ≥ 5 min)",fontsize=6.5,ha="center",va="center",
           bbox=dict(boxstyle="round,pad=.2",fc="white",ec="none",alpha=.85))
    a.text(1.55,1.55,"1.2 ≤ m < 2\nYellow",fontsize=7,ha="center",va="center")
    a.text(2.28,2.28,"m ≥ 2\nGreen",fontsize=7,ha="center",va="center")
    a.text(.6,1.1,"1 ≤ m < 1.2: green",fontsize=6.3,ha="center",va="center")
    a.set(xlim=(0,L),ylim=(0,L),xlabel="Horizontal separation H / 5 NM",
          ylabel="Vertical separation V / 1,000 ft")
    a.set_aspect("equal");a.grid(alpha=.15)
    fig.tight_layout();save(fig,"fig02a_geometry")
    # Panel b: calibration lift heat map
    counts = pd.read_csv(CAL/"heat_pairs_R.csv", index_col=0)
    check(counts.shape==(9,4),"Stage 19 count shape")
    fig,b=plt.subplots(figsize=(3.9,3.4))
    z=heat.to_numpy(float)
    im=b.pcolormesh(np.arange(5)-.5,np.arange(10)-.5,z,shading="flat",cmap="YlOrBr",vmin=.8,vmax=3.3)
    b.set_xlim(-.5,3.5);b.set_ylim(-.5,8.5)
    b.set_xticks(range(4),["0–2","2–5","5–10","10–20"])
    b.set_yticks(range(9),list(heat.index))
    b.set(xlabel="Time to predicted minimum τ (min)",ylabel="Predicted minimum mₘᵢₙ")
    for i in range(9):
        for j in range(4):
            b.text(j,i,f"{z[i,j]:.2f}",ha="center",va="center",fontsize=6.5,
                   color="white" if z[i,j]>2.5 else "#18232d")
    for x,y,w,h,c in [(-.5,-.5,2,1,COLOURS["rojo"]),(1.5,-.5,2,1,COLOURS["naranja"]),
                      (-.5,1.5,4,2,COLOURS["amarillo"]),(-.5,.5,4,1,COLOURS["verde"])]:
        b.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=c,linewidth=2.2))
    cb=fig.colorbar(im,ax=b,fraction=.045,pad=.03);cb.set_label("Lift (density adjusted)")
    fig.tight_layout();save(fig,"fig02b_lift_heatmap")



def read_week_flights(path, week, flights, cols, timecol, start, end):
    sys.path.insert(0,str(ROOT/"src/scat_analysis"))
    import stage13_level_change_geometry as s13
    pf=pq.ParquetFile(path); chunks=[]
    for rg in s13.week_row_groups(pf,week):
        q=pf.read_row_group(rg,columns=cols).to_pandas()
        q=q.loc[q.archive_week.eq(week)&q.flight_key.isin(flights)&q[timecol].between(start,end)]
        if len(q):chunks.append(q)
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def fig3():
    sys.path.insert(0,str(ROOT/"src/scat_analysis"))
    import stage16_pair_hazard as s16
    import stage21_maneuver_screening as s21
    e=pd.read_csv(S21/"stage21_encounters.csv").set_index("episode_id").loc[33]
    check(e.week=="20161015_20161021" and e.split=="calibration" and e.color=="rojo" and
          e.observed_evaluable and e.full_coverage,"Example encounter identity/coverage")
    T=pd.Timestamp(e.anchor_time_utc).timestamp(); flights=[e.flight1,e.flight2]
    scol=["archive_week","flight_key","source_member","time_epoch","latitude","longitude",
          "altitude_ft","ground_speed_kt","track_deg","vertical_rate_fpm"]
    states=read_week_flights(ROOT/"data/processed/scat/02_traffic/scat_states.parquet",e.week,flights,
                             scol,"time_epoch",T-35,T+1200+35)
    pcol=["archive_week","flight_key","source_member","prediction_epoch","route_json"]
    preds=read_week_flights(ROOT/"data/processed/scat/02_traffic/scat_predicted_trajectories.parquet",
                            e.week,flights,pcol,"prediction_epoch",T-900,T)
    start_states=states.loc[states.time_epoch.le(T)&states.time_epoch.ge(T-10)].sort_values("time_epoch").groupby("flight_key").tail(1)
    start_preds=preds.sort_values("prediction_epoch").groupby("flight_key").tail(1)
    start_states=start_states.set_index("flight_key").loc[flights].reset_index()
    start_preds=start_preds.set_index("flight_key").loc[flights].reset_index()
    check(len(start_states)==2 and len(start_preds)==2,"Example anchor inputs")
    x0,y0=s16.to_plane(start_states.latitude,start_states.longitude);z0=start_states.altitude_ft.to_numpy(float)
    ux,uy=s16.plane_direction(start_states.latitude,start_states.longitude,start_states.track_deg)
    speed=start_states.ground_speed_kt.to_numpy(float)/3600
    px=np.full((2,s16.MAX_PATH_FIXES),np.nan);py=px.copy();pz=px.copy();ok=np.zeros(2,bool)
    for i,rec in enumerate(start_preds.itertuples()):
        route=s16.parse_route(rec.route_json)
        if route is None:continue
        fx,fy,fz=s16.build_path(route,T,x0[i],y0[i],ux[i],uy[i])
        if len(fx):px[i,:len(fx)]=fx;py[i,:len(fy)]=fy;pz[i,:len(fz)]=fz;ok[i]=True
    grid=np.arange(0,1201,10,dtype=float)
    XR,YR,ZR=s16.route_positions(x0,y0,z0,ux,uy,speed,px,py,pz,grid)
    rules=s16.s15.load_rules()
    vr=s16.s15.deadbanded(start_states.vertical_rate_fpm.to_numpy(float),rules["deadband"])
    XC,YC,ZC=s16.constant_velocity(x0,y0,z0,ux*speed,uy*speed,vr,grid)
    X=np.where(ok[:,None],XR,XC);Y=np.where(ok[:,None],YR,YC);Z=np.where(ok[:,None],ZR,ZC)
    predicted=np.maximum(np.hypot(X[0]-X[1],Y[0]-Y[1])/5,np.abs(Z[0]-Z[1])/1000)
    check(abs(predicted.min()-e.m_min_predicted_stage16_start)<1e-6 and
          grid[predicted.argmin()]==e.tau_predicted_stage16_start_sec,"Replayed Stage 16 forecast")
    tr=[]
    for f in flights:
        a=states.loc[states.flight_key.eq(f),["time_epoch","latitude","longitude","altitude_ft"]].to_numpy(float)
        a=a[np.argsort(a[:,0])];tr.append(a[np.r_[True,np.diff(a[:,0])>0]])
    observed,(sec,mo)=s21.observed_pair(tr[0],tr[1],T,s21.RULES)
    check(observed["full_coverage"] and abs(observed["m_min_observed"]-e.m_min_observed)<1e-6,
          "Replayed Stage 21 observation")
    man=pd.read_csv(S21/"stage21_clearance_maneuvers.csv");man=man.loc[man.episode_id.eq(33)].copy()
    check(len(man)==2,"Example has two eligible orders")
    for panel,stem in (("a","fig03a_full_horizon"),("b","fig03b_closest_approach")):
        fig,plot=plt.subplots(figsize=(3.55,3.2))
        plot.plot(grid/60,predicted,color="#ab4141",lw=1.6,ls="--",label="Route-based forecast")
        plot.plot(sec/60,mo,color="#24537a",lw=1.6,label="Synchronized observed")
        plot.axhline(1,color="black",lw=.8,ls=":",label="m = 1")
        for k,row in enumerate(man.itertuples()):
            order=(row.clearance_epoch-T)/60
            plot.axvline(order,color="#6d4d88",lw=1,ls="-.",alpha=.8)
            if pd.notna(row.maneuver_onset_sec):
                plot.axvline(order+row.maneuver_onset_sec/60,color="#39836e",lw=1,ls=":")
        plot.plot([],[],color="#6d4d88",ls="-.",label="Order issue")
        plot.plot([],[],color="#39836e",ls=":",label="Estimated onset")
        plot.grid(alpha=.16);plot.set_xlabel("Minutes after 2016-10-16 12:58 UTC")
        plot.set_ylabel("Normalized separation m")
        if panel=="a":
            plot.set(xlim=(0,20));plot.set_yscale("log");plot.set_ylim(.2,100)
            plot.set_yticks([.3,1,3,10,30,100],["0.3","1","3","10","30","100"])
        else:
            plot.set(xlim=(2,8),ylim=(0,5.2))
            for k,row in enumerate(man.itertuples()):
                order=(row.clearance_epoch-T)/60
                plot.text(order-.03,5.05-k*.35,f"{k+1}",ha="right",va="top",fontsize=6.5,color="#6d4d88")
        fig.legend(*plot.get_legend_handles_labels(),loc="lower center",frameon=False,ncol=3,fontsize=6.3)
        fig.tight_layout(rect=(0,.12,1,1));save(fig,stem)


def daily_rates():
    audit=json.loads((CAL/"audit.json").read_text())
    chunks=[]
    for w in audit["weeks"]:
        d=pd.read_parquet(ROOT/f"data/processed/scat/16_pair_hazard/tables/cells_{w}.parquet",
                          columns=["day","predictor","m_bin","tau_bin","n_pairs","n_lab_300"])
        first=int(pd.Timestamp(w[:8],tz="UTC").timestamp()//86400)
        last=int(pd.Timestamp(w[-8:],tz="UTC").timestamp()//86400)
        d=d.loc[d.predictor.eq("R")&d.day.between(first,last)].copy()
        m,t=d.m_bin.to_numpy(),d.tau_bin.to_numpy()
        d["color"]=np.select([(m==0)&(t<2),(m==0)&(t>=2),(m>=2)&(m<=3)],
                              ["rojo","naranja","amarillo"],default="verde")
        d["split"]="test" if w in audit["test"] else "calibration"
        chunks.append(d.groupby(["split","day","color"],as_index=False)[["n_pairs","n_lab_300"]].sum())
    return pd.concat(chunks,ignore_index=True)


def draw_fig4a(rows,cis):
    xs=np.arange(4);width=.35
    fig,ax=plt.subplots(figsize=(3.9,3.5))
    for k,split in enumerate(["calibration","test"]):
        rr=rows.loc[rows.split.eq(split)].set_index("color").loc[ORDER]
        rate=rr.next_clearance_rate.to_numpy()*100
        low,high=cis[split]*100
        pos=xs+(k-.5)*width
        ax.bar(pos,rate,width,color=[COLOURS[c] for c in ORDER],edgecolor="black",linewidth=.65,
               hatch="" if k==0 else "///",alpha=.88)
        ax.errorbar(pos,rate,yerr=np.vstack([rate-low,high-rate]),fmt="none",
                    ecolor="#202020",elinewidth=.8,capsize=2)
        for x,h,n in zip(pos,high,rr.pair_instants):
            ax.text(x,h+1.2,f"n={int(n):,}",ha="center",va="bottom",rotation=90,fontsize=5.5)
    ax.set_xticks(xs,[EN[c] for c in ORDER]);ax.set_ylim(0,72)
    ax.set_ylabel("Pair-instants with later eligible order (%)")
    ax.legend(handles=[Rectangle((0,0),1,1,facecolor="#d9d9d9",edgecolor="black",label="Calibration (9 weeks)"),
                       Rectangle((0,0),1,1,facecolor="#d9d9d9",edgecolor="black",hatch="///",label="Test (4 weeks)")],
              loc="upper right",frameon=True,framealpha=1,edgecolor="#555",fontsize=6.5)
    fig.tight_layout();save(fig,"fig04a_label_rate")


def fig4():
    rows=pd.read_csv(S20/"stage20_by_color.csv").query("split in ['calibration','test']")
    daily=daily_rates();agg=daily.groupby(["split","color"])[["n_pairs","n_lab_300"]].sum()
    for r in rows.itertuples():
        q=agg.loc[(r.split,r.color)]
        check(q.n_pairs==r.pair_instants and q.n_lab_300==r.with_next_clearance,"Stage 16/20 daily count")
    tiers=pd.read_csv(CAL/"scheme_tiers.csv")
    tiers=tiers.loc[(tiers.predictor=="R")&(tiers.W==300)&
                   (tiers.scheme=="4 colores (final, amarillo 1.2-2)")].copy()
    tiers["split"]=tiers["split"].map({"calibracion (9 sem)":"calibration","test (4 sem)":"test"})
    tiers=tiers.loc[tiers.split.isin(["calibration","test"])]
    rng=np.random.default_rng(1919)
    cis={}
    for split in ["calibration","test"]:
        dd=daily.loc[daily.split.eq(split)]
        n=dd.pivot(index="day",columns="color",values="n_pairs").fillna(0)[ORDER].to_numpy()
        y=dd.pivot(index="day",columns="color",values="n_lab_300").fillna(0)[ORDER].to_numpy()
        idx=rng.integers(0,len(n),(400,len(n)))
        rates=y[idx].sum(1)/n[idx].sum(1)
        cis[split]=np.quantile(rates,[.025,.975],axis=0)
    xs=np.arange(4)
    draw_fig4a(rows,cis)
    fig,bx=plt.subplots(figsize=(3.7,3.4))
    for k,split in enumerate(["calibration","test"]):
        tt=tiers.set_index(["split","tier"]).loc[[(split,c) for c in ORDER]]
        lift=tt.lift.to_numpy();lo=tt.lift_lo.to_numpy();hi=tt.lift_hi.to_numpy()
        bx.errorbar(xs+(k-.5)*.16,lift,yerr=np.vstack([lift-lo,hi-lift]),fmt="o" if k==0 else "s",
                    color="#2d3540" if k==0 else "#7e4b32",capsize=2,markersize=4,
                    label="Calibration" if k==0 else "Test")
    bx.set_xticks(xs,[EN[c] for c in ORDER]);bx.set_ylim(.85,3.55)
    bx.set_ylabel("Lift vs green (density adjusted)");bx.axhline(1,color="#888",lw=.8)
    bx.legend(frameon=False,loc="upper left")
    fig.tight_layout();save(fig,"fig04b_relative_lift")


def distinct_pairs_per_order(attr):
    """Reconstruct every (T,T+300] order-to-pair link from canonical week files."""
    out=np.zeros(len(attr),dtype=np.int32)
    counts=np.zeros(len(attr),dtype=np.int32)
    source=pd.read_parquet(ROOT/"data/processed/scat/03_clearances/scat_clearances.parquet",
                           columns=["clearance_id","clearance_epoch"])
    exact_epoch=attr.clearance_id.map(source.set_index("clearance_id").clearance_epoch)
    check(exact_epoch.notna().all(),"All Stage 20 orders present in Stage 03")
    for week,g in attr.groupby("week",sort=False):
        p=ROOT/f"results/stage20/pairs/pairs_{week}.parquet"
        d=pd.read_parquet(p,columns=["T","flight_key_a","flight_key_b","source_member_a","source_member_b"])
        time=d["T"].to_numpy(float)
        check(np.all(time[1:]>=time[:-1]),f"Pair times sorted: {week}")
        ka=d.flight_key_a.astype(str).str.cat(d.source_member_a.astype(str),sep="\x1f")
        kb=d.flight_key_b.astype(str).str.cat(d.source_member_b.astype(str),sep="\x1f")
        keys=pd.Index(pd.concat([ka,kb],ignore_index=True).unique())
        aa=keys.get_indexer(ka).astype(np.int32);bb=keys.get_indexer(kb).astype(np.int32)
        order_keys=g.flight.astype(str).str.cat(g.source_member.astype(str),sep="\x1f")
        order_codes=keys.get_indexer(order_keys)
        times=exact_epoch.loc[g.index].to_numpy(float)
        # Use exact Stage 03 epochs: the Stage 20 CSV time_utc is rounded to seconds,
        # which changes matching at the boundary of a 300-second window.
        for pos,code,t in zip(g.index,order_codes,times):
            if code<0:continue
            l=np.searchsorted(time,t-300,side="left")
            r=np.searchsorted(time,t,side="left")
            mask_a=aa[l:r]==code;mask_b=bb[l:r]==code
            counts[pos]=int(mask_a.sum()+mask_b.sum())
            if counts[pos]:out[pos]=len(np.unique(np.r_[bb[l:r][mask_a],aa[l:r][mask_b]]))
        delta=counts[g.index]-g.matched_pair_instants.to_numpy()
        check(np.array_equal(delta,np.zeros(len(delta),dtype=delta.dtype)),
              f"Distinct-pair reconstruction differs from Stage 20: {week}; "
              f"{np.count_nonzero(delta)} rows, max |delta|={np.max(np.abs(delta))}; "
              f"examples={list(zip(g.clearance_id.to_numpy()[delta!=0][:5],delta[delta!=0][:5]))}")
        print("distinct pairs:",week,len(g),flush=True)
    return out


def draw_fig5ab(counts,pairs,linked):
    n0=int((~linked).sum())
    fig,ax=plt.subplots(figsize=(3.0,3.0))
    ax.hist(counts[linked],bins=np.arange(.5,201.5,5),color="#6686a0",edgecolor="white",linewidth=.4)
    top=ax.get_ylim()[1];ax.set_ylim(0,top*1.45)
    ax.plot([65,65],[0,top*1.08],color="#9f3434",lw=1.3,label="Median (65)")
    ax.plot([108,108],[0,top*1.08],color="#7a562e",lw=1.3,ls="--",label="90th percentile (108)")
    ax.set(xlabel="Pair-instants linked to one clearance",ylabel="Number of clearances",xlim=(0,195))
    ax.legend(loc="upper right",frameon=True,framealpha=1,edgecolor="#555",fontsize=6.3)
    fig.tight_layout();save(fig,"fig05a_links_per_order")
    fig,bx=plt.subplots(figsize=(3.0,3.0))
    bx.hist(pairs[linked],bins=np.arange(.5,pairs.max()+1.5,1),color="#7f8f73",edgecolor="white",linewidth=.25)
    bx.set(xlabel="Distinct flight pairs linked to one clearance",ylabel="Number of clearances")
    fig.tight_layout();save(fig,"fig05b_pairs_per_order")


def fig5():
    attr=pd.read_csv(S20/"stage20_clearance_attribution.csv.gz")
    check(len(attr)==51950 and attr.clearance_id.nunique()==51950,"Eligible clearance count")
    counts=attr.matched_pair_instants.to_numpy(int); linked=counts>0
    check(linked.sum()==47339 and counts.sum()==3176638,"Attribution totals")
    check(np.median(counts[linked])==65 and np.quantile(counts[linked],.9)==108,"Multiplicity quantiles")
    pairs=distinct_pairs_per_order(attr)
    bins=np.r_[-.5,np.arange(.5,201.5,5)]
    draw_fig5ab(counts,pairs,linked)
    fig,cx=plt.subplots(figsize=(2.2,2.9));cx.axis("off")
    y=[.82,.52,.2]; labels=["51,950\neligible orders","47,339\nlinked orders","3,176,638\norder–pair-instant links"]
    for yy,lbl in zip(y,labels):
        cx.add_patch(FancyBboxPatch((.04,yy-.09),.92,.18,boxstyle="round,pad=.02",facecolor="#e9edf2",edgecolor="#617386"))
        cx.text(.5,yy,lbl,ha="center",va="center",fontsize=7,transform=cx.transAxes)
    for ya,yb in zip(y[:-1],y[1:]):
        cx.annotate("",xy=(.5,yb+.12),xytext=(.5,ya-.12),xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->",color="#617386"))
    fig.tight_layout();save(fig,"fig05c_link_totals")


def fig6():
    e=pd.read_csv(S20/"stage20_episodes.csv")
    e=e.loc[e.split.eq("test")&e.highest_color.eq("rojo")]
    cats=[((~e.prior_to_episode_start)&e.any_next_clearance).sum(),
          (e.prior_to_episode_start&e.any_next_clearance).sum(),
          (e.prior_to_episode_start&(~e.any_next_clearance)).sum(),
          ((~e.prior_to_episode_start)&(~e.any_next_clearance)).sum()]
    check(cats==[642,213,187,821] and sum(cats)==1863,"Red test order categories")
    draw_fig6(cats)


def draw_fig6(cats):
    fig,ax=plt.subplots(figsize=(6.6,2.6))
    labels=["Later only","Both","Prior only","Neither"]
    colors=["#a45454","#75547a","#53768e","#bdbdbd"]
    left=0
    for n,label,color in zip(cats,labels,colors):
        ax.barh([0],[n],left=left,height=.55,color=color,edgecolor="white",linewidth=1.5)
        ax.text(left+n/2,0,f"{n:,}\n({100*n/1863:.1f}%)",ha="center",va="center",
                color="white" if label!="Neither" else "#222",fontsize=9,weight="bold",linespacing=1.15)
        ax.text(left+n/2,.40,label,ha="center",va="bottom",fontsize=8.5)
        left+=n
    ax.set(xlim=(0,1863),ylim=(-.45,.75),yticks=[],xlabel="Test red encounter episodes (n = 1,863)")
    ax.spines["left"].set_visible(False)
    fig.tight_layout();save(fig,"fig06_red_episode_orders")



def fig7():
    x=pd.read_csv(S21/"stage21_encounters.csv")
    order=["amarillo","naranja","rojo"]
    g=x.groupby("color")
    total=np.array([len(g.get_group(c)) for c in order])
    ev=np.array([int(g.get_group(c).observed_evaluable.sum()) for c in order])
    check(total.tolist()==[39351,9039,6051] and ev.tolist()==[19905,5802,1744],"Coverage totals")
    y=np.arange(3)
    fig,ax=plt.subplots(figsize=(3.6,2.9))
    ax.barh(y,ev,color=[COLOURS[c] for c in order],edgecolor="black",linewidth=.5,label="Evaluable")
    ax.barh(y,total-ev,left=ev,color="#d4d4d4",edgecolor="black",linewidth=.5,hatch="//",label="Not evaluable")
    ax.set_yticks(y,[EN[c] for c in order]);ax.invert_yaxis();ax.set_xlabel("Encounter episodes (all 13 weeks)")
    ax.legend(frameon=False,loc="lower right",fontsize=6.5)
    for yy,n,k in zip(y,total,ev):ax.text(n+300,yy,f"{k:,}/{n:,}\n({k/n:.1%})",va="center",fontsize=6.5)
    ax.set_xlim(0,66000)
    fig.tight_layout();save(fig,"fig07a_coverage")
    colors2=["#4d718b","#e0a65d","#a8a8a8"]
    fig,bx=plt.subplots(figsize=(3.6,2.9))
    for yy,c in enumerate(["naranja","rojo"]):
        q=g.get_group(c);q=q[q.observed_evaluable]
        below=int(q.observed_below_1.sum());absent=int(q.predicted_loss_not_observed.sum());censored=len(q)-below-absent
        check((below,absent,censored)==({"naranja":(470,5332,0),"rojo":(703,1040,1)}[c]),"Outcome partition")
        left=0
        for n,col,hatch in [(below,colors2[0],""),(absent,colors2[1],"//"),(censored,colors2[2],"xx")]:
            if n:bx.barh(yy,n,left=left,color=col,edgecolor="black",linewidth=.5,hatch=hatch,height=.5)
            if n>50:bx.text(left+n/2,yy,f"{n:,}",ha="center",va="center",fontsize=7,
                            bbox=dict(boxstyle="round,pad=.15",fc="white",ec="none",alpha=.85))
            left+=n
    bx.set_yticks([0,1],["Orange","Red"]);bx.invert_yaxis()
    bx.set_xlabel("Evaluable predicted-loss episodes")
    bx.legend(handles=[Rectangle((0,0),1,1,facecolor=colors2[0],label="Observed m < 1"),
                       Rectangle((0,0),1,1,facecolor=colors2[1],hatch="//",label="m ≥ 1; full coverage"),
                       Rectangle((0,0),1,1,facecolor=colors2[2],hatch="xx",label="m ≥ 1; partial coverage")],
              frameon=False,loc="lower right",fontsize=6.5)
    fig.tight_layout();save(fig,"fig07b_observed_screen")


def fig8():
    bench=ROOT/"results/benchmarks"
    m=pd.read_csv(bench/"metrics.csv")
    r=pd.read_csv(bench/"reliability.csv")
    q=m[m.weight.eq("episode")].copy()
    check(set(q.model)=={"M0","M1","M2","M3","M4","M5_3","M5_4","M5_5"},
          "Benchmark model set is incomplete")
    q=q.sort_values("brier")
    fig,ax=plt.subplots(figsize=(3.7,3.2))
    y=np.arange(len(q))
    delta=q.delta_brier_vs_M0.to_numpy(float)
    ax.errorbar(delta,y,xerr=np.vstack([delta-q.delta_brier_ci_low,
                                        q.delta_brier_ci_high-delta]),
                fmt="o",color="#315a76",capsize=2,lw=1,label="ΔBrier vs M0 (95% CI)")
    ax.axvline(0,color="#555",lw=.8,linestyle="--")
    ax.set_yticks(y,q.model.str.replace("_"," K=",regex=False))
    ax.legend(loc="lower left",frameon=True,framealpha=1,edgecolor="#555",fontsize=7)
    ax.invert_yaxis();ax.set_xlabel("ΔBrier versus M0 (95% day-bootstrap interval)")
    fig.tight_layout();save(fig,"fig08a_brier")
    fig,bx=plt.subplots(figsize=(3.7,3.2))
    for model,color in [("M0","#3f6a82"),("M2","#ad732d"),("M3","#6b5291"),("M4","#b14b4b")]:
        g=r[r.weight.eq("episode") & r.model.eq(model)].sort_values("mean_predicted")
        bx.scatter(g.mean_predicted,g.observed_rate,s=14,label=model,color=color,zorder=3)
    bx.plot([0,1],[0,1],"--",color="#777",lw=.8)
    bx.set(xlabel="Mean predicted probability",ylabel="Observed order rate",xlim=(0,.8),ylim=(0,.8))
    bx.legend(loc="upper left",frameon=True,framealpha=1,edgecolor="#555",ncol=2,fontsize=7)
    fig.tight_layout();save(fig,"fig08b_reliability")


FIGS={1:fig1,2:fig2,3:fig3,4:fig4,5:fig5,6:fig6,7:fig7,8:fig8}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--figure",default="all",choices=["all"]+[str(i) for i in FIGS])
    a=p.parse_args()
    for i in (FIGS if a.figure=="all" else [int(a.figure)]):
        print(f"Generating figure {i}",flush=True);FIGS[i]()
