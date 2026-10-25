"""Build the Stage 11 scientific atlas from Stage 11 tables, then write report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

from stage11_core import BASE, OUT, WEEK, binned_rate, save

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,
                     "axes.labelsize":10,"figure.dpi":120,"savefig.dpi":300})
FIG = OUT / "figures"
H = np.arange(0,52,2)
V = np.arange(0,6250,250)


def finish(fig, folder: str, name: str, title: str, data: pd.DataFrame, metadata: dict) -> str:
    directory=FIG/folder;directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f"{name}.png",dpi=300,bbox_inches="tight")
    fig.savefig(directory/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)
    data.to_csv(directory/f"{name}.csv",index=False)
    (directory/f"{name}.json").write_text(json.dumps({"id":name,"title":title,"week":WEEK,**metadata},indent=2,default=str)+"\n")
    return f"figures/{folder}/{name}.png"


def matrix(frame,x,y,weights=None, xedges=H,yedges=V):
    xval=pd.to_numeric(frame[x],errors="coerce").to_numpy(float)
    yval=pd.to_numeric(frame[y],errors="coerce").to_numpy(float)
    valid=np.isfinite(xval)&np.isfinite(yval)
    w=np.asarray(weights,dtype=float)[valid] if weights is not None else None
    hist,_,_=np.histogram2d(xval[valid],yval[valid],bins=[xedges,yedges],weights=w)
    return hist.T


def mapfig(z, title, xlabel, ylabel, cbar, xedges=H,yedges=V,
           cmap="viridis", norm=None, refs=False, foot=None):
    fig,ax=plt.subplots(figsize=(9.0,6.2),layout="constrained")
    m=ax.pcolormesh(xedges,yedges,np.ma.masked_invalid(z),cmap=cmap,norm=norm,shading="flat")
    if refs:
        ax.axvline(5,color="white",lw=.9,ls="--",alpha=.8)
        ax.axhline(1000,color="white",lw=.9,ls="--",alpha=.8)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=(xedges[0],xedges[-1]),ylim=(yedges[0],yedges[-1]))
    fig.colorbar(m,ax=ax,label=cbar,shrink=.9)
    if foot:fig.text(.01,.005,foot,fontsize=8)
    return fig


def bins_table(z,xedges=H,yedges=V,name="value"):
    xx,yy=np.meshgrid(xedges[:-1],yedges[:-1])
    return pd.DataFrame({"x_low":xx.ravel(),"x_high":(xx+np.diff(xedges)[None,:]).ravel(),
                         "y_low":yy.ravel(),"y_high":(yy+np.diff(yedges)[:,None]).ravel(),name:z.ravel()})


def panel01(a):
    refs=[]
    i=a.loc[a.ATC_INTERVENTION.eq(1)];c=a.loc[a.ATC_INTERVENTION.eq(0)]
    zi=matrix(i,"horizontal_separation_nm","vertical_separation_ft")
    zc=matrix(c,"horizontal_separation_nm","vertical_separation_ft")
    scale=max(zi.max(),zc.max())
    for name,z,title in [("01_interventions",zi,"Interventions: nearest pre-event pair"),
                         ("02_matched_controls",zc,"Matched controls: nearest anchor pair")]:
        fig=mapfig(z,title,"Horizontal separation (NM)","Absolute vertical separation (ft)","Windows per bin",
                   cmap="magma",norm=plt.Normalize(0,scale),refs=True)
        refs.append(finish(fig,"01_intervention_maps",name,title,bins_table(z,name="windows"),
                           {"population":"11A matched cohort","n_windows":len(i) if name.startswith("01") else len(c),
                            "in_plot":int(z.sum()),"out_of_plot_or_missing":len(i)-int(z.sum()) if name.startswith("01") else len(c)-int(z.sum()),
                            "source":"10_dynamic_rule_cruise/snapshot_reconstruction.parquet", "anchor":"backward-as-of native states",
                            "dashed_references":"5 NM and 1000 ft are visual references, not conflict definitions", "bin_nm":2,"bin_ft":250}))
    diff=zi/max(len(i),1)-zc/max(len(c),1)
    lim=max(abs(diff.min()),abs(diff.max()),1e-6)
    fig=mapfig(diff,"Intervention minus control window fraction","Horizontal separation (NM)","Absolute vertical separation (ft)",
               "Difference in fraction of all arm windows",cmap="coolwarm",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),refs=True)
    refs.append(finish(fig,"01_intervention_maps","03_density_difference","Normalized density difference",
                       bins_table(diff,name="fraction_difference"),{"population":"11A; separate arm denominators","n_intervention":len(i),"n_control":len(c)}))
    fig=mapfig(zi,"Interventions: log count","Horizontal separation (NM)","Absolute vertical separation (ft)",
               "Windows per occupied bin (log)",cmap="magma",norm=LogNorm(vmin=1,vmax=max(scale,1)),refs=True)
    refs.append(finish(fig,"01_intervention_maps","04_interventions_log","Log intervention counts",
                       bins_table(zi,name="windows"),{"population":"11A","zero_cells":"masked","n_windows":len(i)}))
    return refs


def table_grid(t,field,xedges=H,yedges=V):
    return t.sort_values(["y_low","x_low"])[field].to_numpy(float).reshape(len(yedges)-1,len(xedges)-1)


def panel02(risk):
    g=risk.loc[risk.has_neighbour]
    z=matrix(g,"horizontal_nm","vertical_ft",g.exposure_hours)
    n=matrix(g,"horizontal_nm","vertical_ft")
    q=matrix(g,"horizontal_nm","vertical_ft",(g.focal_age_s>10).astype(float))
    refs=[]
    for name,title,a,unit,cmap in [
        ("01_exposure_hours","Observed focal-aircraft exposure",z,"Aircraft-hours per bin","viridis"),
        ("02_exposure_fraction","Relative exposure across all observed focal time",z/max(risk.exposure_hours.sum(),1),"Fraction of all observed focal time","viridis"),
        ("03_snapshot_coverage","Snapshot support",n,"Focal snapshots per bin","Blues"),
        ("04_state_age_quality","State age > 10 s",np.divide(q,n,out=np.full_like(q,np.nan),where=n>0),"Fraction of focal states aged > 10 s","cividis")]:
        fig=mapfig(a,title,"Nearest horizontal separation (NM)","Nearest absolute vertical separation (ft)",unit,cmap=cmap,refs=True)
        refs.append(finish(fig,"02_traffic_exposure",name,title,bins_table(a,name="value"),
                           {"population":"11B all first-week cruise focal snapshots","n_snapshots":len(risk),
                            "total_focal_hours":float(risk.exposure_hours.sum()),"in_plot_hours":float(z.sum()),
                            "unit":"one observable focal aircraft at each 30-second tick = 30/3600 h",
                            "source":"02_traffic/scat_states.parquet","state_max_age_s":15}))
    return refs


def panel03(rates,risk):
    t=rates["h2_v250"]
    z=table_grid(t,"rate_per_1000h").copy()
    supported=table_grid(t,"supported").astype(bool)
    z[~supported]=np.nan
    refs=[]
    for name,title,a,unit,cmap in [
        ("01_supported_rate","Intervention rate where supported",z,"Events per 1,000 observed aircraft-hours","plasma"),
        ("02_ci_width","Rate-interval width",np.where(supported,table_grid(t,"ci_high_per_1000h")-table_grid(t,"ci_low_per_1000h"),np.nan),"95% interval width / 1,000 h","cividis"),
        ("03_statistical_support","Support: 10 h, 3 events, 20 flights",supported.astype(float),"Supported cell (1=yes)","Greens"),
        ("04_event_share","Absolute event frequency",table_grid(t,"events"),"Actionable events per bin","magma")]:
        fig=mapfig(a,title,"Nearest horizontal separation (NM)","Nearest absolute vertical separation (ft)",unit,cmap=cmap,refs=True)
        refs.append(finish(fig,"03_intervention_rates",name,title,t,
                           {"population":"11B first-week observed 30 s risk set","source_table":"tables/11b_rate_map_h2_v250.csv",
                           "event_unit":"actionable clearance, any type","interval":"Envelope of Garwood count interval and focal-flight cluster influence interval; between-flight dependence not covered",
                            "suppression":"rate and interval masked below 10 h, 3 events or 20 flights"}))
    for name,key,xlabel in [("05_horizontal_rate","horizontal_2","Nearest horizontal separation (NM)"),
                             ("06_vertical_rate","vertical_250","Nearest absolute vertical separation (ft)")]:
        d=rates[key];x=(d.x_low+d.x_high)/2;ok=d.exposure_hours>=10
        fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
        ax.plot(x[ok],d.rate_per_1000h[ok],marker="o",ms=3,color="#175e91")
        ax.fill_between(x[ok],d.ci_low_per_1000h[ok],d.ci_high_per_1000h[ok],alpha=.2,color="#175e91")
        ax.set(xlabel=xlabel,ylabel="Actionable events / 1,000 observed aircraft-hours",title="Population intervention intensity by separation")
        ax.grid(alpha=.2)
        refs.append(finish(fig,"03_intervention_rates",name,"Population rate by separation",d,
                           {"population":"11B","event":"all actionable clearances","min_hours_shown":10,"interval":"Garwood + focal-flight cluster envelope"}))
    fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
    for label,color in [("level","#156a97"),("heading","#ba5428"),("speed","#40824e")]:
        d=rates[label];x=(d.x_low+d.x_high)/2;ok=d.exposure_hours>=10
        ax.plot(x[ok],d.rate_per_1000h[ok],label=label.upper(),color=color)
        ax.fill_between(x[ok],d.ci_low_per_1000h[ok],d.ci_high_per_1000h[ok],color=color,alpha=.12)
    ax.set(xlabel="Nearest horizontal separation (NM)",ylabel="Events / 1,000 observed aircraft-hours",title="Clearance type specific intensities")
    ax.legend();ax.grid(alpha=.2)
    refs.append(finish(fig,"03_intervention_rates","07_type_rate","Rate by clearance type",
                       pd.concat([rates[k].assign(type=k) for k in ("level","heading","speed")]),
                       {"population":"11B","denominator":"same focal aircraft-hours in every type curve"}))
    fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
    for key,color in [("CONVERGING","#bc4b23"),("DIVERGING_OR_TANGENTIAL","#176895")]:
        d=rates[key];x=(d.x_low+d.x_high)/2;ok=d.exposure_hours>=10
        ax.plot(x[ok],d.rate_per_1000h[ok],label=key.replace("_"," ").title(),color=color)
        ax.fill_between(x[ok],d.ci_low_per_1000h[ok],d.ci_high_per_1000h[ok],color=color,alpha=.12)
    ax.set(xlabel="Nearest horizontal separation (NM)",ylabel="Events / 1,000 observed aircraft-hours",title="Converging versus other encounters")
    ax.legend();ax.grid(alpha=.2)
    refs.append(finish(fig,"03_intervention_rates","08_motion_rate","Rate by relative motion",
                       pd.concat([rates[k].assign(motion=k) for k in ("CONVERGING","DIVERGING_OR_TANGENTIAL")]),
                       {"population":"11B","converging":"positive radial horizontal closing rate"}))
    return refs


def panel04(a):
    refs=[]
    for field,xmax,binwidth,unit,name in [("horizontal_separation_nm",50,2,"NM","01_horizontal_distribution"),
                                           ("vertical_separation_ft",6000,250,"ft","02_vertical_distribution")]:
        bins=np.arange(0,xmax+binwidth,binwidth);fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
        tables=[]
        for label,color in [(1,"#c35025"),(0,"#1e6c98")]:
            s=pd.to_numeric(a.loc[a.ATC_INTERVENTION.eq(label),field],errors="coerce").dropna()
            count,_=np.histogram(s,bins);dens=count/max(len(s),1)
            ax.step(bins[:-1],dens,where="post",label=f"{'Intervention' if label else 'Control'} (n={len(s):,})",color=color,lw=2)
            tables.append(pd.DataFrame({"arm":label,"low":bins[:-1],"high":bins[1:],"count":count,"fraction_all_valid":dens}))
        ax.set(xlabel=f"Nearest pre-anchor separation ({unit})",ylabel="Fraction of valid arm windows / bin",title="Matched-cohort separation distribution")
        ax.legend();ax.grid(alpha=.2)
        refs.append(finish(fig,"04_distributions",name,"Matched separation distributions",pd.concat(tables),
                           {"population":"11A matched 50/50 cohort","out_of_range":"retained in source; shown counts sum only in plot range"}))
        fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
        ec=[]
        for label,color in [(1,"#c35025"),(0,"#1e6c98")]:
            s=np.sort(pd.to_numeric(a.loc[a.ATC_INTERVENTION.eq(label),field],errors="coerce").dropna().to_numpy(float))
            ax.step(s,np.arange(1,len(s)+1)/len(s),where="post",label="Intervention" if label else "Control",color=color)
            ec.append(pd.DataFrame({"arm":label,"value":s,"ecdf":np.arange(1,len(s)+1)/len(s)}))
        ax.set(xlabel=f"Nearest pre-anchor separation ({unit})",ylabel="Empirical cumulative fraction",title="Matched cohort ECDF",xlim=(0,xmax))
        ax.legend();ax.grid(alpha=.2)
        refs.append(finish(fig,"04_distributions",name.replace("distribution","ecdf"),"Matched separation ECDF",pd.concat(ec),
                           {"population":"11A","full_range_in_csv":True}))
    i=a.loc[a.ATC_INTERVENTION.eq(1) & a.clearance_type.isin(["LEVEL","HEADING","SPEED"])]
    for field,unit,name in [("horizontal_separation_nm","NM","05_horizontal_by_type"),("vertical_separation_ft","ft","06_vertical_by_type")]:
        groups=[i.loc[i.clearance_type.eq(t),field].dropna().to_numpy(float) for t in ("LEVEL","HEADING","SPEED")]
        fig,ax=plt.subplots(figsize=(8,5.2),layout="constrained")
        ax.boxplot(groups,tick_labels=[f"{t}\n(n={len(g)})" for t,g in zip(("LEVEL","HEADING","SPEED"),groups)],showfliers=False)
        ax.set(ylabel=f"Nearest separation ({unit})",title="Intervention geometry by authorization type")
        ax.grid(axis="y",alpha=.2)
        d=pd.DataFrame({"type":[t for t,g in zip(("LEVEL","HEADING","SPEED"),groups) for _ in g],"value":np.concatenate(groups)})
        refs.append(finish(fig,"04_distributions",name,"Separation by authorization type",d,
                           {"population":"11A intervention arm","whiskers":"1.5 IQR; fliers hidden visually, retained in CSV"}))
    summary=[]
    for (arm,typ),g in a.groupby(["ATC_INTERVENTION","clearance_type"],dropna=False):
        for field in ("horizontal_separation_nm","vertical_separation_ft","cpa_nm","tcpa_min"):
            s=pd.to_numeric(g[field],errors="coerce").dropna()
            summary.append({"arm":arm,"type":typ,"field":field,"n":len(s),"p05":s.quantile(.05),"p25":s.quantile(.25),
                            "median":s.median(),"p75":s.quantile(.75),"p95":s.quantile(.95)})
    save(pd.DataFrame(summary),"tables/11a_separation_percentiles.csv")
    return refs


def panel05(trace):
    refs=[]
    fields=[("horizontal_separation_nm","Horizontal separation (NM)","01_horizontal"),
            ("vertical_separation_ft","Absolute vertical separation (ft)","02_vertical"),
            ("horizontal_closing_rate_nm_min","Horizontal closing (NM/min)","03_horizontal_closing"),
            ("vertical_closing_rate_fpm","Vertical closing (ft/min)","04_vertical_closing"),
            ("cpa_nm","Horizontal CPA (NM)","05_cpa"),
            ("tcpa_min","Signed TCPA (min)","06_tcpa")]
    for field,ylabel,name in fields:
        d=trace.groupby(["ATC_INTERVENTION","relative_bin_s"])[field].agg(n="count",p05=lambda x:x.quantile(.05),
            p25=lambda x:x.quantile(.25),median="median",p75=lambda x:x.quantile(.75),p95=lambda x:x.quantile(.95)).reset_index()
        fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
        for arm,color in [(1,"#c35025"),(0,"#1e6c98")]:
            s=d.loc[d.ATC_INTERVENTION.eq(arm)].sort_values("relative_bin_s")
            ax.plot(s.relative_bin_s,s["median"],label="Intervention" if arm else "Control",color=color)
            ax.fill_between(s.relative_bin_s,s.p25,s.p75,color=color,alpha=.2)
        ax.axvline(0,color="black",ls="--",lw=.8)
        ax.set(xlabel="Time relative to clearance / control anchor (s)",ylabel=ylabel,title=f"Fixed-neighbour trajectory: {ylabel}")
        ax.legend();ax.grid(alpha=.2)
        refs.append(finish(fig,"05_temporal_evolution",name,"Fixed pair temporal evolution",d,
                           {"population":"11A windows with a pre-event nearest neighbour","interval":"25–75 percentile; 5–95 and n in CSV",
                            "pair_identity":"same neighbour selected at last pre-event Stage05 pair row","post_event":"observed association, not causal"}))
    d=trace.loc[trace.ATC_INTERVENTION.eq(1)&trace.clearance_type.isin(["LEVEL","HEADING","SPEED"])].groupby(
        ["clearance_type","relative_bin_s"]).horizontal_separation_nm.agg(n="count",median="median",p25=lambda x:x.quantile(.25),p75=lambda x:x.quantile(.75)).reset_index()
    fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
    for typ,color in [("LEVEL","#176895"),("HEADING","#ba5428"),("SPEED","#40824e")]:
        s=d.loc[d.clearance_type.eq(typ)].sort_values("relative_bin_s")
        ax.plot(s.relative_bin_s,s["median"],label=typ,color=color)
        ax.fill_between(s.relative_bin_s,s.p25,s.p75,color=color,alpha=.1)
    ax.axvline(0,color="black",ls="--",lw=.8)
    ax.set(xlabel="Time relative to authorization (s)",ylabel="Horizontal separation (NM)",title="Fixed-neighbour trajectory by authorization type")
    ax.legend();ax.grid(alpha=.2)
    refs.append(finish(fig,"05_temporal_evolution","07_by_clearance_type","Trajectory by clearance type",d,
                       {"population":"11A intervention windows","n_by_time_in_csv":True}))
    return refs


def panel06(a,rates,risk):
    q=a.loc[a.tcpa_min.between(1/60,30)&a.cpa_nm.between(0,50)&a.predicted_min_vertical_sep_ft.notna()].copy()
    xe=np.arange(0,31,1);ye=np.arange(0,51,2)
    refs=[];mat={}
    for arm,title,name in [(1,"Interventions: same-pair future CPA–TCPA","01_intervention_cpa_tcpa"),
                            (0,"Controls: same-pair future CPA–TCPA","02_control_cpa_tcpa")]:
        g=q.loc[q.ATC_INTERVENTION.eq(arm)]
        z=matrix(g,"tcpa_min","cpa_nm",xedges=xe,yedges=ye);mat[arm]=z
        fig=mapfig(z,title,"TCPA (min)","Horizontal CPA (NM)","Eligible windows / bin",xe,ye,cmap="magma")
        refs.append(finish(fig,"06_cpa_tcpa_maps",name,title,bins_table(z,xe,ye,"windows"),
                           {"population":"11A","pair":"fixed nearest horizontal pair in Stage10B backward-as-of snapshot","n_quality":len(g),
                            "quality":"1 s <= TCPA <= 30 min; 0 <= CPA <= 50 NM; finite predicted vertical separation"}))
    diff=mat[1]/max((a.ATC_INTERVENTION==1).sum(),1)-mat[0]/max((a.ATC_INTERVENTION==0).sum(),1)
    lim=max(abs(diff.min()),abs(diff.max()),1e-6)
    fig=mapfig(diff,"Intervention minus control CPA–TCPA fraction","TCPA (min)","Horizontal CPA (NM)",
               "Difference / all arm windows",xe,ye,cmap="coolwarm",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    refs.append(finish(fig,"06_cpa_tcpa_maps","03_difference","Matched CPA–TCPA difference",bins_table(diff,xe,ye,"difference"),
                       {"population":"11A","quality_exclusions":"kept in source and reported separately"}))
    t=rates["cpa_tcpa"];z=t.sort_values(["y_low","x_low"]).rate_per_1000h.to_numpy(float).reshape(len(ye)-1,len(xe)-1)
    supported=t.sort_values(["y_low","x_low"]).supported.to_numpy(bool).reshape(len(ye)-1,len(xe)-1)
    fig=mapfig(np.where(supported,z,np.nan),"Population rate by pre-event CPA–TCPA","TCPA (min)","Horizontal CPA (NM)",
               "Events / 1,000 observed aircraft-hours",xe,ye,cmap="plasma")
    refs.append(finish(fig,"06_cpa_tcpa_maps","04_supported_rate","Population CPA–TCPA rate",t,
                       {"population":"11B","quality":"same-pair nearest geometry; 1 s to 30 min future TCPA","suppression":"10 h, 3 events, 20 flights"}))
    g=q.loc[q.horizontal_closing_rate_nm_min.gt(0)]
    z=matrix(g,"tcpa_min","cpa_nm",xedges=xe,yedges=ye)
    fig=mapfig(z,"Converging matched anchors","Future TCPA (min)","Horizontal CPA (NM)",
               "Eligible windows / bin",xe,ye,cmap="magma")
    refs.append(finish(fig,"06_cpa_tcpa_maps","05_converging","Future CPA–TCPA for converging pairs",bins_table(z,xe,ye,"windows"),
                       {"population":"11A","n_quality":len(g),"motion":"positive radial closing"}))
    # With linear relative motion, positive future TCPA implies radial closing.
    # Show divergence on its own signed (negative) TCPA axis; its Stage05-style
    # clipped CPA is current distance, not a future minimum.
    xneg=np.arange(-30,1,1)
    g=a.loc[a.tcpa_min.between(-30,0,inclusive="left") & a.horizontal_closing_rate_nm_min.le(0)
            & a.cpa_nm.between(0,50)]
    z=matrix(g,"tcpa_min","cpa_nm",xedges=xneg,yedges=ye)
    fig=mapfig(z,"Diverging/tangential: negative TCPA","Signed TCPA (min; past)","Current horizontal distance (NM)",
               "Eligible windows / bin",xneg,ye,cmap="magma")
    refs.append(finish(fig,"06_cpa_tcpa_maps","06_diverging","Signed TCPA for diverging pairs",bins_table(z,xneg,ye,"windows"),
                       {"population":"11A","n_quality":len(g),"motion":"nonpositive radial closing",
                        "warning":"negative TCPA; CPA clipped to current horizontal separation, not a future minimum"}))
    d=q.assign(vertical_cpa_band=pd.cut(q.predicted_min_vertical_sep_ft,[0,1000,3000,6000,np.inf],right=False)).groupby(
        ["ATC_INTERVENTION","vertical_cpa_band"],observed=False).size().rename("windows").reset_index()
    d["vertical_cpa_band"]=d.vertical_cpa_band.astype(str)
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    p=d.pivot(index="vertical_cpa_band",columns="ATC_INTERVENTION",values="windows").fillna(0)
    p.plot.bar(ax=ax,color=["#1e6c98","#c35025"])
    ax.set(xlabel="Predicted absolute vertical separation at CPA (ft)",ylabel="Matched windows",title="Vertical context of same-pair CPA")
    ax.legend(["Control","Intervention"])
    refs.append(finish(fig,"06_cpa_tcpa_maps","07_vertical_at_cpa","Predicted vertical separation at CPA",d,
                       {"population":"11A","prediction":"linear kinematics; not observed future minimum"}))
    for field,key,xlabel,name in [("cpa_nm","cpa_nm","Horizontal CPA (NM)","08_cpa_rate"),
                                    ("tcpa_min","tcpa_min","TCPA (min)","09_tcpa_rate")]:
        d=rates[key];x=(d.x_low+d.x_high)/2;ok=d.exposure_hours>=10
        fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
        ax.plot(x[ok],d.rate_per_1000h[ok],marker="o",ms=3)
        ax.fill_between(x[ok],d.ci_low_per_1000h[ok],d.ci_high_per_1000h[ok],alpha=.2)
        ax.set(xlabel=xlabel,ylabel="Events / 1,000 observed aircraft-hours",title=f"Population intensity by {xlabel}")
        ax.grid(alpha=.2)
        refs.append(finish(fig,"06_cpa_tcpa_maps",name,"Population rate by predicted geometry",d,
                           {"population":"11B quality eligible nearest-pair snapshots","interval":"Garwood + focal-flight cluster envelope"}))
    return refs


def colours(a,trace):
    refs=[]
    for field,unit,name in [("horizontal_separation_nm","NM","01_horizontal_by_colour"),
                            ("vertical_separation_ft","ft","02_vertical_by_colour"),
                            ("cpa_nm","NM","03_cpa_by_colour"),
                            ("tcpa_min","min","04_tcpa_by_colour")]:
        d=[];fig,ax=plt.subplots(figsize=(8.5,5.1),layout="constrained")
        groups=[];labels=[]
        for col in ("GREEN","YELLOW","ORANGE","RED"):
            s=pd.to_numeric(a.loc[a.focal_dynamic_state.eq(col),field],errors="coerce").dropna().to_numpy(float)
            groups.append(s);labels.append(f"{col}\n(n={len(s)})")
            d.append(pd.DataFrame({"colour":col,"value":s}))
        ax.boxplot(groups,tick_labels=labels,showfliers=False)
        ax.set(ylabel=f"{field.replace('_',' ')} ({unit})",title="Detector colours as descriptive strata")
        ax.grid(axis="y",alpha=.2)
        refs.append(finish(fig,"07_color_comparison",name,"Geometry by detector colour",pd.concat(d),
                           {"population":"11A matched; intervention and control combined","label_caution":"colour is detector output, not conflict truth"}))
    t=trace.groupby(["focal_dynamic_state","relative_bin_s"]).horizontal_separation_nm.agg(n="count",median="median").reset_index()
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    for col,color in [("GREEN","#478550"),("YELLOW","#bd9c2c"),("ORANGE","#bd6a29"),("RED","#a93431")]:
        s=t.loc[t.focal_dynamic_state.eq(col)]
        ax.plot(s.relative_bin_s,s["median"],label=col,color=color)
    ax.axvline(0,color="black",ls="--",lw=.8)
    ax.set(xlabel="Relative time (s)",ylabel="Median same-pair horizontal separation (NM)",title="Detector colour and fixed-pair evolution")
    ax.legend();ax.grid(alpha=.2)
    refs.append(finish(fig,"07_color_comparison","05_temporal_by_colour","Fixed-pair evolution by colour",t,
                       {"population":"11A; intervention and control combined","n_by_time_in_csv":True}))
    d=a.groupby(["focal_dynamic_state","ATC_INTERVENTION"]).size().rename("windows").reset_index()
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    p=d.pivot(index="focal_dynamic_state",columns="ATC_INTERVENTION",values="windows").fillna(0)
    p.reindex(["GREEN","YELLOW","ORANGE","RED","UNKNOWN"]).plot.bar(ax=ax,color=["#1e6c98","#c35025"])
    ax.set(xlabel="Detector state",ylabel="Matched windows",title="Recorded intervention by detector state")
    ax.legend(["Control","Intervention"])
    refs.append(finish(fig,"07_color_comparison","06_intervention_counts","Interventions by detector colour",d,
                       {"population":"11A artificial 50/50 cohort","interpretation":"counts are not population risk"}))
    return refs


def sensitivity(a,rates,risk):
    rows=[]
    for hn in (1,2,5):
        for vf in (100,250,500):
            zi=matrix(a.loc[a.ATC_INTERVENTION.eq(1)],"horizontal_separation_nm","vertical_separation_ft",xedges=np.arange(0,50+hn,hn),yedges=np.arange(0,6000+vf,vf))
            zc=matrix(a.loc[a.ATC_INTERVENTION.eq(0)],"horizontal_separation_nm","vertical_separation_ft",xedges=np.arange(0,50+hn,hn),yedges=np.arange(0,6000+vf,vf))
            rows.append({"horizontal_bin_nm":hn,"vertical_bin_ft":vf,"interventions_in_range":int(zi.sum()),
                         "controls_in_range":int(zc.sum()),"occupied_intervention_cells":int((zi>0).sum()),
                         "occupied_control_cells":int((zc>0).sum()),"peak_intervention_cell_count":int(zi.max())})
    d=pd.DataFrame(rows);save(d,"tables/11_sensitivity_bins.csv")
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    for vf,g in d.groupby("vertical_bin_ft"):
        ax.plot(g.horizontal_bin_nm,g.occupied_intervention_cells,marker="o",label=f"{vf} ft")
    ax.set(xlabel="Horizontal bin width (NM)",ylabel="Occupied intervention cells",title="Map resolution changes apparent occupancy")
    ax.legend(title="Vertical bin");ax.grid(alpha=.2)
    ref=finish(fig,"08_sensitivity","01_bin_resolution","Map-bin sensitivity",d,
               {"population":"11A","interpretation":"occupancy is discretization-sensitive, not evidence of classes"})
    refs=[ref]
    for hn,vf,name in [(1,100,"02_fine_grid"),(5,500,"03_coarse_grid")]:
        xedges=np.arange(0,50+hn,hn);yedges=np.arange(0,6000+vf,vf)
        z=matrix(a.loc[a.ATC_INTERVENTION.eq(1)],"horizontal_separation_nm","vertical_separation_ft",xedges=xedges,yedges=yedges)
        fig=mapfig(z,f"Interventions: {hn} NM × {vf} ft bins","Horizontal separation (NM)","Absolute vertical separation (ft)",
                   "Interventions per bin",xedges,yedges,cmap="magma",refs=True)
        refs.append(finish(fig,"08_sensitivity",name,"Map-bin resolution sensitivity",bins_table(z,xedges,yedges,"windows"),
                           {"population":"11A intervention arm","horizontal_bin_nm":hn,"vertical_bin_ft":vf,"in_plot":int(z.sum())}))
    q=[]
    for floor in (0,1/60,5/60,1/6):
        m=a.tcpa_min.ge(floor)&a.tcpa_min.le(30)&a.cpa_nm.between(0,50)
        q.append({"tcpa_floor_min":floor,"windows":int(m.sum()),"interventions":int((m&a.ATC_INTERVENTION.eq(1)).sum()),
                  "controls":int((m&a.ATC_INTERVENTION.eq(0)).sum())})
    save(pd.DataFrame(q),"tables/11_sensitivity_tcpa_floor.csv")
    older=pd.read_parquet(BASE/"05_features/window_features_cruise_matching_v2.parquet",
                          columns=["window_id","horizontal_sep_at_event","vertical_sep_at_event"])
    comparison=a[["window_id","ATC_INTERVENTION","horizontal_separation_nm","vertical_separation_ft"]].merge(
        older,on="window_id",validate="one_to_one")
    comparison["h_asof_minus_stage05_nm"]=comparison.horizontal_separation_nm-comparison.horizontal_sep_at_event
    comparison["v_asof_minus_stage05_ft"]=comparison.vertical_separation_ft-comparison.vertical_sep_at_event
    save(comparison,"tables/11a_anchor_method_sensitivity.csv")
    x=comparison.h_asof_minus_stage05_nm.dropna().clip(-50,50)
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    ax.hist(x,bins=np.arange(-50,52,2),color="#496c89")
    ax.axvline(0,color="black",ls="--",lw=.8)
    ax.set(xlabel="Backward-as-of minus Stage05 summary horizontal separation (NM)",ylabel="Matched windows / bin",
           title="Anchor reconstruction changes measured neighbour geometry")
    refs.append(finish(fig,"08_sensitivity","04_anchor_method","Anchor-method sensitivity",comparison,
                       {"population":"11A same windows","stage11_anchor":"Stage10B backward-as-of native snapshots",
                        "comparison":"existing Stage05 horizontal_sep_at_event; no upstream recomputation",
                        "histogram_clip_nm":"-50 to 50 visually, full differences in CSV"}))
    denom_path=OUT/"tables/11b_denominator_sensitivity.csv"
    if denom_path.exists():
        den=pd.read_csv(denom_path)
        selected=den.loc[den.setting.isin(["30s_grid_age15s_primary","30s_grid_both_states_age10s_exclusion",
                                         "60s_grid_age15s_rebuilt"])].copy()
        fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout="constrained")
        labels=["30 s / age 15 s","30 s / both age 10 s","60 s / age 15 s"]
        axes[0].bar(labels,selected.aircraft_hours,color="#386b8a")
        axes[0].set(ylabel="Observed focal aircraft-hours",title="Coverage sensitivity")
        axes[1].bar(labels,selected.rate_per_1000h,color="#b65a36")
        axes[1].set(ylabel="Events per 1,000 observed aircraft-hours",title="Rate sensitivity")
        for ax in axes:ax.tick_params(axis="x",rotation=28,labelsize=8)
        refs.append(finish(fig,"08_sensitivity","05_denominator_rules","Denominator rule sensitivity",den,
                           {"population":"11B","60s":"fully rebuilt from pilot Stage02 states and Stage03 events",
                            "age10":"exclusion from primary risk set; no neighbour reselection"}))
        horizon=den.loc[den.setting.str.contains("future_tcpa")].copy()
        fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
        ax.plot([5,10,30],horizon.rate_per_1000h,marker="o",color="#73569b")
        ax.set(xlabel="Maximum future TCPA (min)",ylabel="Events / 1,000 eligible aircraft-hours",
               title="Prediction-horizon sensitivity")
        ax.grid(alpha=.2)
        refs.append(finish(fig,"08_sensitivity","06_horizon","CPA–TCPA horizon sensitivity",horizon,
                           {"population":"11B","minimum_tcpa_s":1,"max_cpa_nm":50}))
    return refs


def dashboard(paths):
    panels=[("01  Intervention map",paths["p01"][0]),("02  Traffic exposure",paths["p02"][0]),
            ("03  Population rate",paths["p03"][0]),("04  Separation distributions",paths["p04"][0]),
            ("05  Fixed-pair evolution",paths["p05"][0]),("06  CPA–TCPA",paths["p06"][0])]
    refs=[]
    for page in (0,1):
        fig,axes=plt.subplots(3,1,figsize=(10.5,20),layout="constrained")
        for ax,(title,path) in zip(axes,panels[page*3:page*3+3]):
            ax.imshow(plt.imread(OUT/path));ax.axis("off");ax.set_title(title,fontsize=13)
        fig.suptitle("SCAT — ATC Intervention Geometry and Encounter Dynamics",fontsize=16)
        dest=FIG/f"STAGE11_DASHBOARD_PAGE_{page+1}.png"
        fig.savefig(dest,dpi=300,bbox_inches="tight")
        fig.savefig(dest.with_suffix(".svg"),bbox_inches="tight")
        plt.close(fig);refs.append(str(dest.relative_to(OUT)))
        source_data=pd.DataFrame({"panel":[p[0] for p in panels[page*3:page*3+3]],
                                  "source_figure":[p[1] for p in panels[page*3:page*3+3]]})
        source_data.to_csv(dest.with_suffix(".csv"),index=False)
        dest.with_suffix(".json").write_text(json.dumps({"title":"SCAT — ATC Intervention Geometry and Encounter Dynamics",
            "week":WEEK,"page":page+1,"population":"11A and 11B labeled by constituent panel",
            "layout":"three stacked panels for legibility","sources":source_data.to_dict("records")},indent=2)+"\n")
    return refs


def report(paths,a,risk,rates,trace,diag,dash):
    audit=json.loads((OUT/"diagnostics/11b_rate_audit.json").read_text())
    armi=a.loc[a.ATC_INTERVENTION.eq(1)];armc=a.loc[a.ATC_INTERVENTION.eq(0)]
    n_e=int(risk.event_count.sum());hours=float(risk.exposure_hours.sum())
    episode_path=OUT/"tables/11c_observed_pair_segments.csv"
    episodes=pd.read_csv(episode_path) if episode_path.exists() else pd.DataFrame()
    gap_table=pd.read_csv(OUT/"tables/11c_gap_sensitivity.csv") if episode_path.exists() else pd.DataFrame()
    gap_text=", ".join(f"{int(row.max_gap_s)} s: {int(row.segments):,}" for row in gap_table.itertuples())
    denominator_sensitivity=pd.read_csv(OUT/"tables/11b_denominator_sensitivity.csv") if (OUT/"tables/11b_denominator_sensitivity.csv").exists() else pd.DataFrame()
    sixty=denominator_sensitivity.loc[denominator_sensitivity.setting.eq("60s_grid_age15s_rebuilt")]
    sixty_text=(f" La reconstrucción a 60 s aporta {sixty.aircraft_hours.iloc[0]:,.1f} h y {int(sixty.events.iloc[0]):,} eventos elegibles, "
                f"frente a {hours:,.1f} h y {n_e:,} a 30 s; la cobertura cambia con la rejilla." if len(sixty) else "")
    hmed=(armi.horizontal_separation_nm.median(),armc.horizontal_separation_nm.median())
    vmed=(armi.vertical_separation_ft.median(),armc.vertical_separation_ft.median())
    good=rates["h2_v250"].supported.sum()
    hcurve=rates["horizontal_2"].set_index("x_low")
    near=hcurve.loc[6.0];far=hcurve.loc[24.0]
    def pic(p):return f"![{Path(p).stem}](../{p})"
    sections=[]
    sections.append(f"""# STAGE 11 — ATC Intervention Geometry and Encounter Dynamics

## 1. Resumen ejecutivo

Semana piloto {WEEK}, FL290–FL410. La cohorte emparejada 11A contiene {len(armi):,} intervenciones y {len(armc):,} controles; {int(a.other_flight_key.notna().sum()):,} ventanas tienen pareja geométrica al ancla. En 11B, {diag['focal_snapshots']:,} snapshots focales observables aportan {hours:,.1f} aeronaves-hora; se asignan {n_e:,} autorizaciones al intervalo de riesgo observable. La tasa global observada es {1000*n_e/hours:.1f} eventos por 1.000 aeronaves-hora. Es una **intensidad de autorización**, no una probabilidad de conflicto.

Los resultados no validan cuatro clases de severidad: los colores son salidas de un detector, las autorizaciones son acciones ATC y no disponemos de confirmación externa de conflicto. Las variables continuas y su evolución temporal son por ahora la representación científicamente más defendible.

## 2. Motivación científica

La proximidad actual, la aproximación prevista, una autorización registrada y una alerta del detector son conceptos distintos. Se examinan asociaciones geométricas sin usarlas como etiquetas verdaderas de riesgo.

## 3. Datos, población y calidad

Fuentes: estados nativos de Stage 02 (70.216.709 en 13 semanas; {diag['pilot_native_rows_scanned']:,} leídos de la semana piloto); autorizaciones accionables de Stage 03 (1.019.799 globales); cohorte 4.098×2 de Stage 04; trazas cinemáticas de Stage 05; snapshots as-of y colores de Stage 10B. No se repitieron Stage 01–10B ni se usaron otras semanas.

11A fija la pareja horizontal más cercana en el snapshot nativo reconstruido por Stage 10B hacia atrás desde el ancla, con edad máxima 15 s; H, V, CPA y TCPA se calculan con los mismos dos estados previos al evento. Las trazas Stage 05 se filtran después por esa identidad de pareja. 11B forma ticks de 30 s: último estado de cada aeronave a t≤tick y con edad≤15 s. Un focal FL290–FL410 aporta 30 s de exposición por tick; el vecino horizontal más próximo puede tener cualquier altitud. Una autorización en (tick,tick+30] del mismo focal se cuenta en ese tick. Los eventos sin snapshot elegible se excluyen y se cuantifican. Los estados de ambos aviones preceden al evento en los mapas 11A y 11B. Se registran fuente, plot, instante y edad. La tasa es condicional a cobertura observable; no extrapolamos automáticamente a tramos sin estados.

Eventos piloto accionables: {diag['pilot_actionable_events']:,}; con snapshot focal elegible: {diag['events_with_eligible_cruise_snapshot']:,}; con vecino observable: {diag['events_with_eligible_neighbour']:,}. Snapshots focales sin vecino: {diag['focal_snapshots']-diag['focal_snapshots_with_neighbour']:,}. Eventos repetidos en un mismo vuelo/tick elegible: {diag['event_duplicates_same_flight_tick_eligible']:,}; cada autorización se cuenta como evento, no como ensayo binario.

Los intervalos de las figuras de tasa son la envolvente del intervalo Garwood para conteos y un intervalo de influencia agrupado por vuelo focal. Recogen repetición dentro de vuelo, pero no toda la dependencia entre vuelos ni la variación de cobertura; se interpretan como incertidumbre descriptiva. En mapas de tasa se ocultan celdas con <10 h, <3 eventos o <20 vuelos. La unidad de análisis de 11A es ventana emparejada; la de 11B es focal-tick. No se mezclan sus denominadores.
""")
    for num,title,key,meaning,limit in [
        (4,"Panel 01 — Mapa espacial de intervenciones","p01",f"Mediana H 11A: {hmed[0]:.2f} NM en intervención y {hmed[1]:.2f} NM en control; mediana V: {vmed[0]:.0f} y {vmed[1]:.0f} ft.","La diferencia de densidad usa brazos equilibrados; no es una tasa poblacional. Fuera de 0–50 NM / 0–6.000 ft se contabiliza en metadatos."),
        (5,"Panel 02 — Exposición real del tráfico","p02",f"Exposición total observada: {hours:,.1f} aeronaves-hora; con vecino: {diag['neighbour_hours']:,.1f}. En el rango simultáneo 0–50 NM / 0–6.000 ft aparecen {rates['h2_v250'].exposure_hours.sum():,.1f} aeronaves-hora; el resto sigue en la tabla fuente.","La cobertura depende de estados nativos y edad máxima; el mapa no representa pares-hora ni asigna una autorización a un par como causa."),
        (6,"Panel 03 — Tasas normalizadas","p03",f"Tasa global condicional observable: {1000*n_e/hours:.1f} por 1.000 aeronaves-hora. En H 6–8 NM: {near.rate_per_1000h:.0f} ({int(near.events)} eventos/{near.exposure_hours:.1f} h); en 24–26 NM: {far.rate_per_1000h:.0f} ({int(far.events)} eventos/{far.exposure_hours:.1f} h). En el mapa H–V simultáneo hay {int(rates['h2_v250'].events.sum()):,} eventos y {int(good)} de {len(rates['h2_v250'])} celdas cumplen el umbral de soporte.",f"La tasa describe frecuencia de autorizaciones, no probabilidad de conflicto. HEADING tiene solo {int(risk.event_heading.sum())} eventos elegibles, por lo que sus curvas son imprecisas. Las celdas escasas, la cobertura y la dependencia entre vuelos requieren cautela."),
        (7,"Panel 04 — Distribuciones de separación","p04","Histogramas, ECDF y desglose por LEVEL, HEADING y SPEED cuantifican forma y solapamiento.","Los picos por niveles de vuelo o bins no prueban clases naturales; el ancho de bin y la población emparejada afectan la forma."),
        (8,"Panel 05 — Evolución temporal","p05",f"Se siguen {trace.window_id.nunique():,} ventanas con pareja preevento fija; los n por instante constan en cada CSV.","Después de t=0 la evolución es observacional. La cobertura variable por instante impide atribución causal."),
        (9,"Panel 06 — CPA–TCPA","p06",f"11B: {audit['cpa_tcpa_quality_snapshots']:,} snapshots y {audit['cpa_tcpa_quality_events']:,} eventos cumplen TCPA 1 s–30 min, CPA≤50 NM y separación vertical predicha finita. Hay {audit['subsecond_positive_tcpa_snapshots']:,} TCPA positivos subsegundo y {audit['negative_tcpa_snapshots']:,} negativos en el denominador bruto con vecino.","TCPA negativo, subsegundo o faltante se excluye del mapa futuro y se audita. El mapa divergente usa TCPA negativo y distancia actual, nunca lo llama mínima futura. CPA/TCPA de una sola pareja. La predicción lineal no es trayectoria real.")]:
        sections.append(f"## {num}. {title}\n\n**Interpretación:** {meaning}\n\n**Limitación:** {limit}\n\n"+"\n\n".join(pic(p) for p in paths[key])+"\n")
    sections.append("## 10. Cuatro colores del detector\n\nLos colores GREEN, YELLOW, ORANGE y RED son salidas operativas del detector Stage 10B. En la cohorte emparejada se solapan las distribuciones físicas; las celdas no GREEN son pequeñas. GREEN no certifica seguridad y RED no confirma conflicto.\n\n"+"\n\n".join(pic(p) for p in paths["color"])+"\n")
    case_paths=sorted((FIG/"09_case_studies").glob("*.png"))
    sections.append("## 11. Estudios de casos\n\nLos casos individuales usan trayectorias nativas de Stage 02 y la misma pareja de Stage 05 a lo largo del tiempo. La estrella marca el último estado anterior al ancla; en controles t=0 es un pseudoancla. Cada caso es un encuentro observado, no un conflicto confirmado. Datos, IDs y cobertura están en `examples/` y los CSV de cada figura.\n\n"+
                    "\n\n".join(pic(str(p.relative_to(OUT))) for p in case_paths)+"\n")
    sections.append(f"## 12. Sensibilidad y robustez\n\nSe variaron bins H 1/2/5 NM, V 100/250/500 ft, suelo TCPA 0/1/5/10 s, horizonte 5/10/30 min, radio descriptivo 30/50 NM, calidad de edad 10/15 s y ventana ±150/±300 s. Los CSV muestran conteos, exposición y ocupación. Se reconstruyó la población a 60 s desde Stage 02/03 sin repetir fases upstream.{sixty_text} Se comparó además el ancla as-of segura con el resumen Stage 05 en exactamente las mismas ventanas; la diferencia evidencia sensibilidad al método de reconstrucción y selección del vecino. En 11C, segmentos por hueco máximo: {gap_text}; el límite de continuidad es decisivo. Quedan pendientes análisis de otras reglas de vecino y réplicas de otra semana; por ello no debe interpretarse estabilidad de regiones como probada. La forma en V de la mediana temporal también refleja la elección de pareja cercana en t=0 y no debe leerse como efecto de la autorización.\n\n"+"\n\n".join(pic(p) for p in paths["sensitivity"])+"\n")
    sections.append("## 13. Discusión científica\n\nUna intervención registrada muestra una decisión ATC; la proximidad geométrica describe un estado físico; un encuentro físicamente relevante requiere contexto operacional adicional; una alerta refleja reglas del detector; un conflicto confirmado requiere validación independiente. El patrón de 11A y la intensidad condicional de 11B pueden motivar nuevas hipótesis, pero no convierten ninguna de estas entidades en otra.\n")
    sections.append(f"## 14. Limitaciones\n\nPiloto de una semana y una banda; cobertura de ticks condicionada a registros nativos; cinemática lineal; vecino horizontal fijo en trazas; eventos recurrentes del mismo vuelo; intervalos sin dependencia entre vuelos ni incertidumbre de cobertura; falta confirmación operacional de conflictos; selección de la cohorte emparejada. 11C se conserva separado: {len(episodes):,} segmentos observacionales de pareja fija en ±300 s con hueco máximo de 60 s. Son tramos censurados por la ventana y los intervalos sin observación; la tabla `11c_gap_sensitivity.csv` muestra 30/60/90 s. No son episodios de conflicto ni un denominador poblacional.\n")
    sections.append("## 15. Conclusiones\n\n**No existe evidencia suficiente en SCAT para fijar categorías diferenciadas de relevancia de conflicto.** Las distancias, tasas de autorización y dinámicas se deben conservar como variables continuas con sus incertidumbres y proveniencia. Los cuatro colores pueden compararse como estados de un detector, sin tratarlos como verdad de seguridad.\n")
    sections.append("## 16. Siguiente etapa\n\nValidar estabilidad temporal en otra semana y auditar cobertura, formalizar inicio y fin de los segmentos 11C con censura explícita, obtener revisión operacional independiente de una muestra de casos y entonces evaluar si alguna partición mejora una representación continua.\n\n## Dashboard\n\n"+"\n\n".join(pic(p) for p in dash)+"\n")
    report=OUT/"reports/STAGE11_INTERVENTION_GEOMETRY_REPORT.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text("\n".join(sections),encoding="utf-8")
    return report


def manifest():
    rows=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name!="manifest.csv":
            rows.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"kind":p.suffix.lstrip(".")})
    save(pd.DataFrame(rows),"manifest.csv")


def main():
    a=pd.read_parquet(OUT/"tables/11a_anchor_pairs.parquet")
    trace=pd.read_parquet(OUT/"tables/11a_fixed_pair_timeline.parquet")
    risk=pd.read_parquet(OUT/"tables/11b_population_risk_set.parquet")
    diag=json.loads((OUT/"diagnostics/11b_population_audit.json").read_text())
    keys=["h2_v250","horizontal_2","vertical_250","level","heading","speed","CONVERGING","DIVERGING_OR_TANGENTIAL","cpa_tcpa","cpa_nm","tcpa_min"]
    names={"h2_v250":"11b_rate_map_h2_v250.csv","horizontal_2":"11b_horizontal_rate_2nm.csv","vertical_250":"11b_vertical_rate_250ft.csv",
           "level":"11b_horizontal_rate_level.csv","heading":"11b_horizontal_rate_heading.csv","speed":"11b_horizontal_rate_speed.csv",
           "CONVERGING":"11b_horizontal_rate_converging.csv","DIVERGING_OR_TANGENTIAL":"11b_horizontal_rate_diverging_or_tangential.csv",
           "cpa_tcpa":"11b_cpa_tcpa_rate.csv","cpa_nm":"11b_cpa_nm_rate.csv","tcpa_min":"11b_tcpa_min_rate.csv"}
    rates={k:pd.read_csv(OUT/"tables"/names[k]) for k in keys}
    paths={"p01":panel01(a),"p02":panel02(risk),"p03":panel03(rates,risk),"p04":panel04(a),
           "p05":panel05(trace),"p06":panel06(a,rates,risk),"color":colours(a,trace),
           "sensitivity":sensitivity(a,rates,risk)}
    dash=dashboard(paths)
    rep=report(paths,a,risk,rates,trace,diag,dash)
    manifest()
    print(rep)


if __name__=="__main__":main()
