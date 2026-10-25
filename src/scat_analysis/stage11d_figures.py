"""Reproduce the Stage 11D figures exclusively from saved scientific tables."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

from stage11d_core import OUT, PILOT, REPLICATION, CRITERIA

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,
    "axes.labelsize":9,"figure.facecolor":"white","axes.spines.top":False,"axes.spines.right":False,
    "savefig.bbox":"tight","svg.hashsalt":"stage11d"})
PAL={"NEAREST_HORIZONTAL":"#1d4e68","MINIMUM_FUTURE_CPA":"#bd5a3c",
     "MINIMUM_3D_DISTANCE":"#5f8c7d","MAXIMUM_CLOSING":"#9b74a5","NEAREST_VERTICAL":"#d09a3a",
     "BOTH_LEVEL":"#1d4e68","MOVING":"#bd5a3c","UNKNOWN":"#888888"}
SHORT={"NEAREST_HORIZONTAL":"Nearest H","MINIMUM_FUTURE_CPA":"Future CPA",
       "MINIMUM_3D_DISTANCE":"3D distance","MAXIMUM_CLOSING":"Max closing","NEAREST_VERTICAL":"Nearest V"}


def table(week,name): return pd.read_csv(OUT/"tables"/f"{week}_{name}.csv")
def save(fig,number,folder,slug,svg=False):
    path=OUT/"figures"/folder/f"fig{number:02d}_{slug}"
    fig.savefig(path.with_suffix(".png"),dpi=300)
    if svg:fig.savefig(path.with_suffix(".svg"),metadata={"Date":"2026-10-25"})
    plt.close(fig)


def line(ax,d,x,label,color,ci=False):
    d=d.loc[d.supported.astype(bool)].sort_values(x)
    if d.empty:return
    ax.plot(d[x],d.rate_per_1000h,label=label,color=color,lw=1.7,marker="o",ms=2.5)
    if ci:ax.fill_between(d[x],d.ci_low_per_1000h,d.ci_high_per_1000h,color=color,alpha=.12,lw=0)
    ax.set_ylabel("Autorizaciones / 1.000 aeronaves-h")
    ax.grid(alpha=.18)


def map_plot(ax,data,title,vmax=None):
    f=data.loc[data.supported.astype(bool)]
    z=f.pivot(index="v_bin_ft",columns="h_bin_nm",values="rate_per_1000h")
    z=z.reindex(index=np.arange(0,6000,1000),columns=np.arange(0,50,5))
    im=ax.imshow(z.to_numpy(),origin="lower",aspect="auto",extent=[0,50,0,6000],
        cmap="viridis",vmin=0,vmax=vmax,interpolation="nearest")
    ax.set(title=title,xlabel="Separación H (NM)",ylabel="Separación V (ft)")
    return im


def rebin_map(data):
    d=data.copy();d["h_bin_nm"]=(d.h_bin_nm//5)*5;d["v_bin_ft"]=(d.v_bin_ft//1000)*1000
    d=d.groupby(["h_bin_nm","v_bin_ft"],as_index=False).agg(events=("events","sum"),
         exposure_hours=("exposure_hours","sum"),flights=("flights","max"))
    d["rate_per_1000h"]=1000*d.events/d.exposure_hours
    d["supported"]=(d.exposure_hours>=10)&(d.events>=3)&(d.flights>=20)
    return d


def run():
    h1=table(PILOT,"horizontal_rates");v1=table(PILOT,"vertical_rates")
    m1=table(PILOT,"rate_maps");motion=table(PILOT,"motion_vertical_rates")
    mm=table(PILOT,"motion_rate_maps");typ=table(PILOT,"type_motion_vertical_rates")
    phase=table(PILOT,"periodicity_decomposition")
    # 1: Vertical intensity by pre-event motion state.
    fig,ax=plt.subplots(figsize=(8,4.3))
    for k in ("BOTH_LEVEL","MOVING","UNKNOWN"):
        line(ax,motion.loc[motion.motion_group.eq(k)],"v_bin_ft",k.replace("_"," ").title(),PAL[k],True)
    ax.set(xlabel="Separación vertical (ft)",title="1. Estado vertical antes de la autorización")
    ax.legend(ncol=3,fontsize=8);save(fig,1,"01_vertical_periodicity","vertical_motion")
    # 2: type-specific event distributions with denominator plotted in separate panel.
    fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for k,c in (("LEVEL","#1d4e68"),("HEADING","#bd5a3c"),("SPEED","#5f8c7d")):
        d=typ.loc[typ.clearance_type.eq(k)].groupby("v_bin_ft").events.sum()
        axs[0].plot(d.index,d.values,label=k,color=c)
    d=v1.loc[v1.criterion.eq("NEAREST_HORIZONTAL")]
    axs[1].bar(d.v_bin_ft,d.exposure_hours,width=230,color="#7b9bad")
    axs[0].set(ylabel="Autorizaciones",title="2. Distribución por tipo y exposición")
    axs[1].set(xlabel="Separación vertical (ft)",ylabel="Aeronaves-h")
    axs[0].legend();save(fig,2,"01_vertical_periodicity","types_exposure")
    # 3-4: matched 2D scale for level and moving.
    motion_maps={k:rebin_map(mm.loc[mm.motion_group.eq(k)]) for k in ("BOTH_LEVEL","MOVING")}
    pd.concat([x.assign(motion_group=k) for k,x in motion_maps.items()]).to_csv(
        OUT/"tables"/"pilot_display_motion_maps_h5_v1000.csv",index=False)
    sub=pd.concat(motion_maps.values());sub=sub.loc[sub.supported.astype(bool)]
    vmax=np.nanquantile(sub.rate_per_1000h,.95) if len(sub) else None
    for number,k in ((3,"BOTH_LEVEL"),(4,"MOVING")):
        fig,ax=plt.subplots(figsize=(8,4.3));im=map_plot(ax,motion_maps[k],
            f"{number}. {k.replace('_',' ').title()}: tasa por separación",vmax)
        fig.colorbar(im,ax=ax,label="Autorizaciones / 1.000 aeronaves-h")
        save(fig,number,"01_vertical_periodicity",k.lower()+"_map")
    # 5: numerator and denominator decomposition.
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for i,k in enumerate(("BOTH_LEVEL","MOVING")):
        d=phase.loc[phase.motion_group.eq(k)].set_index("phase_group")
        d=d.reindex(["NEAR_1000_LEVEL","NEAR_500_OFFSET","OTHER"])
        x=np.arange(3);axs[i].bar(x-.18,d.events/d.events.sum(),width=.36,label="Event share",color="#bd5a3c")
        axs[i].bar(x+.18,d.exposure_hours/d.exposure_hours.sum(),width=.36,label="Exposure share",color="#1d4e68")
        axs[i].set(xticks=x,xticklabels=["±125 ft of\n1,000", "±125 ft of\n500 offset","Other"],
            ylabel="Share within motion group",title=k.replace("_"," ").title())
        axs[i].legend(fontsize=7)
    fig.suptitle("5. Periodicity: event and exposure components")
    save(fig,5,"01_vertical_periodicity","phase_decomposition")
    # 6: concordance.
    c=pd.read_csv(OUT/"tables"/"neighbor_concordance.csv")
    matrix=c.pivot(index="criterion_a",columns="criterion_b",values="concordance").reindex(index=CRITERIA,columns=CRITERIA)
    fig,ax=plt.subplots(figsize=(7.5,5.8));im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=1)
    ax.set(xticks=np.arange(5),yticks=np.arange(5),xticklabels=[SHORT[x] for x in CRITERIA],
        yticklabels=[SHORT[x] for x in CRITERIA],title="6. Identidad del vecino seleccionado")
    plt.setp(ax.get_xticklabels(),rotation=35,ha="right")
    for i in range(5):
        for j in range(5): ax.text(j,i,f"{matrix.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="Coincidencia entre seleccionados");save(fig,6,"02_neighbour_selection","concordance")
    # 7-8: criterion curves.
    for number,data,x,title in ((7,h1,"h_bin_nm","horizontal"),(8,v1,"v_bin_ft","vertical")):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for k in CRITERIA:line(ax,data.loc[data.criterion.eq(k)],x,SHORT[k],PAL[k])
        ax.set(xlabel="Separación horizontal (NM)" if number==7 else "Separación vertical (ft)",
               title=f"{number}. Tasas por criterio de vecino: {title}")
        ax.legend(ncol=3,fontsize=7);save(fig,number,"02_neighbour_selection",f"{title}_criteria")
    # 9: two independent rate-map numerators and denominators.
    subset=pd.concat([rebin_map(m1.loc[m1.criterion.eq(k)]) for k in
        ("NEAREST_HORIZONTAL","MINIMUM_FUTURE_CPA")])
    subset=subset.loc[subset.supported.astype(bool)]
    vmax=np.nanquantile(subset.rate_per_1000h,.95) if len(subset) else None
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    display_maps=[]
    for ax,k in zip(axs,("NEAREST_HORIZONTAL","MINIMUM_FUTURE_CPA")):
        d=rebin_map(m1.loc[m1.criterion.eq(k)])
        display_maps.append(d.assign(criterion=k))
        im=map_plot(ax,d,SHORT[k],vmax)
    pd.concat(display_maps).to_csv(OUT/"tables"/"pilot_display_criterion_maps_h5_v1000.csv",index=False)
    fig.suptitle("9. Mapas H–V con vecino y exposición recalculados")
    fig.colorbar(im,ax=axs,label="Autorizaciones / 1.000 aeronaves-h",shrink=.8)
    save(fig,9,"02_neighbour_selection","rate_map_comparison")
    # 10: changes in geometry among selectable pairs.
    d=pd.read_parquet(OUT/"tables"/"pilot_nearest_vs_future_cpa.parquet",
        columns=["delta_horizontal_nm","delta_cpa_nm","same_neighbor","event_count"])
    changed=d.loc[~d.same_neighbor]
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,col,title in zip(axs,("delta_horizontal_nm","delta_cpa_nm"),("Δ separación H","Δ CPA")):
        q=np.nanquantile(changed[col],[.01,.99]);ax.hist(changed[col].clip(*q),bins=60,color="#5f8c7d")
        ax.set(xlabel=title+" (NM)",ylabel="Instantes elegibles")
    fig.suptitle("10. Sustitución de vecino horizontal por mínimo CPA futuro")
    save(fig,10,"02_neighbour_selection","geometry_change")
    # 11-13: recurrence and its alternate counting units.
    flights=table(PILOT,"events_per_flight")
    fig,ax=plt.subplots(figsize=(7,4));counts=flights.events.clip(upper=20).value_counts().sort_index()
    ax.bar(counts.index,counts.values,color="#1d4e68");ax.set(xlabel="Autorizaciones por vuelo (20 = ≥20)",ylabel="Vuelos",
        title="11. Recurrencia por vuelo");save(fig,11,"03_clearance_recurrence","counts_per_flight")
    gaps=table(PILOT,"event_gaps").previous_gap_s.dropna()
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(gaps.clip(upper=3600),bins=np.r_[np.arange(0,630,30),900,1200,1800,3600],color="#bd5a3c")
    ax.set(xlabel="Intervalo entre autorizaciones (s; 3.600 = ≥3.600)",ylabel="Pares sucesivos",
        title="12. Huecos entre autorizaciones del mismo vuelo");save(fig,12,"03_clearance_recurrence","interclearance_gaps")
    curves=table(PILOT,"event_episode_binary_curves")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,dim in zip(axs,("h_bin_nm","v_bin_ft")):
        for metric,label,color in (("event_count","Eventos","#1d4e68"),("episode_count","Episodios 120 s","#bd5a3c"),
                                   ("binary_event_tick","Ticks binarios 30 s","#5f8c7d")):
            line(ax,curves.loc[curves.dimension.eq(dim)&curves.metric.eq(metric)],dim,label,color)
        ax.set_xlabel("Separación H (NM)" if dim.startswith("h") else "Separación V (ft)")
    axs[0].legend(fontsize=8);fig.suptitle("13. Eventos, episodios y ticks con ≥1 autorización")
    save(fig,13,"03_clearance_recurrence","event_episode_curves")
    # 14-18 replication.
    h2=table(REPLICATION,"horizontal_rates");v2=table(REPLICATION,"vertical_rates");m2=table(REPLICATION,"rate_maps")
    fig,ax=plt.subplots(figsize=(8,4.4))
    for week,d,c in ((PILOT,h1,"#1d4e68"),(REPLICATION,h2,"#bd5a3c")):
        line(ax,d.loc[d.criterion.eq("NEAREST_HORIZONTAL")],"h_bin_nm",week,c,True)
    ax.set(xlabel="Separación H (NM)",title="14. Replicación de tasa horizontal")
    ax.legend();save(fig,14,"04_week_reproducibility","horizontal_weeks")
    fig,ax=plt.subplots(figsize=(8,4.4))
    for week,d,c in ((PILOT,v1,"#1d4e68"),(REPLICATION,v2,"#bd5a3c")):
        line(ax,d.loc[d.criterion.eq("NEAREST_HORIZONTAL")],"v_bin_ft",week,c,True)
    ax.set(xlabel="Separación V (ft)",title="15. Replicación de tasa vertical")
    ax.legend();save(fig,15,"04_week_reproducibility","vertical_weeks")
    subset=pd.concat([rebin_map(d.loc[d.criterion.eq("NEAREST_HORIZONTAL")]) for d in (m1,m2)])
    subset=subset.loc[subset.supported.astype(bool)]
    vmax=np.nanquantile(subset.rate_per_1000h,.95) if len(subset) else None
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    display_weeks=[]
    for ax,week,data in zip(axs,(PILOT,REPLICATION),(m1,m2)):
        d=rebin_map(data.loc[data.criterion.eq("NEAREST_HORIZONTAL")])
        display_weeks.append(d.assign(week=week))
        im=map_plot(ax,d,week,vmax)
    pd.concat(display_weeks).to_csv(OUT/"tables"/"display_week_maps_h5_v1000.csv",index=False)
    fig.suptitle("16. Mapas H–V comparables entre semanas")
    fig.colorbar(im,ax=axs,label="Autorizaciones / 1.000 aeronaves-h",shrink=.8)
    save(fig,16,"04_week_reproducibility","rate_maps_weeks")
    t2=table(REPLICATION,"type_motion_vertical_rates")
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,week,data in zip(axs,(PILOT,REPLICATION),(typ,t2)):
        for k,c in (("LEVEL","#1d4e68"),("HEADING","#bd5a3c"),("SPEED","#5f8c7d")):
            line(ax,data.loc[data.clearance_type.eq(k)&data.motion_group.eq("BOTH_LEVEL")],"v_bin_ft",k,c)
        ax.set(title=week,xlabel="Separación V (ft)")
    axs[0].legend();fig.suptitle("17. Autorizaciones por tipo en parejas niveladas")
    save(fig,17,"04_week_reproducibility","type_level_weeks")
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,week,data in zip(axs,(PILOT,REPLICATION),(h1,h2)):
        for k in CRITERIA:line(ax,data.loc[data.criterion.eq(k)],"h_bin_nm",SHORT[k],PAL[k])
        ax.set(title=week,xlabel="Separación H (NM)")
    axs[0].legend(ncol=2,fontsize=7);fig.suptitle("18. Estabilidad por criterio de vecino")
    save(fig,18,"04_week_reproducibility","criteria_weeks")
    # Episode audit figures.
    gap=pd.read_csv(OUT/"tables"/"11c_gap_histogram.csv")
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(gap.gap_low_s,gap.observed_gaps,width=gap.gap_high_s-gap.gap_low_s,align="edge",color="#1d4e68")
    ax.set(xlim=(0,120),xlabel="Hueco observado entre bins (s)",ylabel="Huecos",
        title="19. Huecos reales en trazas 11C")
    save(fig,19,"05_episode_audit","observed_gaps")
    seg=pd.read_parquet(OUT/"tables"/"11c_segment_audit.parquet",columns=["gap_rule_s","observed_duration_s","n_pair_bins"])
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for k,c in ((30,"#bd5a3c"),(60,"#1d4e68")):
        q=seg.loc[seg.gap_rule_s.eq(k)]
        axs[0].hist(q.observed_duration_s,bins=np.arange(0,630,30),histtype="step",lw=1.7,label=f"Gap {k}s",color=c)
        axs[1].hist(q.n_pair_bins.clip(upper=21),bins=np.arange(.5,22.5,1),histtype="step",lw=1.7,label=f"Gap {k}s",color=c)
    axs[0].set(xlabel="Duración observada (s)",ylabel="Segmentos")
    axs[1].set(xlabel="Muestras por segmento (21 = ≥21)",ylabel="Segmentos")
    axs[0].legend();fig.suptitle("20. Fragmentación bajo reglas de 30 y 60 s")
    save(fig,20,"05_episode_audit","segment_duration")
    ex=pd.read_csv(OUT/"examples"/"11c_fragmentation_examples.csv")
    fig,axs=plt.subplots(4,2,figsize=(11,9),sharex=True)
    for ax,(_,g) in zip(axs.flat,ex.groupby("window_id")):
        g=g.sort_values("relative_seconds")
        ax.plot(g.relative_seconds,g.horizontal_separation_nm,"o-",ms=2,lw=.8,color="#1d4e68")
        for _,row in g.loc[g.gap_s.gt(30)&g.gap_s.le(60)].iterrows():ax.axvline(row.relative_seconds,color="#bd5a3c",lw=.9)
        ax.set(title=f"…{str(g.window_id.iloc[0])[-18:]} | Δseg={g.fragments_merged.iloc[0]}",ylabel="H (NM)")
    for ax in axs[-1]:ax.set_xlabel("Tiempo relativo a autorización (s)")
    fig.suptitle("21. Mismo par: huecos >30 y ≤60 s (líneas rojas)")
    save(fig,21,"05_episode_audit","fragmentation_examples")
    dashboard(h1,h2,v1,motion,c,flights,curves,seg)
    print("Created figures 1–21 and dashboard",flush=True)


def dashboard(h1,h2,v1,motion,concordance,flights,curves,segments):
    fig,axs=plt.subplots(3,2,figsize=(15,13));ax=axs.flat
    for k in ("BOTH_LEVEL","MOVING"):
        line(ax[0],motion.loc[motion.motion_group.eq(k)],"v_bin_ft",k.replace("_"," ").title(),PAL[k],True)
    ax[0].set(title="1 · Periodicidad y estado vertical",xlabel="V (ft)");ax[0].legend(fontsize=8)
    for k in ("NEAREST_HORIZONTAL","MINIMUM_FUTURE_CPA","MAXIMUM_CLOSING"):
        line(ax[1],h1.loc[h1.criterion.eq(k)],"h_bin_nm",SHORT[k],PAL[k])
    ax[1].set(title="2 · Selección del vecino",xlabel="H (NM)");ax[1].legend(fontsize=8)
    freq=flights.events.clip(upper=10).value_counts().sort_index()
    ax[2].bar(freq.index,freq.values,color="#1d4e68")
    ax[2].set(title=f"3 · Recurrencia | {len(flights):,} vuelos",xlabel="Autorizaciones/vuelo (10 = ≥10)",ylabel="Vuelos")
    for week,d,col in ((PILOT,h1,"#1d4e68"),(REPLICATION,h2,"#bd5a3c")):
        line(ax[3],d.loc[d.criterion.eq("NEAREST_HORIZONTAL")],"h_bin_nm",week,col,True)
    ax[3].set(title="4 · Dos semanas SCAT",xlabel="H (NM)");ax[3].legend(fontsize=8)
    sensitivity=segments.groupby("gap_rule_s").size()
    ax[4].bar(sensitivity.index.astype(str),sensitivity.values,color="#5f8c7d")
    ax[4].set(title="5 · Segmentación observada 11C",xlabel="Hueco máximo (s)",ylabel="Segmentos")
    ax[5].axis("off")
    small=h1.loc[h1.criterion.eq("NEAREST_HORIZONTAL")&h1.h_bin_nm.between(6,8)]
    text=("Población 11B: 476.044 ticks focales\n"
          "Exposición piloto: 3.967,0 aeronaves-h\n"
          "Eventos elegibles: 10.082\n"
          "Criterios: 5, misma población focal\n"
          "Intervalos: envolvente Garwood + clúster por vuelo\n"
          "Tasas = intensidad ATC, no riesgo de conflicto")
    ax[5].text(.04,.92,text,va="top",fontsize=13,linespacing=1.7)
    fig.suptitle("SCAT · Stage 11D | Validación de patrones ATC y geometría",fontsize=19,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.97])
    base=OUT/"figures"/"stage11d_five_panel_dashboard"
    fig.savefig(base.with_suffix(".png"),dpi=300)
    fig.savefig(base.with_suffix(".svg"),metadata={"Date":"2026-10-25"})
    plt.close(fig)


if __name__=="__main__":run()
