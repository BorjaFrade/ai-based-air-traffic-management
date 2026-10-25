"""Six-panel encounter atlas, structure/operation figures and case gallery."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from stage12_audit_inputs import OUT,WEEKS

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white'})
C=['#155e75','#b45309','#9f1239','#4d7c0f','#6d28d9','#0f766e','#334155','#be185d']

def save(fig,relative):
    path=OUT/'figures'/relative;path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig);return str(path.relative_to(OUT))

def select_cases(e):
    base=e.loc[e.quality_for_structure & e.fully_bracketed_at_100nm].copy()
    specs=[
      ('close_simultaneous',base.min_horizontal_nm.lt(5)&base.vertical_at_min_horizontal_ft.lt(1000)),
      ('convergent',base.fraction_converging.gt(.9)&base.observed_duration_s.gt(600)),
      ('divergent',base.fraction_converging.lt(.1)&base.observed_duration_s.gt(600)),
      ('crossing',base.median_crossing_angle_deg.between(70,110)&base.min_horizontal_nm.lt(20)),
      ('parallel',base.median_crossing_angle_deg.lt(15)&base.min_horizontal_nm.lt(20)),
      ('vertical_change',np.maximum(base.a_altitude_change_ft.abs(),base.b_altitude_change_ft.abs()).gt(5000)),
      ('turn',np.maximum(base.a_track_change_deg.abs(),base.b_track_change_deg.abs()).gt(70)),
      ('long_duration',base.observed_duration_s.gt(2700)),
      ('multi_aircraft',base.max_concurrent_neighbors_a.gt(4)&base.max_concurrent_neighbors_b.gt(4))]
    selected=[];used=set()
    for name,mask in specs:
        x=base.loc[mask & ~base.episode_id.isin(used)]
        if x.empty:
            broad=e.loc[e.quality_for_structure & ~e.episode_id.isin(used)]
            if name=='convergent':x=broad.loc[broad.fraction_converging.gt(.9)&broad.observed_duration_s.gt(600)]
            elif name=='parallel':x=broad.loc[broad.median_crossing_angle_deg.lt(15)&broad.min_horizontal_nm.lt(20)]
            elif name=='multi_aircraft':x=broad.loc[broad.max_concurrent_neighbors_a.gt(4)&broad.max_concurrent_neighbors_b.gt(4)]
        if x.empty:continue
        x=x.sort_values(['n_samples','episode_id'],ascending=[False,True]).iloc[0]
        selected.append((name,x));used.add(x.episode_id)
    return selected

def trace_for(ep):
    bucket=ep.episode_id.split(':')[1][:3].replace('B','')
    path=OUT/'episodes'/f'{WEEKS[0]}_episode_traces'/f'bucket_{int(bucket):02d}.parquet'
    return pd.read_parquet(path,filters=[('episode_id','==',ep.episode_id)]).sort_values('tick')

def tracks(ax,t,title):
    lat0=float(t.a_lat.iloc[0]);lon0=float(t.a_lon.iloc[0]);cl=np.cos(np.deg2rad(lat0))
    for side,color in (('a',C[0]),('b',C[2])):
        x=(t[f'{side}_lon'].to_numpy()-lon0)*60*cl;y=(t[f'{side}_lat'].to_numpy()-lat0)*60
        ax.plot(x,y,color=color,lw=1.7,label=side.upper());ax.scatter(x[0],y[0],s=28,color=color,marker='o');
        ax.scatter(x[-1],y[-1],s=35,color=color,marker='s')
        for j in range(8,len(x),max(8,len(x)//8)):
            ax.annotate('',xy=(x[j],y[j]),xytext=(x[j-1],y[j-1]),arrowprops={'arrowstyle':'->','color':color,'lw':1})
    k=int(t.horizontal_nm.argmin());ax.scatter((t.a_lon.iloc[k]-lon0)*60*cl,(t.a_lat.iloc[k]-lat0)*60,s=70,marker='x',color='black')
    ax.set_title(title);ax.set_xlabel('Este desde A inicial (NM)');ax.set_ylabel('Norte (NM)');ax.axis('equal')

def main():
    e={w:pd.read_parquet(OUT/'episodes'/f'{w}_episode_summary.parquet') for w in WEEKS}
    p=e[WEEKS[0]];r=e[WEEKS[1]]
    cases=select_cases(p);case_rows=[];traces={}
    for name,row in cases:
        t=trace_for(row);traces[name]=t
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        tracks(ax[0],t,f'{name}: {row.episode_id}')
        minutes=(t.tick-t.tick.iloc[0])/60
        ax[1].plot(minutes,t.horizontal_nm,color=C[0],label='Horizontal (NM)')
        av=ax[1].twinx();av.plot(minutes,t.vertical_ft,color=C[2],label='Vertical (ft)')
        ax[1].set_xlabel('Minutos desde inicio');ax[1].set_ylabel('Separación horizontal (NM)');av.set_ylabel('Separación vertical (ft)')
        fig.suptitle(f'{name} | min H {row.min_horizontal_nm:.1f} NM, V simultánea {row.vertical_at_min_horizontal_ft:.0f} ft')
        path=OUT/'examples'/f'{name}.png';fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
        case_rows.append(dict(case=name,episode_id=row.episode_id,figure=str(path.relative_to(OUT)),
                              min_horizontal_nm=row.min_horizontal_nm,vertical_at_min_horizontal_ft=row.vertical_at_min_horizontal_ft,
                              duration_s=row.observed_duration_s))
    pd.DataFrame(case_rows).to_csv(OUT/'examples'/'case_gallery.csv',index=False)
    # Panel 01: trajectories
    fig,axes=plt.subplots(3,3,figsize=(17,12),constrained_layout=True)
    for ax,(name,row) in zip(axes.flat,cases):tracks(ax,traces[name],name)
    for ax in axes.flat[len(cases):]:ax.axis('off')
    fig.suptitle('01 | Trayectorias de parejas; círculo inicio, cuadrado fin, X mínimo H',fontsize=15)
    save(fig,'01_trajectories/trajectory_atlas.png')
    # Panel 02: same pair identity across all temporal metrics
    names=list(traces)[:4];fig,axes=plt.subplots(len(names),5,figsize=(19,3.3*len(names)),constrained_layout=True)
    if len(names)==1:axes=axes[None,:]
    for i,name in enumerate(names):
        t=traces[name];x=(t.tick-t.tick.iloc[0])/60
        series=[('H (NM)',t.horizontal_nm),('V (ft)',t.vertical_ft),
                ('Cierre (NM/min)',t.closing_nm_min),('Ángulo (°)',t.crossing_angle_deg),
                ('TCPA bruto (min)',t.tcpa_raw_min)]
        for j,(title,y) in enumerate(series):axes[i,j].plot(x,y,color=C[j],lw=1.4);axes[i,j].set_title(f'{name}: {title}');axes[i,j].set_xlabel('Tiempo (min)')
    fig.suptitle('02 | Evolución temporal de parejas fijas',fontsize=15)
    save(fig,'02_temporal_profiles/temporal_profiles.png')
    # Panel 03: simultaneous geometry and marginal distributions
    s=p.sample(min(80000,len(p)),random_state=12)
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    ax[0,0].hist(p.min_horizontal_nm,bins=70,color=C[0]);ax[0,0].set_title('Mínimo horizontal (NM)')
    ax[0,1].hist(p.vertical_at_min_horizontal_ft,bins=80,color=C[1]);ax[0,1].set_title('Vertical simultánea en min H (ft)')
    ax[0,2].hist(p.observed_duration_s/60,bins=80,color=C[3]);ax[0,2].set_xlim(0,60);ax[0,2].set_title('Duración observada (min)')
    h=ax[1,0].hexbin(s.min_horizontal_nm,s.vertical_at_min_horizontal_ft,gridsize=55,mincnt=1,norm=LogNorm(),cmap='viridis');fig.colorbar(h,ax=ax[1,0]);ax[1,0].set_title('H mínima vs V en ese instante')
    ax[1,1].hist(p.start_closing_nm_min,bins=80,color=C[2]);ax[1,1].set_title('Cierre inicial (NM/min)')
    ax[1,2].hist(p.median_crossing_angle_deg,bins=72,color=C[4]);ax[1,2].set_title('Ángulo de cruce (°)')
    fig.suptitle('03 | Geometrías observadas en el piloto',fontsize=15)
    save(fig,'03_geometry_distributions/geometry_distributions.png')
    # Panel 04 prediction metrics
    pred=pd.read_csv(OUT/'tables'/f'{WEEKS[0]}_prediction_error_summary.csv')
    valid=pred.loc[pred.validation_status.eq('VALIDATED')]
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax[0,0].plot(valid.horizon_min,valid.median_abs_horizontal_end_error_nm,'o-',label='H (NM)')
    ax[0,0].plot(valid.horizon_min,valid.median_abs_cpa_error_nm,'s-',label='CPA (NM)');ax[0,0].legend();ax[0,0].set_title('Error absoluto mediano')
    ax[0,1].plot(valid.horizon_min,valid.median_abs_vertical_end_error_ft,'o-',color=C[2]);ax[0,1].set_title('Error vertical absoluto mediano (ft)')
    status=pred.pivot(index='horizon_min',columns='validation_status',values='n');status.div(status.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax[1,0],color=[C[1],C[2],C[0]],legend=True);ax[1,0].set_title('Cobertura y validez de CPA')
    b=OUT/'tables'/f'{WEEKS[0]}_prediction_validation';one=pd.read_parquet(b/'bucket_00.parquet',columns=['horizon_min','validation_status','predicted_end_h_nm','observed_horizontal_nm'])
    one=one.loc[(one.horizon_min==5)&one.validation_status.eq('VALIDATED')].sample(frac=.2,random_state=12)
    ax[1,1].hexbin(one.predicted_end_h_nm,one.observed_horizontal_nm,gridsize=50,mincnt=1,norm=LogNorm());ax[1,1].plot([0,110],[0,110],'k--');ax[1,1].set_xlim(0,110);ax[1,1].set_ylim(0,110);ax[1,1].set_title('H prevista vs observada a 5 min; bucket 0')
    for a in ax.flat:a.set_xlabel('Horizonte (min)' if a!=ax[1,1] else 'H prevista (NM)')
    ax[1,1].set_ylabel('H observada (NM)')
    fig.suptitle('04 | CPA/TCPA y validación retrospectiva',fontsize=15)
    save(fig,'04_cpa_tcpa/cpa_prediction_validation.png')
    # Panel 05 persistence
    sens=pd.read_csv(OUT/'tables'/f'{WEEKS[0]}_radius_gap_sensitivity.csv');sens=sens.loc[sens.radius_nm.eq(100)]
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax[0,0].hist(p.observed_duration_s/60,bins=75,color=C[0]);ax[0,0].set_xlim(0,60);ax[0,0].set_title('Duración (min)')
    ax[0,1].hist(p.fraction_converging,bins=40,color=C[2]);ax[0,1].set_title('Fracción de muestras convergentes')
    labels=['Ambos cruces','Censura temporal'];values=[int(p.fully_bracketed_at_100nm.sum()),int(p.temporally_censored.sum())]
    ax[1,0].bar(labels,values,color=[C[3],C[1]]);ax[1,0].set_title('Completitud dentro de 100 NM')
    ax[1,1].plot(sens.gap_s,sens.episodes,'o-',color=C[4]);ax[1,1].set_title('Episodios con hueco admisible');ax[1,1].set_xlabel('Hueco máximo (s)')
    fig.suptitle('05 | Persistencia, censura y sensibilidad',fontsize=15)
    save(fig,'05_duration_persistence/duration_persistence.png')
    # Panel 06 two weeks
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for w,d,col in zip(WEEKS,(p,r),(C[0],C[2])):
        label='Piloto' if w==WEEKS[0] else 'Réplica'
        for a,series,title,rng in [(ax[0,0],d.min_horizontal_nm,'Min H (NM)',(0,100)),
            (ax[0,1],d.vertical_at_min_horizontal_ft,'V simultánea (ft)',(0,43000)),
            (ax[0,2],d.observed_duration_s/60,'Duración (min)',(0,60)),
            (ax[1,0],d.start_closing_nm_min,'Cierre inicial',(-18,18)),
            (ax[1,1],d.median_crossing_angle_deg,'Ángulo de cruce',(0,180))]:
            a.hist(series,bins=60,range=rng,density=True,histtype='step',linewidth=1.6,color=col,label=label);a.set_title(title)
    ax[0,0].legend();ax[1,2].bar(['Piloto','Réplica'],[p.fully_bracketed_at_100nm.mean(),r.fully_bracketed_at_100nm.mean()],color=[C[0],C[2]]);ax[1,2].set_title('Fracción con ambos cruces de 100 NM')
    fig.suptitle('06 | Comparación con criterios idénticos',fontsize=15)
    save(fig,'06_week_comparison/week_comparison.png')
    # Stage 12D
    st=pd.read_parquet(OUT/'tables'/f'{WEEKS[0]}_structure_sample.parquet');gm=pd.read_csv(OUT/'tables'/'structure_gmm_metrics.csv')
    fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    ax[0].scatter(st.pc1,st.pc2,c=st.gmm_cluster,s=2,alpha=.35,cmap='tab10');ax[0].set_title('PCA de muestra acotada; color GMM');ax[0].set_xlabel('PC1');ax[0].set_ylabel('PC2')
    ax[1].plot(gm.k,gm.bic,'o-',color=C[4]);ax[1].set_title('GMM BIC, K=1..6');ax[1].set_xlabel('Componentes');ax[1].set_ylabel('BIC')
    save(fig,'07_structure_discovery/structure_discovery.png')
    # Stage 12E (only when enrichment complete)
    op=OUT/'tables'/f'{WEEKS[0]}_operational_enrichment.parquet'
    if op.exists():
        o=pd.read_parquet(op,columns=['episode_id','atc_context','ambiguous_pair_context_count']);j=p[['episode_id','min_horizontal_nm','vertical_at_min_horizontal_ft']].merge(o,on='episode_id')
        fig,ax=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for flag,col,label in [(False,C[0],'Sin contexto ATC'),(True,C[2],'Con contexto ATC')]:
            x=j.loc[j.atc_context.eq(flag),'min_horizontal_nm'];ax[0].hist(x,bins=50,range=(0,100),density=True,histtype='step',color=col,label=label,lw=1.8)
        ax[0].legend();ax[0].set_title('Separación mínima según contexto temporal')
        ax[1].bar(['Asociaciones ambiguas','Sin ambigüedad'],[(j.ambiguous_pair_context_count>0).sum(),((j.atc_context)&(j.ambiguous_pair_context_count==0)).sum()],color=[C[1],C[3]])
        ax[1].set_title('Atribución de autorización a pareja')
        fig.suptitle('08 | Enriquecimiento posterior; asociaciones no causales',fontsize=14)
        save(fig,'08_operational_comparison/operational_comparison.png')
        colour=pd.read_parquet(OUT/'tables'/'pilot_detector_context.parquet')
        ce=p.merge(o,on='episode_id').merge(colour,on='episode_id',how='left')
        operational_cases=[]
        tests=[('green_close',ce.focal_dynamic_state.eq('GREEN')&ce.min_horizontal_nm.lt(5)&ce.vertical_at_min_horizontal_ft.lt(1000)),
               ('red_without_focal_atc',ce.focal_dynamic_state.eq('RED')&ce.ATC_INTERVENTION.eq(0)&~ce.atc_context),
               ('convergent_no_atc_context',ce.fraction_converging.gt(.8)&ce.min_horizontal_nm.lt(20)&~ce.atc_context)]
        for name,mask in tests:
            candidates=ce.loc[mask & ce.quality_for_structure].sort_values(['min_horizontal_nm','episode_id'])
            if candidates.empty:continue
            row=candidates.iloc[0];t=trace_for(row)
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            tracks(axes[0],t,name)
            x=(t.tick-t.tick.iloc[0])/60
            axes[1].plot(x,t.horizontal_nm,color=C[0]);other=axes[1].twinx();other.plot(x,t.vertical_ft,color=C[2])
            axes[1].set_xlabel('Minutos');axes[1].set_ylabel('H (NM)');other.set_ylabel('V (ft)')
            fig.suptitle(f'{name} | {row.episode_id} | H min {row.min_horizontal_nm:.1f} NM, V simultánea {row.vertical_at_min_horizontal_ft:.0f} ft')
            dest=OUT/'examples'/f'{name}.png';fig.savefig(dest,dpi=170,bbox_inches='tight');plt.close(fig)
            operational_cases.append(dict(case=name,episode_id=row.episode_id,figure=str(dest.relative_to(OUT)),
                min_horizontal_nm=row.min_horizontal_nm,vertical_at_min_horizontal_ft=row.vertical_at_min_horizontal_ft,
                focal_dynamic_state=row.focal_dynamic_state,ATC_INTERVENTION=row.ATC_INTERVENTION,
                atc_context=row.atc_context))
        pd.DataFrame(operational_cases).to_csv(OUT/'examples'/'operational_case_gallery.csv',index=False)
    print(json.dumps({'cases':case_rows,'main_panels':6},indent=2,default=str))

if __name__=='__main__':main()
