"""One-page scientific dashboard of Stage 12 primary evidence."""
from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage12_audit_inputs import OUT,WEEKS

p=pd.read_parquet(OUT/'episodes'/f'{WEEKS[0]}_episode_summary.parquet')
r=pd.read_parquet(OUT/'episodes'/f'{WEEKS[1]}_episode_summary.parquet')
a=pd.read_csv(OUT/'tables'/f'{WEEKS[0]}_prediction_error_summary.csv')
b=pd.read_csv(OUT/'tables'/f'{WEEKS[1]}_prediction_error_summary.csv')
g=pd.read_csv(OUT/'tables'/'structure_gmm_metrics.csv')
fig=plt.figure(figsize=(17,10),facecolor='#f5f8fb');grid=fig.add_gridspec(3,4,height_ratios=[1,2,2],hspace=.43,wspace=.27,top=.86,bottom=.10)
fig.text(.04,.96,'SCAT STAGE 12  |  Pairwise encounter discovery',fontsize=24,color='#153d59',weight='bold')
fig.text(.04,.924,'Dos semanas · radio 100 NM · paso 30 s · episodios observacionales, no conflictos confirmados',fontsize=11,color='#4c6271')
for i,(title,value,sub) in enumerate([
 ('PILOTO',f'{len(p):,}'.replace(',','.'),'episodios'),
 ('RÉPLICA',f'{len(r):,}'.replace(',','.'),'episodios'),
 ('CENSURA PILOTO',f'{100*p.temporally_censored.mean():.1f}%'.replace('.',','),'temporal'),
 ('HORIZONTE 15 MIN',f"{a.query('horizon_min==15 and validation_status==\"VALIDATED\"').median_abs_horizontal_end_error_nm.iloc[0]:.1f} NM".replace('.',','),'error H mediano')]):
    ax=fig.add_subplot(grid[0,i]);ax.set_facecolor('white');ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#cbd5e1')
    ax.text(.06,.78,title,transform=ax.transAxes,color='#526b7b',fontsize=10,weight='bold')
    ax.text(.06,.38,value,transform=ax.transAxes,color='#155e75',fontsize=22,weight='bold')
    ax.text(.06,.14,sub,transform=ax.transAxes,color='#526b7b',fontsize=10)
ax=fig.add_subplot(grid[1,:2]);ax.hist(p.min_horizontal_nm,bins=60,range=(0,100),density=True,histtype='step',lw=2,color='#155e75',label='Piloto');ax.hist(r.min_horizontal_nm,bins=60,range=(0,100),density=True,histtype='step',lw=2,color='#a21caf',label='Réplica');ax.legend();ax.set_title('Geometría horizontal reproducible');ax.set_xlabel('Mínimo horizontal observado (NM)')
ax=fig.add_subplot(grid[1,2:]);ax.hist(p.vertical_at_min_horizontal_ft,bins=70,range=(0,43000),density=True,histtype='step',lw=2,color='#155e75',label='Piloto');ax.hist(r.vertical_at_min_horizontal_ft,bins=70,range=(0,43000),density=True,histtype='step',lw=2,color='#a21caf',label='Réplica');ax.legend();ax.set_title('Vertical simultánea en H mínima');ax.set_xlabel('Separación vertical (ft)')
ax=fig.add_subplot(grid[2,:2]);
for d,label,color in [(a,'Piloto','#155e75'),(b,'Réplica','#a21caf')]:
    q=d.loc[d.validation_status.eq('VALIDATED')];ax.plot(q.horizon_min,q.median_abs_horizontal_end_error_nm,'o-',label=label,color=color)
ax.legend();ax.set_title('Predicción cinemática: error horizontal crece con el horizonte');ax.set_xlabel('Horizonte (min)');ax.set_ylabel('Error absoluto mediano (NM)')
ax=fig.add_subplot(grid[2,2:]);ax.plot(g.k,g.bic,'o-',color='#a21caf');ax.set_title('GMM: BIC sigue mejorando hasta K=6');ax.set_xlabel('Número de componentes');ax.set_ylabel('BIC');ax.text(.03,.05,'Silhouette K=6: 0,060 · HDBSCAN sensible a parámetros',transform=ax.transAxes,fontsize=10,color='#526b7b')
fig.text(.04,.025,'Decisión: CONTINUOUS_INTERACTION_STRUCTURE. Censura alta y colores focales impiden inferir conflicto o severidad.',fontsize=12,color='#153d59',weight='bold')
path=OUT/'figures'/'STAGE12_DASHBOARD.png';fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
print(path)
