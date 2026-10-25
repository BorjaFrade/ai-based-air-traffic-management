"""Bounded, reproducible exploration of physical episode structure."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from stage12_audit_inputs import OUT,WEEKS

SEED=12012
FEATURES=['log_min_h','log_simultaneous_v','start_closing','crossing_angle',
          'fraction_converging','relative_speed','track_change','altitude_change']

def physical_matrix(d):
    x=pd.DataFrame(index=d.index)
    x['log_min_h']=np.log1p(d.min_horizontal_nm.clip(lower=0))
    x['log_simultaneous_v']=np.log1p(d.vertical_at_min_horizontal_ft.clip(lower=0))
    x['start_closing']=d.start_closing_nm_min.clip(-20,20)
    x['crossing_angle']=d.median_crossing_angle_deg.clip(0,180)
    x['fraction_converging']=d.fraction_converging.clip(0,1)
    x['relative_speed']=d.start_relative_vx_nm_min.combine(d.start_relative_vy_nm_min,lambda a,b:np.hypot(a,b)).clip(0,20)
    x['track_change']=np.maximum(d.a_track_change_deg.abs(),d.b_track_change_deg.abs()).clip(0,180)
    x['altitude_change']=np.maximum(d.a_altitude_change_ft.abs(),d.b_altitude_change_ft.abs()).clip(0,20000)
    return x

def sample_week(week,n=10000):
    d=pd.read_parquet(OUT/'episodes'/f'{week}_episode_summary.parquet')
    d=d.loc[d.quality_for_structure & d.fully_bracketed_at_100nm].copy()
    # Limit repeated episodes of a pair before sampling; retain all geometric ranges.
    d=d.sort_values('episode_id').groupby('pair_id',sort=False).head(2)
    rng=np.random.default_rng(SEED)
    if len(d)>n:d=d.iloc[np.sort(rng.choice(len(d),n,replace=False))]
    return d.reset_index(drop=True)

def main():
    d1=sample_week(WEEKS[0]);d2=sample_week(WEEKS[1])
    x1=physical_matrix(d1);x2=physical_matrix(d2)
    good1=np.isfinite(x1).all(axis=1);good2=np.isfinite(x2).all(axis=1)
    d1=d1.loc[good1].reset_index(drop=True);x1=x1.loc[good1].reset_index(drop=True)
    d2=d2.loc[good2].reset_index(drop=True);x2=x2.loc[good2].reset_index(drop=True)
    sc=RobustScaler(quantile_range=(10,90)).fit(x1)
    z1=np.clip(sc.transform(x1),-8,8);z2=np.clip(sc.transform(x2),-8,8)
    pca=PCA(n_components=min(8,z1.shape[1]),random_state=SEED).fit(z1)
    results=[];models={}
    for k in range(1,7):
        m=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=.001,
                          random_state=SEED,n_init=2,max_iter=150).fit(z1)
        labels=m.predict(z1);weights=np.bincount(labels,minlength=k)/len(labels)
        eval_idx=np.random.default_rng(SEED).choice(len(z1),min(1800,len(z1)),replace=False)
        sil=float(silhouette_score(z1[eval_idx],labels[eval_idx])) if k>1 and len(np.unique(labels[eval_idx]))>1 else np.nan
        entropy=float((-m.predict_proba(z1)*np.log(np.maximum(m.predict_proba(z1),1e-12))).sum(axis=1).mean())
        results.append(dict(method='GMM_DIAG',k=k,bic=m.bic(z1),silhouette=sil,
                            mean_assignment_entropy=entropy,min_cluster_share=weights.min(),
                            replica_mean_log_likelihood=m.score(z2)))
        models[k]=m
    # Stability is measured on fixed held-out points; component numbers can permute.
    best=min(models,key=lambda k:models[k].bic(z1))
    fixed=np.random.default_rng(SEED+1).choice(len(z1),min(1500,len(z1)),replace=False)
    ref=models[best].predict(z1[fixed]);ari=[]
    for seed in (11,22,33):
        train=np.random.default_rng(seed).choice(len(z1),min(6500,len(z1)),replace=True)
        fit=GaussianMixture(n_components=best,covariance_type='diag',reg_covar=.001,
                            random_state=seed,n_init=2,max_iter=150).fit(z1[train])
        ari.append(adjusted_rand_score(ref,fit.predict(z1[fixed])))
    hdb=[]
    subset=np.random.default_rng(SEED).choice(len(z1),min(3500,len(z1)),replace=False)
    for size in (50,120,250):
        lab=HDBSCAN(min_cluster_size=size,min_samples=20).fit_predict(z1[subset])
        mask=lab>=0; ncl=len(np.unique(lab[mask]));cov=float(mask.mean())
        sil=float(silhouette_score(z1[subset][mask],lab[mask])) if ncl>1 and mask.sum()>ncl else np.nan
        hdb.append(dict(min_cluster_size=size,clusters=ncl,coverage=cov,
                        silhouette=sil if np.isfinite(sil) else None))
    hier=[]
    subset=np.random.default_rng(SEED+2).choice(len(z1),min(1200,len(z1)),replace=False)
    for k in (2,3,4,6):
        lab=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(z1[subset])
        hier.append(dict(k=k,silhouette=float(silhouette_score(z1[subset],lab))))
    labels1=models[best].predict(z1);labels2=models[best].predict(z2)
    replication=pd.DataFrame({'cluster':np.arange(best),
        'pilot_share':np.bincount(labels1,minlength=best)/len(labels1),
        'replica_share':np.bincount(labels2,minlength=best)/len(labels2)})
    replication['abs_share_change']=abs(replication.pilot_share-replication.replica_share)
    replication.to_csv(OUT/'tables'/'structure_replication.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'tables'/'structure_gmm_metrics.csv',index=False)
    pd.DataFrame(hdb).to_csv(OUT/'tables'/'structure_hdbscan_metrics.csv',index=False)
    pd.DataFrame(hier).to_csv(OUT/'tables'/'structure_hierarchical_metrics.csv',index=False)
    # Distributions remain available without imposing labels.
    for d,x,z,lab,week in ((d1,x1,z1,labels1,WEEKS[0]),(d2,x2,z2,labels2,WEEKS[1])):
        coords=pca.transform(z)
        o=d[['episode_id','pair_id','min_horizontal_nm','vertical_at_min_horizontal_ft',
             'observed_duration_s','start_a_alt_ft','start_b_alt_ft']].copy()
        for f in FEATURES:o[f]=x[f]
        o['pc1']=coords[:,0];o['pc2']=coords[:,1];o['gmm_cluster']=lab
        o.to_parquet(OUT/'tables'/f'{week}_structure_sample.parquet',index=False)
    medians=pd.concat([pd.DataFrame(z1,columns=FEATURES).assign(week='pilot',cluster=labels1),
                       pd.DataFrame(z2,columns=FEATURES).assign(week='replica',cluster=labels2)]).groupby(['week','cluster']).median().reset_index()
    medians.to_csv(OUT/'tables'/'structure_cluster_medians_scaled.csv',index=False)
    details=dict(seed=SEED,cohort='quality>=4 samples, duration>=90s, age<=15s and both 100NM boundaries observed; max 2 episodes per pair; random cap 10000/week',
        pilot_n=len(z1),replica_n=len(z2),features=FEATURES,pca_variance_ratio=pca.explained_variance_ratio_.tolist(),
        gmm_bic_best_k=best,gmm_resample_ari=ari,hdbscan=hdb,hierarchical=hier,
        max_replication_share_shift=float(replication.abs_share_change.max()),
        physical_freeze_source_sha256={w:hashlib.sha256((OUT/'episodes'/f'{w}_episode_summary.parquet').read_bytes()).hexdigest() for w in WEEKS})
    (OUT/'diagnostics'/'structure_discovery.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
    print(json.dumps(details,indent=2))

if __name__=='__main__':main()
