"""Scale/feature and temporal refit checks for descriptive mixtures."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score,silhouette_score
from stage12_audit_inputs import OUT,WEEKS

SEED=12012
FULL=['log_min_h','log_simultaneous_v','start_closing','crossing_angle','fraction_converging','relative_speed','track_change','altitude_change']
GEOMETRY=['log_min_h','log_simultaneous_v','start_closing','crossing_angle','relative_speed']

def main():
    a=pd.read_parquet(OUT/'tables'/f'{WEEKS[0]}_structure_sample.parquet')
    b=pd.read_parquet(OUT/'tables'/f'{WEEKS[1]}_structure_sample.parquet')
    base=RobustScaler(quantile_range=(10,90)).fit(a[FULL])
    za=np.clip(base.transform(a[FULL]),-8,8);zb=np.clip(base.transform(b[FULL]),-8,8)
    ref=GaussianMixture(n_components=6,covariance_type='diag',reg_covar=.001,random_state=SEED,n_init=2).fit(za)
    ref_b=ref.predict(zb)
    rows=[]
    for feature_set,cols in [('full',FULL),('geometry_only',GEOMETRY)]:
        for scaling in ('robust','standard'):
            sc=(RobustScaler(quantile_range=(10,90)) if scaling=='robust' else StandardScaler()).fit(a[cols])
            x=np.clip(sc.transform(a[cols]),-8,8);y=np.clip(sc.transform(b[cols]),-8,8)
            metrics=[]
            for k in (2,3,4,5,6):
                m=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=.001,random_state=SEED,n_init=2).fit(x)
                metrics.append((k,m.bic(x)))
            best=min(metrics,key=lambda v:v[1])[0]
            m=GaussianMixture(n_components=best,covariance_type='diag',reg_covar=.001,random_state=SEED,n_init=2).fit(x)
            r=GaussianMixture(n_components=best,covariance_type='diag',reg_covar=.001,random_state=SEED+9,n_init=2).fit(y)
            rows.append(dict(feature_set=feature_set,scaling=scaling,best_k_2_to_6=best,
                bic_at_2=metrics[0][1],bic_at_6=metrics[-1][1],
                replica_refit_ari=float(adjusted_rand_score(m.predict(y),r.predict(y))),
                agreement_with_full_robust_on_replica=float(adjusted_rand_score(ref_b,m.predict(y)))))
    # HDBSCAN parameter sensitivity on full robust matrix.
    idx=np.random.default_rng(SEED).choice(len(za),min(3500,len(za)),replace=False)
    density=[]
    for min_samples in (5,10,20):
        for size in (30,50,120):
            lab=HDBSCAN(min_cluster_size=size,min_samples=min_samples).fit_predict(za[idx]);mask=lab>=0
            density.append(dict(min_samples=min_samples,min_cluster_size=size,
                clusters=int(len(np.unique(lab[mask]))),coverage=float(mask.mean())))
    pd.DataFrame(rows).to_csv(OUT/'tables'/'structure_scaling_feature_sensitivity.csv',index=False)
    pd.DataFrame(density).to_csv(OUT/'tables'/'structure_density_sensitivity.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(density).to_string(index=False))

if __name__=='__main__':main()
