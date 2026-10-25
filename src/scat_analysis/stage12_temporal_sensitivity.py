"""Six-hour pilot comparison of 15 s and 30 s as-of snapshot grids."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from stage12_audit_inputs import OUT,WEEKS
from stage12_build_encounters import load_snapshots,pair_geometry
from stage12_compute_features import segment_flags

START=int(pd.Timestamp('2016-10-17 12:00:00',tz='UTC').timestamp())
END=START+6*3600

def run_grid(step):
    states,audit=load_snapshots(WEEKS[0],step=step,max_age=15)
    states=states.loc[states.tick.between(START,END, inclusive='left')]
    rows=[]
    for tick,g in states.groupby('tick',sort=False):
        p=pair_geometry(g,int(tick),WEEKS[0],100)
        if len(p):rows.append(p[['pair_id','tick','horizontal_nm']])
    pairs=pd.concat(rows,ignore_index=True).sort_values(['pair_id','tick']).reset_index(drop=True)
    counts={}
    for gap in (45,60,90):counts[str(gap)]=int(segment_flags(pairs,gap).sum())
    cadence=states.sort_values(['entity_key','time_epoch']).groupby('entity_key').time_epoch.diff()
    return dict(grid_s=step,window_start_utc='2016-10-17 12:00',window_hours=6,
        snapshots=len(states),unique_entities=int(states.entity_key.nunique()),pair_rows=len(pairs),
        unique_pairs=int(pairs.pair_id.nunique()),episodes_by_gap_s=counts,
        median_source_age_s=float((states.tick-states.time_epoch).median()),
        median_observed_sample_interval_s=float(cadence.dropna().median()),
        p90_observed_sample_interval_s=float(cadence.dropna().quantile(.9)),
        note='Comparison includes differing numbers of pair observations; episode counts depend on grid and gap jointly.')

if __name__=='__main__':
    out=[run_grid(15),run_grid(30)]
    (OUT/'diagnostics'/'pilot_15s_vs_30s_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
