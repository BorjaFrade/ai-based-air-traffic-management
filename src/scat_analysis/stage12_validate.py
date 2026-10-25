"""Run population-level consistency checks without rebuilding upstream stages."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from stage12_audit_inputs import OUT,WEEKS


def main():
    checks=[]
    configs=[]
    for week in WEEKS:
        audit=json.loads((OUT/'diagnostics'/f'{week}_pair_build_audit.json').read_text())
        configs.append(audit['parameters'])
        seg=json.loads((OUT/'diagnostics'/f'{week}_segmentation_audit.json').read_text())
        ep=pd.read_parquet(OUT/'episodes'/f'{week}_episode_summary.parquet')
        snap=OUT/'tables'/f'{week}_snapshots_30s.parquet'
        cand=OUT/'tables'/f'{week}_candidate_pairs_150nm.parquet'
        pred=pd.read_csv(OUT/'tables'/f'{week}_prediction_error_summary.csv')
        conditions={
          'snapshot_metadata_matches_audit':pq.ParquetFile(snap).metadata.num_rows==audit['snapshots'],
          'candidate_metadata_matches_audit':pq.ParquetFile(cand).metadata.num_rows==audit['candidate_rows_within_nm']['150'],
          'episode_count_matches_audit':len(ep)==seg['episodes'],
          'episode_trace_rows_match_audit':int(ep.n_samples.sum())==seg['primary_rows'],
          'no_duplicate_pair_ticks':seg['duplicate_pair_ticks']==0,
          'no_physical_jump_splits':seg['physical_jump_splits']==0,
          'episode_id_unique':ep.episode_id.is_unique,
          'pair_identity_canonical':bool((ep.a_key<ep.b_key).all()),
          'start_not_after_end':bool((ep.start_tick<=ep.end_tick).all()),
          'min_h_le_start_end':bool((ep.min_horizontal_nm<=ep[['start_horizontal_nm','end_horizontal_nm']].min(axis=1)+1e-8).all()),
          'vertical_at_min_h_at_least_min_vertical':bool((ep.vertical_at_min_horizontal_ft+1e-8>=ep.min_vertical_ft).all()),
          'quality_age_not_future':bool((ep.max_state_age_s>=0).all() and (ep.max_state_age_s<=15+1e-6).all()),
          'boundary_partition':bool((ep.fully_bracketed_at_100nm|ep.temporally_censored).all()),
          'forecast_status_partition':bool((pred.groupby('horizon_min').n.sum()==len(ep)).all()),
          'structure_columns_exclude_operations':not {'ATC_INTERVENTION','focal_dynamic_state','clearance_type'}.intersection(
              pq.ParquetFile(OUT/'tables'/f'{week}_structure_sample.parquet').schema.names),
        }
        sample=pd.read_parquet(OUT/'episodes'/f'{week}_episode_traces/bucket_00.parquet',columns=[
            'tick','a_source_epoch','b_source_epoch','a_key','b_key','pair_id','episode_id'])
        conditions['asof_state_only']=bool((sample.a_source_epoch<=sample.tick).all() and (sample.b_source_epoch<=sample.tick).all())
        conditions['trace_pair_canonical']=bool((sample.a_key<sample.b_key).all())
        conditions['sample_no_pair_tick_duplicates']=not sample.duplicated(['pair_id','tick']).any()
        if not all(conditions.values()):raise AssertionError({week:[k for k,v in conditions.items() if not v]})
        checks.append(dict(week=week,checks=conditions,episodes=len(ep),primary_trace_rows=seg['primary_rows']))
    if configs[0]!=configs[1]:raise AssertionError('Pilot/replica reconstruction configurations differ')
    checks.append(dict(cross_week_check='identical_reconstruction_configuration',pass_=True))
    path=OUT/'diagnostics'/'population_validation.json';path.write_text(json.dumps(checks,indent=2)+'\n')
    print('PASS',sum(len(x.get('checks',{})) for x in checks)+1,'checks',path)

if __name__=='__main__':main()
