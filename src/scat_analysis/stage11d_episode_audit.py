"""Audit observation-contiguous Stage 11C pair segments inside ±300 s windows."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from stage11_core import OUT as STAGE11
from stage11_episodes import observed_segments
from stage11d_core import OUT


def audit() -> None:
    trace=pd.read_parquet(STAGE11/"tables/11a_fixed_pair_timeline.parquet")
    trace=trace.sort_values(["window_id","time_epoch","relative_bin_s"],kind="mergesort").reset_index(drop=True)
    trace["gap_s"]=trace.groupby("window_id").time_epoch.diff()
    trace["bin_gap_s"]=trace.groupby("window_id").relative_bin_s.diff()
    trace[["window_id","other_flight_key","time_epoch","relative_seconds","relative_bin_s",
           "horizontal_separation_nm","gap_s","bin_gap_s"]].to_parquet(
           OUT/"tables"/"11c_trace_gap_audit.parquet",index=False,compression="zstd")
    gaps=trace.gap_s.dropna()
    bins=np.array([0,1,5,15,29,30,30.001,31,45,59,60,60.001,61,90,120,180,300,600],float)
    counts,_=np.histogram(gaps,bins)
    pd.DataFrame({"gap_low_s":bins[:-1],"gap_high_s":bins[1:],"observed_gaps":counts}).to_csv(
        OUT/"tables"/"11c_gap_histogram.csv",index=False)
    parts=[]; summary=[]
    for threshold in (30,45,60,90):
        s=observed_segments(trace,threshold)
        parts.append(s)
        summary.append(dict(gap_rule_s=threshold,segments=len(s),windows=s.window_id.nunique(),
            median_duration_s=float(s.observed_duration_s.median()),
            median_samples=float(s.n_pair_bins.median()),single_sample=int(s.n_pair_bins.eq(1).sum()),
            left_censored=int(s.left_censored.sum()),right_censored=int(s.right_censored.sum()),
            any_edge_censored=int((s.left_censored|s.right_censored).sum()),
            segments_min_h_over_50=int(s.min_horizontal_nm.gt(50).sum())))
    segments=pd.concat(parts,ignore_index=True)
    segments.to_parquet(OUT/"tables"/"11c_segment_audit.parquet",index=False,compression="zstd")
    pd.DataFrame(summary).to_csv(OUT/"tables"/"11c_segment_gap_sensitivity.csv",index=False)
    s30=segments.loc[segments.gap_rule_s.eq(30)].groupby("window_id").size().rename("segments_30")
    s60=segments.loc[segments.gap_rule_s.eq(60)].groupby("window_id").size().rename("segments_60")
    comparison=pd.concat([s30,s60],axis=1).reset_index()
    comparison["fragments_merged"]=comparison.segments_30-comparison.segments_60
    comparison.to_csv(OUT/"tables"/"11c_fragmentation_by_window.csv",index=False)
    chosen=comparison.loc[comparison.fragments_merged.ge(3)].sort_values("fragments_merged",ascending=False).head(8)
    if len(chosen)<4:
        chosen=comparison.loc[comparison.fragments_merged.gt(0)].sort_values("fragments_merged",ascending=False).head(8)
    examples=trace.loc[trace.window_id.isin(chosen.window_id)].copy()
    examples=examples.merge(chosen,on="window_id",validate="many_to_one")
    examples.to_csv(OUT/"examples"/"11c_fragmentation_examples.csv",index=False)
    anchor=pd.read_parquet(STAGE11/"tables/11a_anchor_pairs.parquet",
        columns=["window_id","focal_flight_key","other_flight_key","event_epoch"])
    pair_change=anchor.groupby("focal_flight_key").agg(windows=("window_id","size"),
       distinct_selected_neighbors=("other_flight_key","nunique")).reset_index()
    pair_change.to_csv(OUT/"tables"/"11c_focal_pair_identity.csv",index=False)
    diagnostic=dict(trace_rows=len(trace),windows=int(trace.window_id.nunique()),
        duplicate_window_bin=int(trace.duplicated(["window_id","relative_bin_s"]).sum()),
        duplicate_window_pair_time=int(trace.duplicated(["window_id","other_flight_key","time_epoch"]).sum()),
        window_pair_switches=int(trace.groupby("window_id").other_flight_key.nunique().gt(1).sum()),
        focal_with_multiple_selected_neighbors_across_windows=int(pair_change.distinct_selected_neighbors.gt(1).sum()),
        gaps_gt30_le60=int(((gaps>30)&(gaps<=60)).sum()),
        gaps_gt60=int((gaps>60).sum()),
        gap_median_s=float(gaps.median()),gap_p95_s=float(gaps.quantile(.95)),
        traces_over_50nm=int(trace.horizontal_separation_nm.gt(50).sum()),
        traces_over_100nm=int(trace.horizontal_separation_nm.gt(100).sum()),
        fraction_trace_at_window_edges=float(trace.relative_bin_s.abs().eq(300).mean()),
        note="Trace was deduplicated to one row per 30 s bin in Stage 11; raw pre-deduplication multiplicity is not inferred. Fixed pairs and ±300 s windows are event conditioned, not complete encounters." )
    (OUT/"diagnostics"/"11c_episode_audit.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    print(json.dumps(diagnostic,indent=2),flush=True)


if __name__=="__main__": audit()
