"""11C: observation-contiguous fixed-pair segments, separate from rate analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stage11_core import OUT, save


def observed_segments(trace: pd.DataFrame, max_gap_s: float = 60) -> pd.DataFrame:
    """Split on observed timestamp gaps; ±300 s window edges remain censored."""
    if max_gap_s<=0:
        raise ValueError("max_gap_s must be positive")
    if trace.empty:
        return pd.DataFrame()
    rows=[]
    for (window,neighbour),g in trace.groupby(["window_id","other_flight_key"],sort=False):
        g=g.sort_values("time_epoch").copy()
        if not g.time_epoch.is_monotonic_increasing:
            raise ValueError("Nonmonotonic pair times")
        g["segment_number"]=(g.time_epoch.diff().fillna(0)>max_gap_s).cumsum()
        for number,s in g.groupby("segment_number",sort=False):
            rows.append({"window_id":window,"focal_flight_key":s.focal_flight_key.iloc[0],
                         "other_flight_key":neighbour,"segment_number":int(number),
                         "ATC_INTERVENTION":int(s.ATC_INTERVENTION.iloc[0]),
                         "clearance_type":s.clearance_type.iloc[0],
                         "start_relative_s":float(s.relative_seconds.min()),
                         "end_relative_s":float(s.relative_seconds.max()),
                         "observed_duration_s":float(s.time_epoch.max()-s.time_epoch.min()),
                         "n_pair_bins":len(s),"min_horizontal_nm":float(s.horizontal_separation_nm.min()),
                         "min_vertical_ft":float(s.vertical_separation_ft.min()),
                         "contains_anchor_bin":bool(s.relative_bin_s.eq(0).any()),
                         "left_censored":bool(s.relative_seconds.min()<=-285),
                         "right_censored":bool(s.relative_seconds.max()>=285),
                         "gap_rule_s":max_gap_s,
                         "meaning":"observation-contiguous fixed-pair segment; not conflict"})
    result=pd.DataFrame(rows)
    if result.duplicated(["window_id","other_flight_key","segment_number"]).any():
        raise ValueError("Duplicate segment identity")
    return result


def main():
    trace=pd.read_parquet(OUT/"tables/11a_fixed_pair_timeline.parquet")
    parts=[]
    for gap in (30,60,90):
        d=observed_segments(trace,gap)
        if gap==60:
            save(d,"tables/11c_observed_pair_segments.csv")
        parts.append({"max_gap_s":gap,"segments":len(d),"windows":d.window_id.nunique(),
                      "median_duration_s":d.observed_duration_s.median(),
                      "left_censored":int(d.left_censored.sum()),
                      "right_censored":int(d.right_censored.sum())})
    save(pd.DataFrame(parts),"tables/11c_gap_sensitivity.csv")
    print(pd.DataFrame(parts).to_string(index=False))


if __name__=="__main__":main()
