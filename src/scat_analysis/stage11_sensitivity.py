"""Stage 11 denominator, quality, horizon and window sensitivity on pilot data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stage11_core import OUT, build_population, save


def summary(risk: pd.DataFrame, label: str) -> dict:
    hours=float(risk.exposure_hours.sum());events=int(risk.event_count.sum())
    return {"setting":label,"focal_snapshots":len(risk),"aircraft_hours":hours,"events":events,
            "rate_per_1000h":1000*events/hours if hours else np.nan,
            "neighbour_snapshots":int(risk.neighbour_key.notna().sum()),
            "neighbour_events":int(risk.loc[risk.neighbour_key.notna(),"event_count"].sum())}


def main():
    risk=pd.read_parquet(OUT/"tables/11b_population_risk_set.parquet")
    rows=[summary(risk,"30s_grid_age15s_primary")]
    rows.append(summary(risk.loc[risk.focal_age_s.le(10)&(risk.neighbour_age_s.le(10)|risk.neighbour_key.isna())],
                        "30s_grid_both_states_age10s_exclusion"))
    for radius in (30,50):
        rows.append(summary(risk.loc[risk.horizontal_nm.le(radius)],f"30s_nearest_within_{radius}nm"))
    for horizon in (5,10,30):
        rows.append(summary(risk.loc[risk.tcpa_min.between(1/60,horizon)&risk.cpa_nm.between(0,50)],
                            f"30s_future_tcpa_1s_to_{horizon}min_cpa_le50nm"))
    alt,diag=build_population(step=60,max_age=15)
    rows.append(summary(alt,"60s_grid_age15s_rebuilt"))
    table=pd.DataFrame(rows)
    save(table,"tables/11b_denominator_sensitivity.csv")
    trace=pd.read_parquet(OUT/"tables/11a_fixed_pair_timeline.parquet")
    windows=[]
    for width in (150,300):
        g=trace.loc[trace.relative_seconds.abs().le(width)]
        windows.append({"half_window_s":width,"windows":g.window_id.nunique(),"pair_rows":len(g),
                        "median_horizontal_nm":g.horizontal_separation_nm.median(),
                        "median_vertical_ft":g.vertical_separation_ft.median(),
                        "median_bins_per_window":g.groupby("window_id").size().median()})
    save(pd.DataFrame(windows),"tables/11a_time_window_sensitivity.csv")
    print(table.to_string(index=False))
    print("60s audit",json.dumps(diag))


if __name__=="__main__":main()
