#!/usr/bin/env python3
"""Reproduce the selection funnel and observable coverage categories."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/benchmarks"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((ROOT / "results/stage20/stage20_summary.json").read_text())
    e = s["event_cohort"]
    a = e["selection_audit"]
    check((e["stage03_rows"], e["stage03_heading_or_level_unique_events"],
           e["eligible_unique_events"]) == (1019799, 908165, 51950),
          "Frozen order funnel totals differ")
    rows = []
    for kind in ("level", "heading"):
        for step, n in a[kind].items():
            rows.append({"kind": kind, "step": step, "count": n})
    pd.DataFrame(rows).to_csv(OUT / "eligibility_funnel.csv", index=False)
    check(a["level"]["eligible_clearance_events"] + a["heading"]["eligible_clearance_events"] == 51950,
          "Eligible type counts do not reconcile")
    ids = set(pd.read_csv(ROOT / "results/stage20/stage20_clearance_attribution.csv.gz",
                          usecols=["clearance_id"]).clearance_id.astype(str))
    raw = pq.read_table(ROOT / "data/processed/scat/03_clearances/scat_clearances.parquet",
                        columns=["clearance_id", "archive_week", "clearance_type",
                                 "matched_flight_level"]).to_pandas()
    broad = raw.clearance_type.astype(str).str.contains("HEADING|LEVEL")
    check(int(broad.sum()) == 908165, "Broader heading/level count differs")
    raw = raw.loc[broad].copy()
    raw["cohort"] = raw.clearance_id.astype(str).isin(ids).map({True: "eligible", False: "broader_only"})
    raw["order_kind"] = raw.clearance_type.astype(str).map(
        lambda x: "level" if "LEVEL" in x else "heading")
    raw["flight_level_band"] = pd.cut(pd.to_numeric(raw.matched_flight_level, errors="coerce"),
                                      [-float("inf"), 290, 411, float("inf")],
                                      labels=["below_290", "290_to_410", "above_410"],
                                      right=False).astype(str).fillna("missing")
    comparison = (raw.groupby(["cohort", "archive_week", "order_kind", "flight_level_band"])
                  .size().rename("orders").reset_index())
    comparison.to_csv(OUT / "eligibility_comparison.csv", index=False)
    check(int(comparison.loc[comparison.cohort.eq("eligible"), "orders"].sum()) == 51950,
          "Eligible rows in broader comparison differ")
    check(int(comparison.orders.sum()) == 908165, "Broad comparison loses rows")
    stage21 = pd.read_csv(ROOT / "results/stage21/stage21_encounters.csv")
    check(len(stage21) == 54441, "Stage 21 episode count differs")
    check(int(stage21.loc[stage21.color.eq("rojo"), "observed_evaluable"].sum()) == 1744,
          "Red evaluability differs")
    # These are observation-pattern categories, not inferred causal reasons.
    stage21["observable_coverage_pattern"] = "unresolved_incomplete_grid"
    stage21.loc[stage21.observed_evaluable, "observable_coverage_pattern"] = "evaluable"
    stage21.loc[~stage21.observed_evaluable & ~stage21.start_observed,
                "observable_coverage_pattern"] = "missing_start_state"
    stage21.loc[~stage21.observed_evaluable & stage21.start_observed & ~stage21.end_observed,
                "observable_coverage_pattern"] = "missing_end_state"
    coverage = stage21.groupby(["color", "observable_coverage_pattern"]).size().rename("episodes").reset_index()
    coverage.to_csv(OUT / "coverage_patterns.csv", index=False)
    check(coverage.episodes.sum() == 54441, "Coverage categories do not partition episodes")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
