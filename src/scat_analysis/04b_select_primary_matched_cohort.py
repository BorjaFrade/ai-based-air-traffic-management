#!/usr/bin/env python3
"""Select the density-preserving SCAT matched cohort from stage 04 metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import ANALYSIS_SCOPE, PILOT_WEEK, assert_pilot_frame, scope_config

PRIMARY_LEVELS = {"exact", "without_aircraft_type"}


def smd(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce")
    b = pd.to_numeric(right, errors="coerce")
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    if paired.empty:
        return np.nan
    scale = np.sqrt((paired.a.var(ddof=1) + paired.b.var(ddof=1)) / 2)
    difference = paired.a.mean() - paired.b.mean()
    return float(difference / scale) if scale > 0 else (0.0 if difference == 0 else np.inf)


def balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, left, right in (
        ("local_traffic_density", "intervention_local_traffic_density", "control_local_traffic_density"),
        ("flight_level", "intervention_flight_level", "control_flight_level"),
        ("hour_utc", "intervention_hour_utc", "control_hour_utc"),
    ):
        rows.append({"covariate": name, "category": "", "intervention_mean": pd.to_numeric(frame[left], errors="coerce").mean(),
                     "control_mean": pd.to_numeric(frame[right], errors="coerce").mean(), "smd": smd(frame[left], frame[right])})
    for name in ("phase", "centre"):
        left = frame[f"intervention_{name}"].fillna("").astype(str)
        right = frame[f"control_{name}"].fillna("").astype(str)
        for category in sorted(set(left) | set(right)):
            rows.append({"covariate": name, "category": category, "intervention_mean": (left == category).mean(),
                         "control_mean": (right == category).mean(), "smd": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows" / "matching_quality.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows")
    args = parser.parse_args()
    quality = pd.read_csv(args.matching_path, low_memory=False)
    primary = quality.loc[quality["matched"].astype(str).str.lower().isin({"true", "1"})
                          & quality["matching_level"].isin(PRIMARY_LEVELS)].copy()
    if primary.empty or primary["clearance_id"].isna().any() or primary["clearance_id"].duplicated().any():
        raise SystemExit("Primary matching IDs are missing or not unique")
    if not (primary["intervention_window_id"] == "I:" + primary["clearance_id"].astype(str)).all():
        raise SystemExit("Intervention window IDs do not match clearance IDs")
    if not (primary["intervention_archive_week"] == primary["control_archive_week"]).all():
        raise SystemExit("Cross-week primary match found")
    primary["archive_week"] = primary["intervention_archive_week"].astype(str)
    primary["control_window_id"] = "C:" + primary["clearance_id"].astype(str)
    pilot = primary.loc[primary["archive_week"] == PILOT_WEEK].copy()
    assert_pilot_frame(pilot)
    args.output_root.mkdir(parents=True, exist_ok=True)
    primary.to_parquet(args.output_root / "primary_matching_cohort.parquet", index=False)
    pilot.to_parquet(args.output_root / f"primary_matching_cohort_{PILOT_WEEK}.parquet", index=False)
    balance = balance_table(pilot)
    balance.to_csv(args.output_root / "primary_matching_balance.csv", index=False)
    density_smd = balance.loc[balance.covariate == "local_traffic_density", "smd"].iloc[0]
    level_smd = balance.loc[balance.covariate == "flight_level", "smd"].iloc[0]
    time_distance = pd.to_numeric(pilot["control_time_difference_s"], errors="coerce")
    lines = ["# SCAT primary matching cohort", "", f"- analysis_scope: `{ANALYSIS_SCOPE}`", f"- archive_week: `{PILOT_WEEK}`", "",
             "## All 13 source weeks (stage 04 only)", "", f"- Primary pairs: **{len(primary):,}**", "",
             "## FIRST WEEK PILOT", "", f"- Matched pairs: **{len(pilot):,}**",
             f"- Exact: **{int((pilot.matching_level == 'exact').sum()):,}**",
             f"- Without aircraft type: **{int((pilot.matching_level == 'without_aircraft_type').sum()):,}**",
             f"- Intervention windows: **{pilot.intervention_window_id.nunique():,}**",
             f"- Control windows: **{pilot.control_window_id.nunique():,}**",
             f"- SMD local density: **{density_smd:.4f}**",
             f"- SMD flight level: **{level_smd:.4f}**",
             f"- Control time distance median / p95 (s): **{time_distance.median():.1f} / {time_distance.quantile(.95):.1f}**",
             "", "### Phases and centres", "", "| Covariate | Category | Intervention share | Control share |", "|---|---|---:|---:|"]
    for row in balance.loc[balance.covariate.isin(["phase", "centre"])].itertuples():
        lines.append(f"| {row.covariate} | {row.category} | {row.intervention_mean:.3f} | {row.control_mean:.3f} |")
    (args.output_root / "primary_matching_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_root / "primary_matching_run_config.json").write_text(json.dumps({**scope_config(), "source": str(args.matching_path),
        "primary_levels": sorted(PRIMARY_LEVELS), "global_cohort_is_stage04_only": True}, indent=2) + "\n", encoding="utf-8")
    print(f"Primary matching: global={len(primary)} pilot={len(pilot)} exact={(pilot.matching_level == 'exact').sum()} no_type={(pilot.matching_level == 'without_aircraft_type').sum()}", flush=True)


if __name__ == "__main__":
    main()
