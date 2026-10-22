#!/usr/bin/env python3
"""Aggregate first-week primary interactions to one pre-event row per window."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK, SCOPE_ERROR, assert_pilot_frame, scope_config

NUMERIC = [
    "horizontal_separation_nm", "vertical_separation_ft", "cpa_nm", "tcpa_min",
    "predicted_min_horizontal_sep_nm", "predicted_min_vertical_sep_ft",
    "horizontal_closing_rate_nm_min", "vertical_closing_rate_fpm",
    "relative_heading_deg", "crossing_angle_deg", "relative_bearing_deg",
    "traffic_density", "nearby_aircraft_count_30nm", "relative_velocity_kt",
    "topsky_predicted_min_sep_nm_proxy", "topsky_predicted_time_to_min_s_proxy",
    "flight_level", "focal_ground_speed_kt", "focal_vertical_rate_fpm",
]
META = ["window_id", "archive_week", "clearance_id", "control_for_clearance_id", "ATC_INTERVENTION",
        "focal_flight_key", "focal_flight_phase", "focal_sector", "event_epoch", "relative_seconds",
        "predictor_eligible", "other_flight_key"]
USED = META + NUMERIC
HORIZONS = (60, 120, 300, 600)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def percentile(series: pd.Series, q: float) -> float:
    clean = numeric(series).dropna()
    return float(clean.quantile(q)) if len(clean) else np.nan


def first_value(group: pd.DataFrame, column: str):
    valid = group[column].dropna()
    return valid.iloc[0] if len(valid) else None


def trend_at_horizon(times: np.ndarray, values: np.ndarray, horizon: int) -> float:
    good = np.isfinite(times) & np.isfinite(values)
    if not good.any():
        return np.nan
    t = times[good]
    v = values[good]
    current_index = int(np.argmin(np.abs(t)))
    reference_index = int(np.argmin(np.abs(t + horizon)))
    if abs(t[reference_index] + horizon) > 15 or abs(t[current_index]) > 15:
        return np.nan
    return float(v[current_index] - v[reference_index])


def aggregate_window(group: pd.DataFrame) -> dict:
    if group["ATC_INTERVENTION"].nunique(dropna=False) != 1 or group["archive_week"].nunique(dropna=False) != 1:
        raise SystemExit("Inconsistent target or week within window")
    row = {"window_id": group["window_id"].iloc[0], "archive_week": PILOT_WEEK,
           "analysis_scope": "FIRST_WEEK_PILOT", "ATC_INTERVENTION": int(group["ATC_INTERVENTION"].iloc[0]),
           "clearance_id": first_value(group, "clearance_id"),
           "control_for_clearance_id": first_value(group, "control_for_clearance_id"),
           "focal_flight_key": first_value(group, "focal_flight_key"),
           "flight_phase": first_value(group, "focal_flight_phase"),
           "centre": first_value(group, "focal_sector"),
           "event_epoch": first_value(group, "event_epoch"),
           "n_pair_rows_pre_event": len(group),
           "n_unique_neighbours": group["other_flight_key"].nunique()}
    rel = numeric(group["relative_seconds"])
    anchor_time = float(rel.iloc[np.argmin(np.abs(rel.to_numpy()))])
    anchor = group.loc[rel == anchor_time]
    anchor_nearest = anchor.iloc[int(np.nanargmin(numeric(anchor["horizontal_separation_nm"]).fillna(np.inf).to_numpy()))]
    row["anchor_relative_seconds"] = anchor_time
    row["horizontal_sep_at_event"] = numeric(pd.Series([anchor_nearest["horizontal_separation_nm"]])).iloc[0]
    row["vertical_sep_at_event"] = numeric(pd.Series([anchor_nearest["vertical_separation_ft"]])).iloc[0]
    for source, prefix in (("horizontal_separation_nm", "horizontal_separation"),
                           ("vertical_separation_ft", "vertical_separation")):
        vals = numeric(group[source])
        row[f"min_{prefix}"] = vals.min()
        row[f"p05_{prefix}"] = vals.quantile(.05)
        row[f"median_{prefix}"] = vals.median()
    cpa = numeric(group["cpa_nm"])
    row["min_predicted_horizontal_cpa"] = cpa.min()
    if cpa.notna().any():
        cpa_row = group.loc[cpa.idxmin()]
        row["vertical_sep_at_min_cpa"] = cpa_row["predicted_min_vertical_sep_ft"]
        row["tcpa_at_min_cpa"] = cpa_row["tcpa_min"]
        row["horizontal_closing_rate_at_min_cpa"] = cpa_row["horizontal_closing_rate_nm_min"]
    else:
        for field in ("vertical_sep_at_min_cpa", "tcpa_at_min_cpa", "horizontal_closing_rate_at_min_cpa"):
            row[field] = np.nan
    tcpa = numeric(group["tcpa_min"])
    row["min_positive_tcpa"] = tcpa[tcpa > 0].min()
    for source, name in (("horizontal_closing_rate_nm_min", "max_horizontal_closing_rate"),
                         ("vertical_closing_rate_fpm", "max_vertical_closing_rate")):
        row[name] = numeric(group[source]).max()
    for source, name in (("relative_heading_deg", "median_relative_heading"),
                         ("crossing_angle_deg", "median_crossing_angle"),
                         ("relative_bearing_deg", "median_relative_bearing"),
                         ("traffic_density", "median_traffic_density"),
                         ("nearby_aircraft_count_30nm", "median_nearby_aircraft_30nm"),
                         ("relative_velocity_kt", "median_relative_velocity_kt"),
                         ("flight_level", "flight_level"),
                         ("focal_ground_speed_kt", "ground_speed_kt"),
                         ("focal_vertical_rate_fpm", "vertical_rate_fpm")):
        row[name] = numeric(group[source]).median()
    row["min_kinematic_predicted_sep_nm"] = numeric(group["predicted_min_horizontal_sep_nm"]).min()
    row["min_topsky_predicted_sep_nm_proxy"] = numeric(group["topsky_predicted_min_sep_nm_proxy"]).min()
    row["min_topsky_time_to_min_s_proxy"] = numeric(group["topsky_predicted_time_to_min_s_proxy"]).min()
    by_time = group.groupby("relative_seconds", sort=True)[["horizontal_separation_nm", "vertical_separation_ft"]].min()
    times = pd.to_numeric(by_time.index, errors="coerce").to_numpy(dtype=float)
    for horizon in HORIZONS:
        row[f"horizontal_sep_trend_{horizon}s"] = trend_at_horizon(times, by_time["horizontal_separation_nm"].to_numpy(dtype=float), horizon)
        row[f"vertical_sep_trend_{horizon}s"] = trend_at_horizon(times, by_time["vertical_separation_ft"].to_numpy(dtype=float), horizon)
    if pd.notna(row["event_epoch"]):
        row["hour_utc"] = int(pd.to_datetime(row["event_epoch"], unit="s", utc=True).hour)
    else:
        row["hour_utc"] = np.nan
    return row


def write_partitions(source: Path, root: Path, bucket_count: int, batch_size: int) -> tuple[list[Path], int]:
    parquet = pq.ParquetFile(source)
    missing = sorted(set(USED) - set(parquet.schema.names))
    if missing:
        raise SystemExit(f"Missing stage 05 columns: {missing}")
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"pilot_pre_event_bucket_{i:02d}.parquet" for i in range(bucket_count)]
    writers: list[pq.ParquetWriter | None] = [None] * bucket_count
    retained = 0
    checked = 0
    try:
        for batch in parquet.iter_batches(columns=USED, batch_size=batch_size):
            week = batch.column(batch.schema.get_field_index("archive_week"))
            if week.null_count or set(pc.unique(week).to_pylist()) != {PILOT_WEEK}:
                raise SystemExit(SCOPE_ERROR)
            eligible = pc.and_(pc.less_equal(batch.column(batch.schema.get_field_index("relative_seconds")), 0),
                               batch.column(batch.schema.get_field_index("predictor_eligible")))
            table = pa.Table.from_batches([batch]).filter(eligible)
            checked += batch.num_rows
            if table.num_rows == 0:
                continue
            retained += table.num_rows
            ids = table["window_id"].to_pylist()
            # A stable mapping keeps every observation of a window in one bucket.
            import zlib
            buckets = np.fromiter((zlib.crc32(str(value).encode()) % bucket_count for value in ids), dtype=np.int16, count=len(ids))
            for bucket in np.unique(buckets):
                indices = np.flatnonzero(buckets == bucket)
                subset = table.take(pa.array(indices, type=pa.int64()))
                if writers[bucket] is None:
                    writers[bucket] = pq.ParquetWriter(paths[bucket], subset.schema, compression="zstd")
                writers[bucket].write_table(subset)
            if checked // 5_000_000 != (checked - batch.num_rows) // 5_000_000:
                print(f"Stage 05b partition: scanned {checked:,}; pre-event rows {retained:,}", flush=True)
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
    return [path for path in paths if path.exists()], retained


def quality_report(frame: pd.DataFrame, cohort: pd.DataFrame, pre_rows: int) -> str:
    expected = set(cohort.intervention_window_id) | set(cohort.control_window_id)
    seen = set(frame.window_id)
    missing = expected - seen
    interventions = int((frame.ATC_INTERVENTION == 1).sum())
    controls = int((frame.ATC_INTERVENTION == 0).sum())
    numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    lines = ["# SCAT first-week window feature quality", "", "- analysis_scope: `FIRST_WEEK_PILOT`",
             f"- archive_week: `{PILOT_WEEK}`", f"- Primary matched pairs: **{len(cohort):,}**",
             f"- Expected windows: **{len(expected):,}**", f"- Observed windows: **{len(frame):,}**",
             f"- Intervention windows: **{interventions:,}**", f"- Control windows: **{controls:,}**",
             f"- Unique focal flights: **{frame.focal_flight_key.nunique():,}**",
             f"- Missing windows: **{len(missing):,}**", f"- Duplicate window IDs: **{int(frame.window_id.duplicated().sum())}**",
             f"- Eligible pre-event pair rows aggregated: **{pre_rows:,}**", "",
             "Missing windows are reported and never synthesized.", "", "## Missingness", "",
             "| Feature | Missing | Missing share |", "|---|---:|---:|"]
    for col in numeric_cols:
        n = int(frame[col].isna().sum())
        lines.append(f"| {col} | {n:,} | {n / len(frame):.4f} |")
    if missing:
        lines += ["", "## Missing window IDs (first 20)", ""] + [f"- `{value}`" for value in sorted(missing)[:20]]
    return "\n".join(lines) + "\n"


def physical_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    mapping = {"horizontal separation": "min_horizontal_separation", "vertical separation": "min_vertical_separation",
               "CPA": "min_predicted_horizontal_cpa", "TCPA": "min_positive_tcpa",
               "horizontal closing rate": "max_horizontal_closing_rate", "vertical closing rate": "max_vertical_closing_rate",
               "crossing angle": "median_crossing_angle", "ground speed": "ground_speed_kt",
               "vertical rate": "vertical_rate_fpm", "traffic density": "median_traffic_density",
               "number neighbours": "n_unique_neighbours"}
    rows = []
    warnings = []
    for label, column in mapping.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        clean = values.replace([np.inf, -np.inf], np.nan).dropna()
        result = {"feature": label, "column": column, "nan_count": int(values.isna().sum()),
                  "inf_count": int(np.isinf(values).sum())}
        for name, q in (("min", 0), ("p01", .01), ("p05", .05), ("median", .5), ("p95", .95), ("p99", .99), ("max", 1)):
            result[name] = float(clean.quantile(q)) if len(clean) else np.nan
        rows.append(result)
        if result["inf_count"]:
            warnings.append(f"{column}: {result['inf_count']} infinite values")
    for column in ("min_horizontal_separation", "min_vertical_separation", "min_predicted_horizontal_cpa", "ground_speed_kt"):
        if (pd.to_numeric(frame[column], errors="coerce") < 0).any():
            warnings.append(f"{column}: negative value")
    if (pd.to_numeric(frame["median_crossing_angle"], errors="coerce") > 180).any() or (pd.to_numeric(frame["median_crossing_angle"], errors="coerce") < 0).any():
        warnings.append("median_crossing_angle: outside 0–180 degrees")
    if (pd.to_numeric(frame["median_traffic_density"], errors="coerce") < 1).any():
        warnings.append("median_traffic_density: less than one")
    if (pd.to_numeric(frame["max_vertical_closing_rate"], errors="coerce") > 15000).any():
        warnings.append("Some pairwise vertical closing rates exceed 15,000 ft/min; inspect tails and units before interpreting clusters")
    if (pd.to_numeric(frame["min_positive_tcpa"], errors="coerce") < 1 / 60).any():
        warnings.append("Minimum positive TCPA can be below one second because it is an extreme over all pre-event pairs/times; interpret cautiously")
    if (pd.to_numeric(frame["min_horizontal_separation"], errors="coerce") > 200).any():
        warnings.append("Some windows have minimum pairwise horizontal separation above 200 NM; confirm neighbourhood definition")
    return pd.DataFrame(rows), warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"primary_interaction_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows" / f"primary_matching_cohort_{PILOT_WEEK}.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--buckets", type=int, default=64)
    args = parser.parse_args()
    cohort = pd.read_parquet(args.cohort)
    assert_pilot_frame(cohort)
    bucket_root = args.output.parent / "window_feature_partitions"
    paths, pre_rows = write_partitions(args.input, bucket_root, args.buckets, 100_000)
    pieces = []
    for index, path in enumerate(paths, start=1):
        frame = pd.read_parquet(path)
        assert_pilot_frame(frame)
        group_rows = [aggregate_window(group) for _, group in frame.groupby("window_id", sort=False)]
        pieces.append(pd.DataFrame(group_rows))
        print(f"Stage 05b aggregate: partition {index}/{len(paths)}; windows={sum(len(piece) for piece in pieces):,}", flush=True)
    result = pd.concat(pieces, ignore_index=True)
    assert_pilot_frame(result)
    if result.window_id.duplicated().any():
        raise SystemExit("Duplicate window IDs after aggregation")
    if (pd.to_numeric(result.anchor_relative_seconds, errors="coerce") > 0).any():
        raise SystemExit("Temporal leakage: post-event anchor")
    expected = set(cohort.intervention_window_id) | set(cohort.control_window_id)
    if set(result.window_id) - expected:
        raise SystemExit("Unexpected window IDs in stage 05b")
    result.to_parquet(args.output, index=False, compression="zstd")
    (args.output.parent / "window_feature_quality_report.md").write_text(quality_report(result, cohort, pre_rows), encoding="utf-8")
    sanity, warnings = physical_report(result)
    sanity.to_csv(args.output.parent / "window_feature_physical_percentiles.csv", index=False)
    (args.output.parent / "window_feature_physical_report.md").write_text(
        "# SCAT first-week physical sanity\n\n" + ("Status: WARNING\n\n" + "\n".join(f"- {warning}" for warning in warnings)
        if warnings else "Status: PASS\n\nNo impossible values detected by the configured checks.") + "\n\nNo outliers were removed.\n", encoding="utf-8")
    (args.output.parent / "window_feature_run_config.json").write_text(json.dumps({**scope_config(), "input": str(args.input),
        "cohort": str(args.cohort), "output": str(args.output), "pre_event_only": True, "bucket_count": args.buckets,
        "predictor_blacklist": ["ATC_INTERVENTION", "clearance_type", "matching_level", "assigned_heading", "assigned_level", "assigned_speed"]}, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 05b complete: windows={len(result):,}, missing={len(expected - set(result.window_id)):,}, physical_warnings={len(warnings)}", flush=True)


if __name__ == "__main__":
    main()
