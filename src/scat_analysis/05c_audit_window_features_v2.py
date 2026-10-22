#!/usr/bin/env python3
"""Audit first-week pair-feature tails and add traceable altitude to Stage 05b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK, SCOPE_ERROR, assert_pilot_frame, assert_pilot_parquet, scope_config


PAIR_COLUMNS = ["window_id", "archive_week", "relative_seconds", "predictor_eligible", "other_flight_key",
                "focal_altitude_ft", "flight_level", "horizontal_separation_nm",
                "cpa_nm", "tcpa_min", "vertical_closing_rate_fpm",
                "focal_vertical_rate_fpm", "other_vertical_rate_fpm", "vertical_separation_ft",
                "focal_ground_speed_kt", "focal_flight_phase", "focal_sector",
                "horizontal_closing_rate_nm_min"]
NUMERIC_COLUMNS = ["focal_altitude_ft", "flight_level", "horizontal_separation_nm", "cpa_nm",
                   "tcpa_min", "vertical_closing_rate_fpm", "focal_vertical_rate_fpm",
                   "other_vertical_rate_fpm", "vertical_separation_ft", "focal_ground_speed_kt",
                   "horizontal_closing_rate_nm_min"]


def band_of(altitude: float) -> str:
    if not np.isfinite(altitude) or altitude < -1000 or altitude > 60000:
        return "UNKNOWN_INVALID"
    if altitude < 2000:
        return "VERY_LOW"
    if altitude < 8000:
        return "LOW"
    if altitude < 29000:
        return "MID"
    if altitude <= 41000:
        return "CRUISE_FL290_410"
    return "ABOVE_FL410"


def update_min(current, value):
    return value if pd.isna(current) else current if pd.isna(value) else min(current, value)


def update_max(current, value):
    return value if pd.isna(current) else current if pd.isna(value) else max(current, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"primary_interaction_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--windows", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_v2_{PILOT_WEEK}.parquet")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this generated V2 artifact after a validated rerun")
    args = parser.parse_args()
    if args.pairs.name != f"primary_interaction_features_{PILOT_WEEK}.parquet" or args.windows.name != f"window_features_{PILOT_WEEK}.parquet":
        raise SystemExit("V2 requires the existing first-week primary pair and V1 window files")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"V2 already exists: {args.output}; preserve it until explicitly reviewed")
    assert_pilot_parquet(args.windows)
    windows = pd.read_parquet(args.windows)
    assert_pilot_frame(windows)
    if windows.window_id.duplicated().any():
        raise SystemExit("Duplicate V1 window IDs")
    parquet = pq.ParquetFile(args.pairs)
    if set(PAIR_COLUMNS) - set(parquet.schema.names):
        raise SystemExit("Missing required pair columns for V2 audit")
    stats: dict[str, dict] = {}
    anchor: dict[str, tuple] = {}
    nearby_neighbours: dict[str, set[str]] = {}
    totals = {key: 0 for key in ("scanned_rows", "eligible_rows", "far30_rows", "far200_rows",
                                    "tiny_positive_tcpa_rows", "high_vertical_closing_rows",
                                    "measured_fl_rows", "fl_altitude_mismatch_rows", "fallback_altitude_rows",
                                    "vertical_rate_formula_mismatch_rows")}
    max_rate_formula_error = 0.0
    for batch in parquet.iter_batches(columns=PAIR_COLUMNS, batch_size=100_000):
        weeks = batch.column(batch.schema.get_field_index("archive_week"))
        if weeks.null_count or set(pc.unique(weeks).to_pylist()) != {PILOT_WEEK}:
            raise SystemExit(SCOPE_ERROR)
        totals["scanned_rows"] += batch.num_rows
        eligible = pc.and_(pc.less_equal(batch.column(batch.schema.get_field_index("relative_seconds")), 0),
                           batch.column(batch.schema.get_field_index("predictor_eligible")))
        table = pa.Table.from_batches([batch]).filter(eligible)
        if table.num_rows == 0:
            continue
        x = table.to_pandas()
        totals["eligible_rows"] += len(x)
        for col in NUMERIC_COLUMNS:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        sep = x.horizontal_separation_nm
        tcpa = x.tcpa_min
        rate = x.vertical_closing_rate_fpm
        has_fl = x.flight_level.notna()
        altitude_error = (x.focal_altitude_ft - x.flight_level * 100).abs()
        totals["measured_fl_rows"] += int(has_fl.sum())
        totals["fallback_altitude_rows"] += int((~has_fl).sum())
        totals["fl_altitude_mismatch_rows"] += int((has_fl & (altitude_error > .01)).sum())
        rate_error = (rate.abs() - (x.other_vertical_rate_fpm - x.focal_vertical_rate_fpm).abs()).abs()
        comparable = x.vertical_separation_ft.gt(0) & rate_error.notna()
        totals["vertical_rate_formula_mismatch_rows"] += int((comparable & rate_error.gt(.01)).sum())
        x["rate_metadata_mismatch"] = (comparable & rate_error.gt(.01)).astype("int8")
        if comparable.any():
            max_rate_formula_error = max(max_rate_formula_error, float(rate_error[comparable].max()))
        x["far30"] = sep.gt(30).astype("int8")
        x["far200"] = sep.gt(200).astype("int8")
        x["tiny_tcpa"] = (tcpa.gt(0) & tcpa.lt(1 / 60)).astype("int8")
        x["high_rate"] = rate.abs().gt(15000).astype("int8")
        x["near_sep"] = sep.where(sep.le(30))
        x["near_cpa"] = x.cpa_nm.where(sep.le(30))
        x["near_rate"] = rate.where(sep.le(30))
        x["near_horizontal_closing"] = x.horizontal_closing_rate_nm_min.where(sep.le(30))
        x["tcpa_ge_1s_near"] = tcpa.where(sep.le(30) & tcpa.ge(1 / 60))
        for row in x.loc[sep.le(30), ["window_id", "other_flight_key"]].drop_duplicates().itertuples(index=False):
            nearby_neighbours.setdefault(str(row.window_id), set()).add(str(row.other_flight_key))
        totals["far30_rows"] += int(x.far30.sum())
        totals["far200_rows"] += int(x.far200.sum())
        totals["tiny_positive_tcpa_rows"] += int(x.tiny_tcpa.sum())
        totals["high_vertical_closing_rows"] += int(x.high_rate.sum())
        grouped = x.groupby("window_id", sort=False).agg(
            eligible_pair_rows=("window_id", "size"), far30_rows=("far30", "sum"),
            far200_rows=("far200", "sum"), tiny_positive_tcpa_rows=("tiny_tcpa", "sum"),
            high_vertical_closing_rows=("high_rate", "sum"),
            rate_metadata_mismatch_rows=("rate_metadata_mismatch", "sum"),
            min_horizontal_sep_within_30nm=("near_sep", "min"),
            min_cpa_within_30nm=("near_cpa", "min"),
            max_vertical_closing_within_30nm=("near_rate", "max"),
            max_horizontal_closing_within_30nm=("near_horizontal_closing", "max"),
            min_tcpa_ge_1s_within_30nm=("tcpa_ge_1s_near", "min"))
        for window_id, row in grouped.iterrows():
            key = str(window_id)
            item = stats.setdefault(key, {"eligible_pair_rows": 0, "far30_rows": 0, "far200_rows": 0,
                "tiny_positive_tcpa_rows": 0, "high_vertical_closing_rows": 0,
                "rate_metadata_mismatch_rows": 0,
                "min_horizontal_sep_within_30nm": np.nan, "min_cpa_within_30nm": np.nan,
                "max_vertical_closing_within_30nm": np.nan, "max_horizontal_closing_within_30nm": np.nan,
                "min_tcpa_ge_1s_within_30nm": np.nan})
            for field in ("eligible_pair_rows", "far30_rows", "far200_rows", "tiny_positive_tcpa_rows", "high_vertical_closing_rows", "rate_metadata_mismatch_rows"):
                item[field] += int(row[field])
            for field in ("min_horizontal_sep_within_30nm", "min_cpa_within_30nm", "min_tcpa_ge_1s_within_30nm"):
                item[field] = update_min(item[field], row[field])
            for field in ("max_vertical_closing_within_30nm", "max_horizontal_closing_within_30nm"):
                item[field] = update_max(item[field], row[field])
        x["abs_relative_seconds"] = x.relative_seconds.abs()
        chosen = x.loc[x.groupby("window_id", sort=False).abs_relative_seconds.idxmin()]
        for row in chosen.itertuples(index=False):
            key = str(row.window_id)
            candidate = (float(row.abs_relative_seconds), float(row.relative_seconds),
                         float(row.focal_altitude_ft), float(row.flight_level) if pd.notna(row.flight_level) else np.nan,
                         str(row.focal_flight_phase or ""), str(row.focal_sector or ""),
                         float(row.focal_ground_speed_kt) if pd.notna(row.focal_ground_speed_kt) else np.nan,
                         float(row.focal_vertical_rate_fpm) if pd.notna(row.focal_vertical_rate_fpm) else np.nan)
            if key not in anchor or candidate[0] < anchor[key][0]:
                anchor[key] = candidate
        if totals["scanned_rows"] // 5_000_000 != (totals["scanned_rows"] - batch.num_rows) // 5_000_000:
            print(f"V2 audit: scanned {totals['scanned_rows']:,}; eligible {totals['eligible_rows']:,}", flush=True)
    if totals["fl_altitude_mismatch_rows"]:
        print(f"V2 validation diagnostics: {totals}; max_rate_formula_error_fpm={max_rate_formula_error}", flush=True)
        raise SystemExit("V2 FL-to-altitude unit validation failed; inspect SCAT pair features")
    expected = set(windows.window_id.astype(str))
    if set(stats) != expected or set(anchor) != expected:
        raise SystemExit(f"V2 window coverage failed: stats={len(stats)} anchors={len(anchor)} expected={len(expected)}")
    additions = []
    for window_id in windows.window_id.astype(str):
        best = anchor[window_id]
        altitude_ft, measured_fl = best[2], best[3]
        source = "measured_FL_I062_136_x100" if np.isfinite(measured_fl) else "selected_altitude_fallback_I062_380"
        item = {"window_id": window_id, **stats[window_id], "flight_level_original": measured_fl,
                "altitude_original": altitude_ft, "altitude_source": source,
                "analysis_altitude_ft": altitude_ft, "altitude_band": band_of(altitude_ft),
                "altitude_anchor_relative_seconds": best[1],
                "flight_phase_at_anchor": best[4], "centre_at_anchor": best[5],
                "ground_speed_at_anchor_kt": best[6], "vertical_rate_at_anchor_fpm": best[7]}
        item["n_unique_neighbours_within_30nm"] = len(nearby_neighbours.get(window_id, set()))
        item["far30_pair_fraction"] = item["far30_rows"] / item["eligible_pair_rows"]
        additions.append(item)
    result = windows.merge(pd.DataFrame(additions), on="window_id", how="left", validate="one_to_one")
    result["flight_phase_v1_first_pre_event"] = result["flight_phase"]
    result["centre_v1_first_pre_event"] = result["centre"]
    result["flight_phase"] = result["flight_phase_at_anchor"]
    result["centre"] = result["centre_at_anchor"]
    assert_pilot_frame(result)
    if result.window_id.duplicated().any() or len(result) != len(windows) or result.analysis_altitude_ft.isna().any():
        raise SystemExit("Invalid V2 output coverage or altitude")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.stem + ".pending.parquet")
    result.to_parquet(pending, index=False, compression="zstd")
    pending.replace(args.output)
    totals.update({"window_count": len(result), "altitude_source_counts": result.altitude_source.value_counts().to_dict(),
                   "band_counts": result.altitude_band.value_counts().to_dict(),
                   "windows_without_neighbour_within_30nm": int(result.min_horizontal_sep_within_30nm.isna().sum()),
                   "windows_with_any_far200_row": int(result.far200_rows.gt(0).sum()),
                   "max_vertical_rate_formula_error_fpm": max_rate_formula_error})
    (args.output.parent / "window_features_v2_audit.json").write_text(json.dumps({**scope_config(), **totals}, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "window_features_v2_run_config.json").write_text(json.dumps({**scope_config(),
        "source_pairs": str(args.pairs), "source_windows": str(args.windows), "output": str(args.output),
        "pre_event_only": True, "analysis_altitude_ft": "focal_altitude_ft at nearest nonpositive predictor-eligible time",
        "flight_level_original": "I062/136 measured_flight_level, in hundreds of feet",
        "altitude_original": "Stage 05 focal_altitude_ft; measured FL*100 or selected altitude fallback",
        "flight_phase_and_centre": "re-anchored to nearest predictor-eligible pre-event pair row; V1 first-pre-event values retained in suffixed columns",
        "altitude_valid_range_ft": [-1000, 60000], "neighbour_diagnostic_radius_nm": 30,
        "post_event_rows_used": False, "no_stage05_recompute": True}, indent=2) + "\n", encoding="utf-8")
    report = ["# Stage 05b V2 physical and altitude audit", "", f"- analysis_scope: `FIRST_WEEK_PILOT`",
              f"- archive_week: `{PILOT_WEEK}`", f"- Windows preserved: **{len(result):,}**",
              f"- Eligible pre-event pair rows: **{totals['eligible_rows']:,}**",
              "- `flight_level_original`: I062/136 measured flight level in hundreds of feet.",
              "- `altitude_original`: Stage 05 `focal_altitude_ft` at the nearest eligible pre-event time.",
              "- `analysis_altitude_ft`: same numeric altitude in feet; measured FL × 100 when present, otherwise selected-altitude fallback from I062/380. It is not a geometric altitude.",
              "- V2 `flight_phase` and `centre` are re-anchored to the nearest eligible pre-event row; V1 first-pre-event values are retained in audit columns. Ground speed and vertical rate at that anchor are also recorded separately from V1 window medians.",
              "- Fallback source cannot distinguish I062/380 subitems 6/7 from Stage 05 alone; the extractor tries 6 before 7.",
              f"- FL-to-altitude mismatches: **{totals['fl_altitude_mismatch_rows']:,}**; pairwise vertical-rate versus focal metadata discrepancies: **{totals['vertical_rate_formula_mismatch_rows']:,}**.",
              "- Vertical-rate metadata discrepancy is a WARNING, not an altitude-unit failure: Stage 05 computes pair geometry from a focal state in the same 5-second bucket, then decorates it with the target row's focal rate; sampled discrepancies match this alignment difference. V2 records affected rows per window.",
              f"- Pair rows >30 NM: **{totals['far30_rows']:,}**; >200 NM: **{totals['far200_rows']:,}**. Stage 05 selected up to ten nearest neighbours without a hard 30-NM cutoff.",
              f"- Positive TCPA <1 s rows: **{totals['tiny_positive_tcpa_rows']:,}**; |vertical closing| >15,000 ft/min rows: **{totals['high_vertical_closing_rows']:,}**.",
              "- V2 keeps original V1 variables for auditability and adds within-30-NM neighbour counts, separation, CPA and closing-rate diagnostics; it does not remove outliers or fabricate missing windows.",
              "- Minimum TCPA over many pairs/times is an unstable extreme. `min_tcpa_ge_1s_within_30nm` is diagnostic only, not a conflict threshold or automatic model input.",
              "- `UNKNOWN_INVALID` means non-finite altitude or outside the documented -1,000 to 60,000 ft plausibility range, not a safety state.",
              "", "## Altitude source counts", ""]
    report.extend(f"- {key}: **{value:,}**" for key, value in totals["altitude_source_counts"].items())
    report += ["", "## Band counts (windows before pair exclusions)", ""]
    report.extend(f"- {key}: **{value:,}**" for key, value in totals["band_counts"].items())
    (args.output.parent / "window_features_v2_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"V2 complete: windows={len(result):,}; bands={totals['band_counts']}", flush=True)


if __name__ == "__main__":
    main()
