#!/usr/bin/env python3
"""Reuse the completed first-week stage 05 output by exact window IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK, SCOPE_ERROR, assert_pilot_frame, scope_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / "interaction_features.parquet")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows" / f"primary_matching_cohort_{PILOT_WEEK}.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"primary_interaction_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    cohort = pd.read_parquet(args.cohort)
    assert_pilot_frame(cohort)
    if not cohort.matching_level.isin({"exact", "without_aircraft_type"}).all():
        raise SystemExit("Pilot cohort includes a non-primary matching level")
    ids = set(cohort.intervention_window_id.astype(str)) | set(cohort.control_window_id.astype(str))
    if len(ids) != 2 * len(cohort):
        raise SystemExit("Matched pair window IDs are missing or duplicated")
    value_set = pa.array(sorted(ids))
    source = pq.ParquetFile(args.input)
    if "archive_week" not in source.schema.names or "window_id" not in source.schema.names:
        raise SystemExit(f"{SCOPE_ERROR}; required columns missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".parquet.tmp")
    writer = None
    seen = set()
    selected_rows = 0
    scanned_rows = 0
    try:
        for index, batch in enumerate(source.iter_batches(batch_size=args.batch_size), start=1):
            week = batch.column(batch.schema.get_field_index("archive_week"))
            if week.null_count or set(pc.unique(week).to_pylist()) != {PILOT_WEEK}:
                raise SystemExit(SCOPE_ERROR)
            mask = pc.is_in(batch.column(batch.schema.get_field_index("window_id")), value_set=value_set)
            filtered = pa.Table.from_batches([batch]).filter(mask)
            scanned_rows += batch.num_rows
            if filtered.num_rows:
                seen.update(pc.unique(filtered["window_id"]).to_pylist())
                if writer is None:
                    writer = pq.ParquetWriter(temporary, filtered.schema, compression="zstd")
                writer.write_table(filtered)
                selected_rows += filtered.num_rows
            if index % 100 == 0:
                print(f"Stage 05 reuse: scanned {scanned_rows:,}/{source.metadata.num_rows:,}; retained {selected_rows:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise SystemExit("No primary features found")
    missing = sorted(ids - seen)
    unexpected = sorted(seen - ids)
    if unexpected:
        raise SystemExit(f"Unexpected window IDs in selection: {unexpected[:5]}")
    temporary.replace(args.output)
    report = ["# First-week primary interaction features", "", "- analysis_scope: `FIRST_WEEK_PILOT`",
              f"- archive_week: `{PILOT_WEEK}`", f"- Source rows scanned: **{scanned_rows:,}**",
              f"- Rows retained: **{selected_rows:,}**", f"- Expected window IDs: **{len(ids):,}**",
              f"- Window IDs found: **{len(seen):,}**", f"- Missing window IDs: **{len(missing):,}**",
              "", "Selection used exact `window_id` membership only. The source week was checked in every batch.", ""]
    if missing:
        report.extend(["## Missing IDs (first 20)", "", *[f"- `{value}`" for value in missing[:20]], ""])
    (args.output.parent / "primary_interaction_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output.parent / "primary_interaction_selection_config.json").write_text(
        json.dumps({**scope_config(), "source": str(args.input), "cohort": str(args.cohort), "output": str(args.output),
                    "selection_key": "window_id", "selected_rows": selected_rows, "missing_window_ids": len(missing)}, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 05 primary selection complete: rows={selected_rows:,} windows={len(seen):,} missing={len(missing):,}", flush=True)


if __name__ == "__main__":
    main()
