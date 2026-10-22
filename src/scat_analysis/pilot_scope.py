"""Shared hard guard for the SCAT first-week methodological pilot."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PILOT_WEEK = "20161015_20161021"
ANALYSIS_SCOPE = "FIRST_WEEK_PILOT"
SCOPE_ERROR = f"ERROR: PILOT MODE ONLY ALLOWS WEEK {PILOT_WEEK}"


def assert_pilot_weeks(values) -> None:
    weeks = {str(value) for value in values if value is not None}
    if weeks != {PILOT_WEEK}:
        raise SystemExit(f"{SCOPE_ERROR}; found {sorted(weeks)}")


def assert_pilot_frame(frame) -> None:
    if "archive_week" not in frame:
        raise SystemExit(f"{SCOPE_ERROR}; archive_week is missing")
    assert_pilot_weeks(frame["archive_week"].dropna().unique())
    if frame["archive_week"].isna().any():
        raise SystemExit(f"{SCOPE_ERROR}; null archive_week found")


def assert_pilot_parquet(path) -> None:
    parquet = pq.ParquetFile(path)
    if "archive_week" not in parquet.schema.names:
        raise SystemExit(f"{SCOPE_ERROR}; archive_week is missing")
    seen = set()
    for batch in parquet.iter_batches(columns=["archive_week"], batch_size=250_000):
        column = batch.column(0)
        if column.null_count:
            raise SystemExit(f"{SCOPE_ERROR}; null archive_week found")
        seen.update(pc.unique(column).to_pylist())
        if seen - {PILOT_WEEK}:
            raise SystemExit(f"{SCOPE_ERROR}; found {sorted(seen)}")
    assert_pilot_weeks(seen)


def scope_config() -> dict[str, str]:
    return {"analysis_scope": ANALYSIS_SCOPE, "archive_week": PILOT_WEEK}
