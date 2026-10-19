#!/usr/bin/env python3
"""
Detect SCAT separation events while preserving flight-plan and ATC-clearance
context.

The geometric/safety calculation is the same research model used by the
OpenSky detector.  The input adapter is different: SCAT has already been
normalised by preprocess_scat_10s.py, and every row contains the latest FPL,
clearance, centre-control and predicted-route values available at that
surveillance time.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
ACQUISITION_DIR = SRC_DIR / "acquisition"

if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import detect_opensky_conflicts as opensky_detector  # noqa: E402
from scat_units import (  # noqa: E402
    DEFAULT_ANALYSIS_FL_MAX,
    DEFAULT_ANALYSIS_FL_MIN,
    filter_flight_level_band,
)


INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "scat_10s"
EVENT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "scat_separation"

READ_CHUNK_ROWS = 250_000
FILE_RE = re.compile(r"^scat_(?P<date>\d{4}-\d{2}-\d{2})_10s\.csv$")

CONTEXT_FIELDS = (
    "scat_flight_id",
    "fpl_adep",
    "fpl_ades",
    "fpl_aircraft_type",
    "fpl_wtc",
    "fpl_flight_rules",
    "fpl_route",
    "fpl_rfl",
    "fpl_tas",
    "fpl_copn",
    "fpl_copn_pel",
    "fpl_copx",
    "fpl_copx_pel",
    "fpl_update_timestamp",
    "clearance_cfl",
    "clearance_cfl_unit",
    "clearance_heading_deg",
    "clearance_heading_beacon",
    "clearance_speed",
    "clearance_speed_unit",
    "clearance_timestamp",
    "active_centre_id",
    "centre_since",
    "predicted_route_timestamp",
    "predicted_next_fix",
    "predicted_next_fix_eta",
    "predicted_route",
    "source_plot_time",
    "source_plot_offset_s",
)

SCAT_EVENT_COLUMNS = list(opensky_detector.EVENT_COLUMNS)
for suffix in ("a", "b"):
    SCAT_EVENT_COLUMNS.extend(f"{field}_{suffix}" for field in CONTEXT_FIELDS)


def parse_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def discover_files(area: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    directory = INPUT_ROOT / area
    if not directory.exists():
        raise FileNotFoundError(f"Missing SCAT processed directory: {directory}")
    paths = []
    for path in directory.rglob("*_10s.csv"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        day = pd.Timestamp(match.group("date"), tz="UTC")
        if start.normalize() <= day <= end.normalize():
            paths.append(path)
    paths.sort()
    if not paths:
        raise FileNotFoundError(
            f"No SCAT 10-second files found for {area} between {start} and {end}"
        )
    return paths


def iter_snapshots(
    files: list[Path],
    start: float,
    end: float,
    chunksize: int,
) -> Iterator[tuple[float, pd.DataFrame]]:
    previous_timestamp = float("-inf")
    for path in files:
        carry = pd.DataFrame()
        for raw in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            if "time" not in raw.columns or "icao24" not in raw.columns:
                raise ValueError(f"Missing time/icao24 fields in {path}")
            raw = raw.copy()
            raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
            raw = raw.loc[raw["time"].notna()].copy()
            if not carry.empty:
                raw = pd.concat([carry, raw], ignore_index=True)
            if raw.empty:
                carry = pd.DataFrame()
                continue
            raw.sort_values(["time", "icao24"], inplace=True, kind="mergesort")
            end_of_chunk = float(raw["time"].max())
            complete = raw.loc[raw["time"] < end_of_chunk]
            carry = raw.loc[raw["time"] == end_of_chunk].copy()
            for stamp, snapshot in complete.groupby("time", sort=True):
                stamp = float(stamp)
                if stamp < previous_timestamp:
                    raise ValueError(f"Processed SCAT input is not chronological: {path}")
                previous_timestamp = stamp
                if start <= stamp <= end:
                    yield stamp, opensky_detector.snapshot_unique(snapshot)
        for stamp, snapshot in carry.groupby("time", sort=True):
            stamp = float(stamp)
            if stamp < previous_timestamp:
                raise ValueError(f"Processed SCAT input is not chronological: {path}")
            previous_timestamp = stamp
            if start <= stamp <= end:
                yield stamp, opensky_detector.snapshot_unique(snapshot)


def value_or_empty(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def build_scat_event_row(
    area: str,
    timestamp: float,
    snapshot: pd.DataFrame,
    event: dict,
) -> dict:
    row = opensky_detector.build_event_row(area, timestamp, snapshot, event)
    for suffix, index_key in (("a", "index_a"), ("b", "index_b")):
        aircraft = snapshot.loc[event[index_key]]
        for field in CONTEXT_FIELDS:
            row[f"{field}_{suffix}"] = value_or_empty(aircraft.get(field, ""))
    row["dataset"] = "SCAT"
    return row


def output_paths(area: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    run_name = f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"
    directory = EVENT_OUTPUT_ROOT / area / "runs" / run_name
    return {
        "directory": directory,
        opensky_detector.STATE_PROXIMITY: directory / "yellow_proximity_events.csv",
        opensky_detector.STATE_HIGH_RISK: directory / "orange_high_risk_events.csv",
        opensky_detector.STATE_CONFLICT: directory / "red_loss_of_separation_events.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect SCAT conflicts and export FPL/ATC-clearance context."
    )
    parser.add_argument("--area", required=True, help="Processed SCAT area key")
    parser.add_argument("--start", required=True, help="UTC inclusive")
    parser.add_argument("--end", required=True, help="UTC inclusive")
    parser.add_argument("--chunksize", type=int, default=READ_CHUNK_ROWS)
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="0 means all snapshots; useful for a smoke test",
    )
    parser.add_argument(
        "--min-fl",
        type=float,
        default=DEFAULT_ANALYSIS_FL_MIN,
        help=f"Minimum analysed flight level (default: FL{DEFAULT_ANALYSIS_FL_MIN:g})",
    )
    parser.add_argument(
        "--max-fl",
        type=float,
        default=DEFAULT_ANALYSIS_FL_MAX,
        help=f"Maximum analysed flight level (default: FL{DEFAULT_ANALYSIS_FL_MAX:g})",
    )
    parser.add_argument(
        "--all-flight-levels",
        action="store_true",
        help="Disable the default FL290-FL410 first-pass filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if start > end or args.chunksize <= 0 or args.max_snapshots < 0:
        raise SystemExit("Invalid date range, chunksize or max-snapshots")
    min_fl = None if args.all_flight_levels else args.min_fl
    max_fl = None if args.all_flight_levels else args.max_fl
    if min_fl is not None and (min_fl < 0 or max_fl is None or min_fl > max_fl):
        raise SystemExit("Invalid flight-level analysis band")

    files = discover_files(args.area, start, end)
    paths = output_paths(args.area, start, end)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}
    writers: dict[str, csv.DictWriter] = {}
    totals = {
        "dataset": "SCAT",
        "area": args.area,
        "start_utc": start.isoformat(),
        "end_utc_inclusive": end.isoformat(),
        "snapshots": 0,
        "pairs_total": 0,
        "pairs_distance_pruned": 0,
        "pairs_cpa_pruned": 0,
        "pairs_detailed": 0,
        "aircraft_rows_before_fl_filter": 0,
        "aircraft_rows_after_fl_filter": 0,
        "analysis_fl_min": "" if min_fl is None else min_fl,
        "analysis_fl_max": "" if max_fl is None else max_fl,
        opensky_detector.STATE_PROXIMITY: 0,
        opensky_detector.STATE_HIGH_RISK: 0,
        opensky_detector.STATE_CONFLICT: 0,
    }
    daily: dict[str, dict] = {}
    started = time.monotonic()

    print("=" * 78)
    print("SCAT separation detector")
    print(f"Area: {args.area} | Start UTC: {start} | End UTC: {end}")
    print(f"Input files: {len(files)} | Output: {paths['directory']}")
    if min_fl is None:
        analysis_band = "all flight levels"
    else:
        analysis_band = f"FL{min_fl:g}-FL{max_fl:g}"
    print(
        "Safety model: shared ENROUTE profile; "
        f"analysis band: {analysis_band}; look-ahead: 1, 2 and 5 minutes"
    )
    print("=" * 78, flush=True)

    try:
        for state in opensky_detector.ACTIVE_EVENT_STATES:
            handle = paths[state].open("w", encoding="utf-8", newline="")
            handles[state] = handle
            writers[state] = csv.DictWriter(
                handle,
                fieldnames=SCAT_EVENT_COLUMNS,
                extrasaction="ignore",
            )
            writers[state].writeheader()

        for timestamp, snapshot in iter_snapshots(
            files, start.timestamp(), end.timestamp(), args.chunksize
        ):
            if args.max_snapshots and totals["snapshots"] >= args.max_snapshots:
                break
            totals["snapshots"] += 1
            day = pd.to_datetime(timestamp, unit="s", utc=True).strftime("%Y-%m-%d")
            daily.setdefault(
                day,
                {"day": day, "yellow": 0, "orange": 0, "red": 0, "snapshots": 0},
            )["snapshots"] += 1
            totals["aircraft_rows_before_fl_filter"] += len(snapshot)
            snapshot = filter_flight_level_band(snapshot, min_fl, max_fl)
            snapshot.reset_index(drop=True, inplace=True)
            totals["aircraft_rows_after_fl_filter"] += len(snapshot)
            if len(snapshot) < 2:
                continue

            events, stats = opensky_detector.analyse_snapshot_fast(snapshot)
            for key in (
                "pairs_total",
                "pairs_distance_pruned",
                "pairs_cpa_pruned",
                "pairs_detailed",
            ):
                totals[key] += stats[key]
            for event in events:
                state = event["event_state"]
                if state not in writers:
                    continue
                writers[state].writerow(
                    build_scat_event_row(args.area, timestamp, snapshot, event)
                )
                totals[state] += 1
                colour = opensky_detector.STATE_COLOURS[state].lower()
                daily[day][colour] += 1
            if totals["snapshots"] % 1000 == 0:
                print(
                    f"snapshots={totals['snapshots']:,} | "
                    f"pairs={totals['pairs_total']:,} | "
                    f"yellow={totals[opensky_detector.STATE_PROXIMITY]:,} | "
                    f"orange={totals[opensky_detector.STATE_HIGH_RISK]:,} | "
                    f"red={totals[opensky_detector.STATE_CONFLICT]:,}",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    elapsed = time.monotonic() - started
    totals["elapsed_s"] = elapsed
    pd.DataFrame([totals]).to_csv(paths["directory"] / "run_summary.csv", index=False)
    pd.DataFrame(sorted(daily.values(), key=lambda row: row["day"])).to_csv(
        paths["directory"] / "daily_summary.csv", index=False
    )
    print("=" * 78)
    print(
        f"COMPLETE | snapshots={totals['snapshots']:,} | "
        f"yellow={totals[opensky_detector.STATE_PROXIMITY]:,} | "
        f"orange={totals[opensky_detector.STATE_HIGH_RISK]:,} | "
        f"red={totals[opensky_detector.STATE_CONFLICT]:,}"
    )
    print(f"Output: {paths['directory']}")


if __name__ == "__main__":
    main()
