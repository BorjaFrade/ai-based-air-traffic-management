#!/usr/bin/env python3
"""
Resumable OpenSky Trino archive downloader for one study sector.

Purpose
-------
Download historical state vectors for one sector/area preset between two UTC
dates without generating PNG frames or video.

Design goals
------------
- One Trino query at a time (no parallelism).
- Every query is restricted to a single OpenSky `hour` partition.
- Large date ranges are split into small time windows.
- Geographic filtering is pushed to Trino through the sector bounding box.
- Polygon presets are clipped locally after download.
- Optional FL290-FL410 barometric-altitude filtering is pushed to Trino.
- Results are assembled as one CSV per calendar month under data/raw/opensky_sectors/<PRESET>/.
- Completed query windows are cached locally so interrupted runs can resume.
- pyOpenSky query caching is also enabled.
- Empty query windows are cached too, preventing pointless re-queries.

The script expects these files next to it:
    area_presets.py
    bbox_presets.json

Example
-------
python3 download_opensky_sector_archive.py \
    --sector LECMSAN \
    --start 2016-01-01 \
    --end 2026-01-01

Important
---------
OpenSky is a shared research resource. Read the current Trino performance
guidelines before running large downloads:
https://openskynetwork.github.io/opensky-api/trino.html#tools-performance-considerations

The end timestamp is EXCLUSIVE.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyopensky.schema import StateVectorsData4
from pyopensky.trino import Trino

from area_presets import (
    filter_dataframe_to_area,
    load_area_presets,
    resolve_area_preset,
)


# =============================================================================
# PROJECT / AUTHENTICATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / "credentials.env")

BBOX_PRESETS_FILE = Path(__file__).with_name("bbox_presets.json")

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "opensky_sectors"
)


# =============================================================================
# DEFAULT USER CONFIGURATION
# =============================================================================

DEFAULT_SECTOR = "GALICIA"

# UTC. End is exclusive.
DEFAULT_START_UTC = "2025-01-01 00:00:00"
DEFAULT_END_UTC = "2025-02-01 00:00:00"

# OpenSky state vectors can remain in the table briefly after coverage is lost.
MAX_LAST_CONTACT_AGE_SECONDS = 15

# The thesis cruise experiment currently focuses on RVSM FL290-FL410.
FILTER_FL_BAND = True
FL_MIN = 290
FL_MAX = 410

KEEP_ONLY_AIRBORNE = True

# Keep each request inside a single `hour` partition. 60 minutes is the natural
# first choice. If a sector/hour query becomes slow or hits a scan limit, reduce
# this to 30 or 15 minutes; every sub-query still remains inside one hour
# partition.
QUERY_WINDOW_MINUTES = 60

# A courtesy delay between sequential queries. This is NOT an official OpenSky
# quota; it simply avoids immediately firing the next request in a tight loop.
POLITE_DELAY_SECONDS = 2.0

# Retry transient failures sequentially. No parallel retry workers are used.
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 30.0

# OpenSky says queries taking more than five minutes should be made smaller.
SLOW_QUERY_WARNING_SECONDS = 300.0

# Optional guard for initial tests. Example: 24 = stop after one day of queried
# windows and resume on the next execution. None means no run-level cap.
MAX_WINDOWS_PER_RUN: int | None = None

# If True, remove the temporary per-window CSV cache after a month has been
# successfully assembled. Keeping it False gives the strongest resume/audit
# trail but consumes more disk.
DELETE_WINDOW_CACHE_AFTER_MONTH = False

# pyOpenSky local query cache.
PYOPENSKY_CACHED = True

TRINO_COLUMNS = (
    "time",
    "icao24",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "callsign",
    "onground",
    "alert",
    "spi",
    "squawk",
    "baroaltitude",
    "geoaltitude",
    "lastposupdate",
    "lastcontact",
    "hour",
)


# =============================================================================
# HELPERS
# =============================================================================

@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def key(self) -> str:
        return (
            f"{self.start.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{self.end.strftime('%Y%m%dT%H%M%SZ')}"
        )


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(
        year=ts.year,
        month=ts.month,
        day=1,
        tz="UTC",
    )


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.month == 12:
        return pd.Timestamp(
            year=ts.year + 1,
            month=1,
            day=1,
            tz="UTC",
        )
    return pd.Timestamp(
        year=ts.year,
        month=ts.month + 1,
        day=1,
        tz="UTC",
    )


def next_hour_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    floor = ts.floor("h")
    if ts == floor:
        return floor + pd.Timedelta(hours=1)
    return floor + pd.Timedelta(hours=1)


def build_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_minutes: int,
) -> list[Window]:
    """
    Split [start, end) into windows that never cross an OpenSky `hour`
    partition boundary.
    """
    if window_minutes <= 0 or window_minutes > 60:
        raise ValueError("QUERY_WINDOW_MINUTES must be between 1 and 60")

    step = pd.Timedelta(minutes=window_minutes)
    windows: list[Window] = []
    current = start

    while current < end:
        hour_end = next_hour_boundary(current)
        candidate_end = current + step
        window_end = min(candidate_end, hour_end, end)

        windows.append(Window(current, window_end))
        current = window_end

    return windows


def month_ranges(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    current_month = month_floor(start)

    while current_month < end:
        following_month = next_month(current_month)

        yield (
            max(start, current_month),
            min(end, following_month),
            current_month,
        )

        current_month = following_month


def print_authentication_mode() -> None:
    username = os.getenv("OPENSKY_USERNAME")
    password = os.getenv("OPENSKY_PASSWORD")

    if username and password:
        print("OpenSky authentication: credentials loaded from environment/.env")
    else:
        print(
            "OpenSky authentication: no username/password found in "
            "environment/.env; pyOpenSky may use browser authentication"
        )


def load_sector(sector_key: str) -> dict:
    presets = load_area_presets(BBOX_PRESETS_FILE)
    return resolve_area_preset(sector_key, presets)


def trino_filters():
    filters = [
        (
            StateVectorsData4.time
            - StateVectorsData4.lastcontact
        )
        <= MAX_LAST_CONTACT_AGE_SECONDS
    ]

    if KEEP_ONLY_AIRBORNE:
        filters.append(
            StateVectorsData4.onground == False
        )

    if FILTER_FL_BAND:
        lower_m = FL_MIN * 100.0 * 0.3048
        upper_m = FL_MAX * 100.0 * 0.3048

        filters.extend(
            [
                StateVectorsData4.baroaltitude >= lower_m,
                StateVectorsData4.baroaltitude <= upper_m,
            ]
        )

    return filters


def query_window(
    trino: Trino,
    area: dict,
    window: Window,
) -> pd.DataFrame:
    """
    Query one short window.

    The call is kept inside a single `hour` partition. pyOpenSky's history()
    method builds the partition-aware Trino request and splits requests by hour
    by default; here we additionally ensure each call itself never crosses an
    hour boundary.
    """
    bbox = area["bbox"]

    bounds = (
        bbox["lomin"],
        bbox["lamin"],
        bbox["lomax"],
        bbox["lamax"],
    )

    result = trino.history(
        window.start,
        window.end,
        *trino_filters(),
        bounds=bounds,
        selected_columns=TRINO_COLUMNS,
        cached=PYOPENSKY_CACHED,
        date_delta=pd.Timedelta(hours=1),
    )

    if result is None:
        result = pd.DataFrame(columns=TRINO_COLUMNS)
    else:
        result = result.copy()

    if result.empty:
        return pd.DataFrame(columns=TRINO_COLUMNS)

    # Defensive exact temporal clipping to [start, end), preventing duplicate
    # boundary rows when adjacent windows are concatenated.
    numeric_time = pd.to_numeric(
        result["time"],
        errors="coerce",
    )

    start_epoch = window.start.timestamp()
    end_epoch = window.end.timestamp()

    result = result.loc[
        (numeric_time >= start_epoch)
        & (numeric_time < end_epoch)
    ].copy()

    # Exact local clipping for polygon presets. Trino receives only the tight
    # enclosing BBOX; this removes points that are inside the BBOX but outside
    # the actual sector polygon.
    result = filter_dataframe_to_area(
        result,
        area,
        latitude_column="lat",
        longitude_column="lon",
    )

    result.drop_duplicates(inplace=True)
    return result


def query_with_retry(
    trino: Trino,
    area: dict,
    window: Window,
) -> tuple[pd.DataFrame, float]:
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.monotonic()

        try:
            dataframe = query_window(
                trino,
                area,
                window,
            )

            elapsed = time.monotonic() - started
            return dataframe, elapsed

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            elapsed = time.monotonic() - started

            print(
                f"ERROR on attempt {attempt}/{MAX_RETRIES} "
                f"after {elapsed:.1f}s: {exc}"
            )

            if attempt >= MAX_RETRIES:
                raise

            wait_seconds = (
                RETRY_BACKOFF_SECONDS
                * (2 ** (attempt - 1))
            )

            print(
                f"Waiting {wait_seconds:.0f}s before retry..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Unreachable retry state")


def monthly_paths(
    area_key: str,
    month: pd.Timestamp,
) -> dict[str, Path]:
    directory = (
        OUTPUT_ROOT
        / area_key
        / f"year={month.year:04d}"
        / f"month={month.month:02d}"
    )

    cache_dir = directory / "_window_cache"

    basename = (
        f"{area_key.lower()}_"
        f"{month.year:04d}-{month.month:02d}"
    )

    return {
        "directory": directory,
        "cache": cache_dir,
        "csv": directory / f"{basename}.csv",
        "manifest": directory / f"{basename}.manifest.json",
    }


def window_cache_path(
    cache_dir: Path,
    window: Window,
) -> Path:
    return cache_dir / f"window_{window.key}.csv"


def save_window_csv(
    path: Path,
    dataframe: pd.DataFrame,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(path.suffix + ".tmp")

    dataframe.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def assemble_month_csv(
    output_csv: Path,
    window_files: list[Path],
) -> tuple[int, int]:
    """
    Assemble cached window CSVs into one monthly CSV atomically.

    Returns:
        (row_count, file_count)
    """
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_output = output_csv.with_suffix(
        output_csv.suffix + ".tmp"
    )

    if tmp_output.exists():
        tmp_output.unlink()

    wrote_header = False
    row_count = 0

    with tmp_output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:

        for source_path in window_files:
            with source_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as source:

                header = source.readline()

                if not header:
                    continue

                if not wrote_header:
                    destination.write(header)
                    wrote_header = True

                for line in source:
                    destination.write(line)
                    row_count += 1

    if not wrote_header:
        pd.DataFrame(
            columns=TRINO_COLUMNS
        ).to_csv(
            tmp_output,
            index=False,
        )

    tmp_output.replace(output_csv)

    return row_count, len(window_files)


def write_manifest(
    path: Path,
    *,
    sector_key: str,
    area: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    windows: list[Window],
    row_count: int,
) -> None:
    payload = {
        "sector": sector_key,
        "sector_name": area["name"],
        "sector_type": area["type"],
        "start_utc": month_start.isoformat(),
        "end_utc_exclusive": month_end.isoformat(),
        "query_window_minutes": QUERY_WINDOW_MINUTES,
        "window_count": len(windows),
        "row_count": row_count,
        "filters": {
            "max_last_contact_age_seconds": MAX_LAST_CONTACT_AGE_SECONDS,
            "keep_only_airborne": KEEP_ONLY_AIRBORNE,
            "filter_fl_band": FILTER_FL_BAND,
            "fl_min": FL_MIN if FILTER_FL_BAND else None,
            "fl_max": FL_MAX if FILTER_FL_BAND else None,
            "polygon_local_clip": bool(area.get("polygon")),
        },
        "bbox": area["bbox"],
        "polygon": area.get("polygon"),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(path)


def process_month(
    trino: Trino,
    sector_key: str,
    area: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    *,
    remaining_window_budget: int | None,
) -> tuple[int, bool]:
    """
    Process one month range.

    Returns:
        (new_queries_performed, month_complete)
    """
    paths = monthly_paths(
        sector_key,
        month_floor(month_start),
    )

    windows = build_windows(
        month_start,
        month_end,
        QUERY_WINDOW_MINUTES,
    )

    # If final monthly CSV + manifest exist, the month is complete.
    if (
        paths["csv"].exists()
        and paths["manifest"].exists()
    ):
        print(
            f"SKIP completed month: "
            f"{paths['csv']}"
        )
        return 0, True

    paths["cache"].mkdir(
        parents=True,
        exist_ok=True,
    )

    queried = 0

    print()
    print("=" * 78)
    print(
        f"MONTH {month_start.strftime('%Y-%m')} | "
        f"{month_start} -> {month_end}"
    )
    print("=" * 78)

    for index, window in enumerate(
        windows,
        start=1,
    ):
        cache_file = window_cache_path(
            paths["cache"],
            window,
        )

        if cache_file.exists():
            print(
                f"[{index:04d}/{len(windows):04d}] "
                f"CACHED {window.start} -> {window.end}"
            )
            continue

        if (
            remaining_window_budget is not None
            and queried >= remaining_window_budget
        ):
            print(
                "Run-level query limit reached. "
                "Progress is safely cached; rerun the script to continue."
            )
            return queried, False

        print(
            f"[{index:04d}/{len(windows):04d}] "
            f"QUERY  {window.start} -> {window.end}"
        )

        dataframe, elapsed = query_with_retry(
            trino,
            area,
            window,
        )

        save_window_csv(
            cache_file,
            dataframe,
        )

        queried += 1

        print(
            f"    rows={len(dataframe):,} | "
            f"elapsed={elapsed:.1f}s | "
            f"saved={cache_file.name}"
        )

        if elapsed > SLOW_QUERY_WARNING_SECONDS:
            print()
            print("!" * 78)
            print(
                "WARNING: this query exceeded five minutes. "
                "OpenSky recommends reducing the requested time frame."
            )
            print(
                "Consider setting QUERY_WINDOW_MINUTES to 30 or 15 "
                "before continuing with similar periods."
            )
            print("!" * 78)
            print()

        if POLITE_DELAY_SECONDS > 0:
            time.sleep(POLITE_DELAY_SECONDS)

    expected_files = [
        window_cache_path(
            paths["cache"],
            window,
        )
        for window in windows
    ]

    missing = [
        path
        for path in expected_files
        if not path.exists()
    ]

    if missing:
        return queried, False

    row_count, file_count = assemble_month_csv(
        paths["csv"],
        expected_files,
    )

    write_manifest(
        paths["manifest"],
        sector_key=sector_key,
        area=area,
        month_start=month_start,
        month_end=month_end,
        windows=windows,
        row_count=row_count,
    )

    print()
    print(
        f"MONTH COMPLETE: {paths['csv']}"
    )
    print(
        f"  rows: {row_count:,}"
    )
    print(
        f"  cached windows: {file_count:,}"
    )

    if DELETE_WINDOW_CACHE_AFTER_MONTH:
        shutil.rmtree(
            paths["cache"],
            ignore_errors=True,
        )
        print(
            "  temporary window cache deleted"
        )

    return queried, True


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Download OpenSky Trino state vectors for one sector and "
            "store one resumable CSV per calendar month."
        )
    )

    parser.add_argument(
        "--sector",
        default=DEFAULT_SECTOR,
        help=(
            "Key from bbox_presets.json "
            f"(default: {DEFAULT_SECTOR})"
        ),
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START_UTC,
        help=(
            "UTC start timestamp, inclusive "
            f"(default: {DEFAULT_START_UTC})"
        ),
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END_UTC,
        help=(
            "UTC end timestamp, exclusive "
            f"(default: {DEFAULT_END_UTC})"
        ),
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=(
            0
            if MAX_WINDOWS_PER_RUN is None
            else MAX_WINDOWS_PER_RUN
        ),
        help=(
            "Maximum number of new Trino query windows in this run. "
            "0 means unlimited. Useful for cautious first tests."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    start = parse_utc(args.start)
    end = parse_utc(args.end)

    if end <= start:
        raise ValueError("--end must be later than --start")

    sector_key = args.sector
    area = load_sector(sector_key)

    max_windows = (
        None
        if args.max_windows == 0
        else args.max_windows
    )

    print()
    print("=" * 78)
    print("OpenSky sector archive downloader")
    print("=" * 78)
    print_authentication_mode()
    print(f"Sector: {sector_key} — {area['name']}")
    print(f"Area type: {area['type']}")
    print(f"Start UTC: {start}")
    print(f"End UTC (exclusive): {end}")
    print(
        f"Query window: {QUERY_WINDOW_MINUTES} min "
        "(never crosses an hour partition)"
    )
    print("Concurrent Trino queries: 1")
    print(
        f"FL filter: "
        + (
            f"FL{FL_MIN}-FL{FL_MAX}"
            if FILTER_FL_BAND
            else "disabled"
        )
    )
    print(
        f"Polygon local clipping: "
        f"{'yes' if area.get('polygon') else 'no'}"
    )
    print(
        f"Run-level new-query limit: "
        f"{max_windows if max_windows is not None else 'unlimited'}"
    )
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 78)
    print()

    trino = Trino()

    new_queries_total = 0

    for (
        month_start,
        month_end,
        _month_identity,
    ) in month_ranges(
        start,
        end,
    ):
        if (
            max_windows is not None
            and new_queries_total >= max_windows
        ):
            print(
                "Run-level query limit reached before the next month. "
                "Rerun the same command to resume."
            )
            break

        remaining = (
            None
            if max_windows is None
            else max_windows - new_queries_total
        )

        queried, complete = process_month(
            trino,
            sector_key,
            area,
            month_start,
            month_end,
            remaining_window_budget=remaining,
        )

        new_queries_total += queried

        if not complete:
            break

    print()
    print("=" * 78)
    print("RUN FINISHED")
    print(f"New Trino queries executed: {new_queries_total:,}")
    print(
        "The downloader is resumable: rerunning the same command skips "
        "completed windows/months."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
