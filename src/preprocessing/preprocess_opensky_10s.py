#!/usr/bin/env python3
"""
Create reusable 10-second UTC aircraft-state CSVs from monthly OpenSky Trino
archives.  This preprocessor does not perform conflict detection.

Keep this file in project_root/src/preprocessing/. Dependencies (area_presets
and bbox_presets.json) are located in project_root/src/acquisition/.

Raw archives are not modified. For every aircraft and 10-second grid instant,
use a genuine nearby lastposupdate, or interpolate/extrapolate briefly.
Filter the reconstructed state by the exact LECMSAN polygon and FL290-FL410.
Output: data/processed/opensky_10s/<AREA>/year=YYYY/month=MM/*_10s.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from pyproj import Geod


# =============================================================================
# LOCAL DEPENDENCIES
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ACQUISITION_DIR = SRC_DIR / "acquisition"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if not ACQUISITION_DIR.exists():
    raise FileNotFoundError(
        "OpenSky acquisition dependency directory not found:\n"
        f"{ACQUISITION_DIR}"
    )

if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))

from area_presets import (  # noqa: E402
    filter_dataframe_to_area,
    load_area_presets,
    resolve_area_preset,
)

# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "opensky_sectors"
SNAPSHOT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "opensky_10s"

BBOX_PRESETS_FILE = ACQUISITION_DIR / "bbox_presets.json"

DEFAULT_SECTOR = "LECMSAN_ENVELOPE"
DEFAULT_ANALYSIS_AREA = "LECMSAN"

# Observation grid.
SNAPSHOT_STEP_SECONDS = 10

# Prediction grid.
LOOKAHEAD_MINUTES = (1, 2, 5)
LOOKAHEAD_SECONDS = int(max(LOOKAHEAD_MINUTES) * 60)
PREDICTION_STEP_SECONDS = 5

# Reconstruction safeguards.
MAX_INTERPOLATION_GAP_SECONDS = 20.0
MAX_EXTRAPOLATION_SECONDS = 10.0
MAX_STATE_AGE_SECONDS = 15.0
EXACT_POSITION_TOLERANCE_SECONDS = 0.75

# Cruise experiment.
FILTER_CRUISE_FL_BAND = True
CRUISE_FL_MIN = 290
CRUISE_FL_MAX = 410
CHECK_GROUND_AIRCRAFT = False

# Input streaming.
READ_CHUNK_ROWS = 250_000
REQUIRE_CHRONOLOGICAL_INPUT = True

# Pair-screening switches.
ENABLE_DISTANCE_PREFILTER = True
ENABLE_CPA_PREFILTER = True
CPA_PREFILTER_MARGIN_NM = 1.0

# Save regularised 10-second traffic states by default.
SAVE_REGULARIZED_SNAPSHOTS = True

MPS_TO_KT = 1.9438444924406
MPS_TO_FPM = 196.8503937008
GEOD = Geod(ellps="WGS84")

# =============================================================================
# GENERIC HELPERS
# =============================================================================

MONTHLY_FILE_RE = re.compile(r"^(?P<name>.+)_(?P<year>\d{4})-(?P<month>\d{2})\.csv$")


def parse_day_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return numeric
        magnitude = float(finite.abs().median())
        if magnitude > 1e17:
            return numeric / 1_000_000_000.0
        if magnitude > 1e14:
            return numeric / 1_000_000.0
        if magnitude > 1e11:
            return numeric / 1_000.0
        return numeric

    values = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return values.sub(epoch).dt.total_seconds().astype("float64")


def row_numeric(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = row.get(name, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def ground_speed_ms(row: pd.Series) -> float | None:
    value = row_numeric(row, "velocity")
    if value is not None:
        return value
    value_kt = row_numeric(row, "velocity_kts")
    return None if value_kt is None else value_kt / MPS_TO_KT


def track_deg(row: pd.Series) -> float | None:
    return row_numeric(row, "true_track", "heading")


def vertical_rate_ms(row: pd.Series) -> float | None:
    value = row_numeric(row, "vertical_rate", "vertrate")
    if value is not None:
        return value
    value_fpm = row_numeric(row, "vertical_rate_fpm")
    return None if value_fpm is None else value_fpm / MPS_TO_FPM


def altitude_ft(row: pd.Series) -> float | None:
    value = row_numeric(row, "baroaltitude")
    return None if value is None else value * 3.2808398950131


def callsign_of(row: pd.Series) -> str:
    value = str(row.get("callsign", "") or "").strip()
    return value if value else str(row.get("icao24", "") or "").upper()


def is_true_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def load_analysis_area(area_key: str | None) -> dict | None:
    if not area_key:
        return None
    presets = load_area_presets(BBOX_PRESETS_FILE)
    return resolve_area_preset(area_key, presets)


def discover_monthly_csvs(
    sector: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> list[tuple[Path, int, int]]:
    sector_root = INPUT_ROOT / sector
    if not sector_root.exists():
        raise FileNotFoundError(f"Sector archive folder not found:\n{sector_root}")

    start_value = start_day.year * 12 + start_day.month
    end_value = end_day.year * 12 + end_day.month
    output = []

    for path in sector_root.rglob("*.csv"):
        if "_window_cache" in path.parts:
            continue
        match = MONTHLY_FILE_RE.match(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        month = int(match.group("month"))
        value = year * 12 + month
        if start_value <= value <= end_value:
            output.append((path, year, month))

    output.sort(key=lambda item: (item[1], item[2], str(item[0])))
    return output


# =============================================================================
# RAW STREAMING BY DAY
# =============================================================================


def normalize_raw_chunk(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.copy()
    required = {"time", "icao24", "lat", "lon"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    dataframe["time"] = to_epoch_seconds(dataframe["time"])

    for column in (
        "lat", "lon", "velocity", "heading", "vertrate", "baroaltitude",
        "geoaltitude", "lastposupdate", "lastcontact",
    ):
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe["icao24"] = dataframe["icao24"].astype(str).str.strip().str.lower()
    valid = (
        dataframe["time"].notna()
        & np.isfinite(dataframe["time"])
        & dataframe["icao24"].ne("")
    )
    dataframe = dataframe.loc[valid].copy()
    dataframe.sort_values(["time", "icao24"], inplace=True, kind="mergesort")
    return dataframe


def iter_days_with_context(
    csv_file: Path,
    chunksize: int,
    requested_start: pd.Timestamp,
    requested_end_exclusive: pd.Timestamp,
    context_seconds: float,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """
    Stream the monthly CSV and yield complete UTC days with a small context
    margin from the previous and next day.

    This relies on the archive downloader writing hourly query windows in
    chronological order. Rows inside an individual hour may be unsorted.
    """
    broad_start = float((requested_start - pd.Timedelta(seconds=context_seconds)).timestamp())
    broad_end = float((requested_end_exclusive + pd.Timedelta(seconds=context_seconds)).timestamp())

    buffers: dict[int, list[pd.DataFrame]] = {}
    previous_tail = pd.DataFrame()
    max_time_seen = -np.inf
    last_day_seen: int | None = None

    def emit_ready_days(force: bool = False):
        nonlocal previous_tail
        while buffers:
            oldest_day = min(buffers)
            day_end = (oldest_day + 1) * 86400.0
            if not force and max_time_seen < day_end + context_seconds:
                break

            day_frame = pd.concat(buffers[oldest_day], ignore_index=True)
            day_frame.sort_values(["time", "icao24"], inplace=True, kind="mergesort")

            next_parts = buffers.get(oldest_day + 1, [])
            if next_parts:
                next_frame = pd.concat(next_parts, ignore_index=True)
                next_head = next_frame.loc[
                    next_frame["time"] < day_end + context_seconds
                ].copy()
            else:
                next_head = pd.DataFrame()

            context_frame = pd.concat(
                [previous_tail, day_frame, next_head],
                ignore_index=True,
            )
            context_frame.sort_values(["time", "icao24"], inplace=True, kind="mergesort")

            previous_tail = day_frame.loc[
                day_frame["time"] >= day_end - context_seconds
            ].copy()

            del buffers[oldest_day]
            yield (
                pd.Timestamp(oldest_day * 86400.0, unit="s", tz="UTC"),
                context_frame,
            )

    for raw_chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
        chunk = normalize_raw_chunk(raw_chunk)
        if chunk.empty:
            continue

        raw_min = float(chunk["time"].min())
        raw_max = float(chunk["time"].max())

        if raw_min >= broad_end:
            break

        chunk = chunk.loc[
            (chunk["time"] >= broad_start) & (chunk["time"] < broad_end)
        ].copy()

        if chunk.empty:
            continue

        max_time_seen = max(max_time_seen, float(chunk["time"].max()))
        day_keys = np.floor(chunk["time"].to_numpy(dtype=float) / 86400.0).astype("int64")

        for day_key in np.unique(day_keys):
            day_key = int(day_key)
            if REQUIRE_CHRONOLOGICAL_INPUT and last_day_seen is not None and day_key < last_day_seen - 1:
                raise ValueError("The monthly archive moved backwards in time.")
            part = chunk.loc[day_keys == day_key].copy()
            buffers.setdefault(day_key, []).append(part)
            last_day_seen = day_key if last_day_seen is None else max(last_day_seen, day_key)

        for item in emit_ready_days(force=False):
            yield item

        if raw_max >= broad_end:
            break

    for item in emit_ready_days(force=True):
        yield item


# =============================================================================
# 10-SECOND REGULARISATION
# =============================================================================


def geodesic_interpolate(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    alpha: float,
) -> tuple[float, float]:
    alpha = min(max(float(alpha), 0.0), 1.0)
    azimuth, _, distance_m = GEOD.inv(lon1, lat1, lon2, lat2)
    lon, lat, _ = GEOD.fwd(lon1, lat1, azimuth, distance_m * alpha)
    return float(lat), float(lon)


def geodesic_extrapolate(
    lat: float,
    lon: float,
    speed_ms: float,
    track: float,
    dt_seconds: float,
) -> tuple[float, float]:
    lon_out, lat_out, _ = GEOD.fwd(
        lon,
        lat,
        track,
        speed_ms * dt_seconds,
    )
    return float(lat_out), float(lon_out)


def regularize_aircraft_to_grid(
    aircraft_rows: pd.DataFrame,
    day_start_epoch: float,
    day_end_epoch: float,
) -> list[dict]:
    if aircraft_rows.empty:
        return []

    rows = aircraft_rows.sort_values("time", kind="mergesort").copy()
    row_times = rows["time"].to_numpy(dtype=float)
    if row_times.size == 0:
        return []

    first_target = (
        math.ceil(max(day_start_epoch, float(row_times.min())) / SNAPSHOT_STEP_SECONDS)
        * SNAPSHOT_STEP_SECONDS
    )
    last_target = (
        math.floor(
            min(day_end_epoch - 1e-6, float(row_times.max()) + MAX_EXTRAPOLATION_SECONDS)
            / SNAPSHOT_STEP_SECONDS
        )
        * SNAPSHOT_STEP_SECONDS
    )
    if last_target < first_target:
        return []

    targets = np.arange(
        first_target,
        last_target + SNAPSHOT_STEP_SECONDS * 0.5,
        SNAPSHOT_STEP_SECONDS,
        dtype=float,
    )

    pos_rows = rows.loc[
        rows["lat"].notna()
        & rows["lon"].notna()
        & np.isfinite(rows["lat"])
        & np.isfinite(rows["lon"])
    ].copy()
    if pos_rows.empty:
        return []

    if "lastposupdate" in pos_rows.columns:
        pos_rows["_position_time"] = pd.to_numeric(
            pos_rows["lastposupdate"], errors="coerce"
        )
    else:
        pos_rows["_position_time"] = np.nan

    pos_rows["_position_time"] = pos_rows["_position_time"].where(
        pos_rows["_position_time"].notna(),
        pos_rows["time"],
    )
    pos_rows = pos_rows.loc[np.isfinite(pos_rows["_position_time"])].copy()
    if pos_rows.empty:
        return []

    pos_rows.sort_values(["_position_time", "time"], inplace=True, kind="mergesort")
    pos_rows.drop_duplicates(subset=["_position_time"], keep="last", inplace=True)

    pos_times = pos_rows["_position_time"].to_numpy(dtype=float)
    pos_lats = pos_rows["lat"].to_numpy(dtype=float)
    pos_lons = pos_rows["lon"].to_numpy(dtype=float)

    exact_grid_times = set(
        np.rint(
            row_times[
                np.isclose(
                    np.mod(row_times, SNAPSHOT_STEP_SECONDS),
                    0.0,
                    atol=1e-6,
                )
            ]
        ).astype("int64")
    )

    output = []

    for target in targets:
        state_idx = np.searchsorted(row_times, target, side="right") - 1
        if state_idx < 0:
            continue

        state_row = rows.iloc[int(state_idx)]
        state_age = target - float(state_row["time"])
        if state_age < -1e-6 or state_age > MAX_STATE_AGE_SECONDS:
            continue

        if not CHECK_GROUND_AIRCRAFT and is_true_value(state_row.get("onground", False)):
            continue

        prev_idx = np.searchsorted(pos_times, target, side="right") - 1
        next_idx = prev_idx + 1

        lat = None
        lon = None
        source = None
        position_age = None

        # 1) True position update essentially at the requested grid instant.
        if prev_idx >= 0:
            prev_time = float(pos_times[prev_idx])
            position_age = target - prev_time
            if abs(target - prev_time) <= EXACT_POSITION_TOLERANCE_SECONDS:
                lat = float(pos_lats[prev_idx])
                lon = float(pos_lons[prev_idx])
                source = "REAL_POSITION"

        # 2) Interpolate only between two genuine position updates that are
        # sufficiently close in time.
        if lat is None and prev_idx >= 0 and next_idx < len(pos_times):
            prev_time = float(pos_times[prev_idx])
            next_time = float(pos_times[next_idx])
            gap = next_time - prev_time
            if (
                prev_time <= target <= next_time
                and gap > 0
                and gap <= MAX_INTERPOLATION_GAP_SECONDS
            ):
                alpha = (target - prev_time) / gap
                lat, lon = geodesic_interpolate(
                    float(pos_lats[prev_idx]),
                    float(pos_lons[prev_idx]),
                    float(pos_lats[next_idx]),
                    float(pos_lons[next_idx]),
                    alpha,
                )
                source = "INTERPOLATED_POSITION"
                position_age = target - prev_time

        # 3) Short forward extrapolation from the last genuine position update.
        if lat is None and prev_idx >= 0:
            prev_time = float(pos_times[prev_idx])
            dt = target - prev_time
            speed_ms = ground_speed_ms(state_row)
            track = track_deg(state_row)
            if (
                0.0 <= dt <= MAX_EXTRAPOLATION_SECONDS
                and speed_ms is not None
                and track is not None
            ):
                lat, lon = geodesic_extrapolate(
                    float(pos_lats[prev_idx]),
                    float(pos_lons[prev_idx]),
                    float(speed_ms),
                    float(track),
                    float(dt),
                )
                source = "EXTRAPOLATED_POSITION"
                position_age = dt

        if lat is None:
            continue

        record = {
            column: state_row.get(column, np.nan)
            for column in rows.columns
            if not column.startswith("_")
        }
        record["time"] = float(target)
        record["lat"] = lat
        record["lon"] = lon
        record["state_source"] = source
        record["exact_grid_row"] = int(round(target)) in exact_grid_times
        record["state_age_s"] = float(state_age)
        record["position_age_s"] = None if position_age is None else float(position_age)
        output.append(record)

    return output


def regularize_day(
    raw_day_with_context: pd.DataFrame,
    day_start: pd.Timestamp,
    analysis_area: dict | None,
) -> tuple[pd.DataFrame, dict]:
    started = time.monotonic()
    day_start_epoch = float(day_start.timestamp())
    day_end_epoch = day_start_epoch + 86400.0

    main_day = raw_day_with_context.loc[
        (raw_day_with_context["time"] >= day_start_epoch)
        & (raw_day_with_context["time"] < day_end_epoch)
    ].copy()

    raw_rows = len(main_day)
    if raw_rows:
        raw_times = main_day["time"].to_numpy(dtype=float)
        rows_on_grid = int(
            np.count_nonzero(
                np.isclose(
                    np.mod(raw_times, SNAPSHOT_STEP_SECONDS),
                    0.0,
                    atol=1e-6,
                )
            )
        )
    else:
        rows_on_grid = 0

    records = []
    for _, aircraft_rows in raw_day_with_context.groupby("icao24", sort=False):
        records.extend(
            regularize_aircraft_to_grid(
                aircraft_rows,
                day_start_epoch,
                day_end_epoch,
            )
        )

    regularized = pd.DataFrame(records)

    if not regularized.empty and analysis_area is not None:
        regularized = filter_dataframe_to_area(
            regularized,
            analysis_area,
            latitude_column="lat",
            longitude_column="lon",
        )

    if not regularized.empty and FILTER_CRUISE_FL_BAND:
        lower_m = CRUISE_FL_MIN * 100.0 * 0.3048
        upper_m = CRUISE_FL_MAX * 100.0 * 0.3048
        baro = pd.to_numeric(regularized.get("baroaltitude"), errors="coerce")
        regularized = regularized.loc[
            baro.between(lower_m, upper_m, inclusive="both")
        ].copy()

    if not regularized.empty:
        regularized.sort_values(["time", "icao24"], inplace=True, kind="mergesort")
        regularized.drop_duplicates(
            subset=["time", "icao24"],
            keep="last",
            inplace=True,
        )

    source_counts = (
        regularized["state_source"].value_counts()
        if not regularized.empty and "state_source" in regularized.columns
        else pd.Series(dtype="int64")
    )

    stats = {
        "raw_rows": raw_rows,
        "raw_aircraft": int(main_day["icao24"].nunique()) if raw_rows else 0,
        "rows_on_10s_grid": rows_on_grid,
        "regularized_states": int(len(regularized)),
        "snapshots": int(regularized["time"].nunique()) if not regularized.empty else 0,
        "real_positions": int(source_counts.get("REAL_POSITION", 0)),
        "interpolated_positions": int(source_counts.get("INTERPOLATED_POSITION", 0)),
        "extrapolated_positions": int(source_counts.get("EXTRAPOLATED_POSITION", 0)),
        "elapsed_s": time.monotonic() - started,
    }
    return regularized, stats



# =============================================================================
# MONTHLY OUTPUT AND RESUMABLE DAILY CACHE
# =============================================================================

PREPROCESS_VERSION = "10s_v1"


def output_paths(area_key: str, year: int, month: int) -> dict[str, Path]:
    directory = (SNAPSHOT_OUTPUT_ROOT / area_key /
                 f"year={year:04d}" / f"month={month:02d}")
    name = f"{area_key.lower()}_{year:04d}-{month:02d}_10s"
    return {
        "directory": directory,
        "cache": directory / "_daily_cache",
        "csv": directory / f"{name}.csv",
        "manifest": directory / f"{name}.manifest.json",
    }


def daily_cache_file(paths: dict, area_key: str, day: pd.Timestamp) -> Path:
    return paths["cache"] / f"{area_key.lower()}_{day.strftime('%Y-%m-%d')}_10s.csv"


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    if month == 12:
        last = pd.Timestamp(year=year+1, month=1, day=1, tz="UTC")
    else:
        last = pd.Timestamp(year=year, month=month+1, day=1, tz="UTC")
    return first, last


def ensure_columns(df: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(input_columns + [
        "timestamp_utc", "state_source", "exact_grid_row",
        "state_age_s", "position_age_s",
    ]))
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    for name in columns:
        if name not in df.columns:
            df[name] = pd.NA
    return df[columns]


def atomic_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(tmp, index=False)
    tmp.replace(path)


def assemble_month(paths: dict, day_files: list[Path]) -> int:
    tmp = paths["csv"].with_suffix(".csv.tmp")
    rows = 0
    with tmp.open("w", encoding="utf-8", newline="") as dst:
        first_header = None
        for file in day_files:
            with file.open("r", encoding="utf-8", newline="") as src:
                header = src.readline()
                if not header:
                    raise ValueError(f"Empty or corrupt daily cache: {file}")
                if first_header is None:
                    first_header = header
                    dst.write(header)
                elif header != first_header:
                    raise ValueError(f"Incompatible daily headers: {file}")
                for line in src:
                    dst.write(line)
                    rows += 1
    tmp.replace(paths["csv"])
    return rows


def process_month(
    input_csv: Path,
    year: int,
    month: int,
    source_sector: str,
    area_key: str,
    area: dict,
    chunksize: int,
    overwrite: bool,
    keep_cache: bool,
) -> None:
    paths = output_paths(area_key, year, month)
    if (paths["csv"].exists() and paths["manifest"].exists()
            and not overwrite):
        print(f"SKIP completed month: {paths['csv']}", flush=True)
        return
    if overwrite:
        paths["csv"].unlink(missing_ok=True)
        paths["manifest"].unlink(missing_ok=True)
        shutil.rmtree(paths["cache"], ignore_errors=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)

    first, last = month_bounds(year, month)
    days = pd.date_range(first, last - pd.Timedelta(days=1), freq="D")
    raw_columns = list(pd.read_csv(input_csv, nrows=0).columns)
    context_s = max(MAX_INTERPOLATION_GAP_SECONDS,
                    MAX_EXTRAPOLATION_SECONDS, MAX_STATE_AGE_SECONDS)
    processed_days: set[str] = set()
    print(f"\nMONTH START: {year:04d}-{month:02d} | input={input_csv}", flush=True)
    start_time = time.monotonic()
    stats_by_day = {}

    for day, raw in iter_days_with_context(
        input_csv, chunksize, first, last, context_s
    ):
        if day < first or day >= last:
            continue
        label = day.strftime("%Y-%m-%d")
        processed_days.add(label)
        path = daily_cache_file(paths, area_key, day)
        if path.exists() and not overwrite:
            print(f"[{label}] CACHED: {path.name}", flush=True)
            continue
        print(f"[{label}] PREPROCESS START | raw/context rows={len(raw):,}", flush=True)
        regularized, stat = regularize_day(raw, day, area)
        regularized = ensure_columns(regularized, raw_columns)
        atomic_csv(path, regularized)
        stats_by_day[label] = {key: (float(value) if key == "elapsed_s" else int(value))
                               for key, value in stat.items()}
        print(
            f"[{label}] PREPROCESS COMPLETE | elapsed={stat['elapsed_s']:.1f}s | "
            f"raw={stat['raw_rows']:,} | processed={stat['regularized_states']:,} | "
            f"snapshots={stat['snapshots']:,} | "
            f"real={stat['real_positions']:,} | "
            f"interpolated={stat['interpolated_positions']:,} | "
            f"extrapolated={stat['extrapolated_positions']:,}",
            flush=True,
        )

    # Days with no raw observations still have a valid empty cache and header.
    for day in days:
        label = day.strftime("%Y-%m-%d")
        path = daily_cache_file(paths, area_key, day)
        if not path.exists():
            if label not in processed_days:
                print(f"[{label}] NO RAW OBSERVATIONS | empty day", flush=True)
                atomic_csv(path, ensure_columns(pd.DataFrame(), raw_columns))
            else:
                raise RuntimeError(f"Expected daily cache missing: {path}")

    day_files = [daily_cache_file(paths, area_key, day) for day in days]
    print(f"MONTH ASSEMBLY START: {year:04d}-{month:02d}", flush=True)
    count = assemble_month(paths, day_files)
    manifest = {
        "version": PREPROCESS_VERSION,
        "source_sector": source_sector,
        "analysis_area": area_key,
        "source_csv": str(input_csv),
        "start_utc": first.isoformat(),
        "end_utc_exclusive": last.isoformat(),
        "snapshot_step_seconds": SNAPSHOT_STEP_SECONDS,
        "max_interpolation_gap_seconds": MAX_INTERPOLATION_GAP_SECONDS,
        "max_extrapolation_seconds": MAX_EXTRAPOLATION_SECONDS,
        "max_state_age_seconds": MAX_STATE_AGE_SECONDS,
        "filter_fl_min": CRUISE_FL_MIN,
        "filter_fl_max": CRUISE_FL_MAX,
        "rows": count,
        "day_stats": stats_by_day,
    }
    temp_manifest = paths["manifest"].with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp_manifest.replace(paths["manifest"])
    print(
        f"MONTH COMPLETE: {year:04d}-{month:02d} | "
        f"rows={count:,} | elapsed={time.monotonic()-start_time:.1f}s | "
        f"output={paths['csv']}", flush=True,
    )
    if not keep_cache:
        shutil.rmtree(paths["cache"], ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSky 10-second reusable preprocessing")
    parser.add_argument("--source-sector", default="LECMSAN_ENVELOPE")
    parser.add_argument("--analysis-area", default="LECMSAN")
    parser.add_argument("--start-month", required=True, help="YYYY-MM inclusive")
    parser.add_argument("--end-month", required=True, help="YYYY-MM inclusive")
    parser.add_argument("--chunksize", type=int, default=READ_CHUNK_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-daily-cache", action="store_true")
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("--chunksize must be >0")
    try:
        start_y, start_m = (int(v) for v in args.start_month.split("-"))
        end_y, end_m = (int(v) for v in args.end_month.split("-"))
        first, _ = month_bounds(start_y, start_m)
        last, _ = month_bounds(end_y, end_m)
    except (ValueError, TypeError):
        parser.error("months must use YYYY-MM with month 01..12")
    if last < first:
        parser.error("--end-month must not precede --start-month")

    area = load_analysis_area(args.analysis_area)
    if area is None:
        parser.error("--analysis-area must name a polygon preset")
    files = discover_monthly_csvs(args.source_sector, first, last)
    if not files:
        raise FileNotFoundError("No monthly raw archive CSV files found")
    print("=" * 78)
    print("OpenSky 10-second PREPROCESSOR")
    print(f"Source archive  : {args.source_sector}")
    print(f"Analysis polygon: {args.analysis_area}")
    print(f"Months          : {args.start_month} to {args.end_month} (inclusive)")
    print(f"10-second output: {SNAPSHOT_OUTPUT_ROOT / args.analysis_area}")
    print(f"Found input CSVs: {len(files)}")
    print("=" * 78, flush=True)
    for path, year, month in files:
        process_month(path, year, month, args.source_sector, args.analysis_area,
                      area, args.chunksize, args.overwrite, args.keep_daily_cache)
    print("PREPROCESSING RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
