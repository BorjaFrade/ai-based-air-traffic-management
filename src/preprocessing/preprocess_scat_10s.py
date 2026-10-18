#!/usr/bin/env python3
"""
Convert SCAT weekly ZIP archives into chronological 10-second snapshots.

SCAT is not an OpenSky state-vector table.  Each ZIP contains one JSON file
per flight, and every flight JSON contains surveillance plots, flight-plan
updates, controller clearances and predicted trajectories.  This script
streams those JSON files directly from the ZIP, attaches the latest temporal
flight-plan/clearance context to each selected surveillance plot, filters to a
configured area and writes sorted daily CSV snapshots.

The output deliberately keeps both the normalised kinematic fields used by
the shared separation model and the SCAT-specific metadata needed to explain
an event later:

    data/processed/scat_10s/<AREA>/date=YYYY-MM-DD/scat_YYYY-MM-DD_10s.csv

The raw ZIP archives are never modified or extracted into thousands of small
files.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
ACQUISITION_DIR = SRC_DIR / "acquisition"

if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))

from area_presets import (  # noqa: E402
    load_area_presets,
    point_is_inside_area,
    resolve_area_preset,
)


INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "scat" / "weekly_archives"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "scat_10s"
BBOX_PRESETS_FILE = ACQUISITION_DIR / "bbox_presets.json"

DEFAULT_AREA = "SCAT_ESMM_GOTHENBURG_GG"
SNAPSHOT_STEP_SECONDS = 10
DEFAULT_SORT_CHUNKSIZE = 250_000

ARCHIVE_RE = re.compile(
    r"^scat(?P<start>\d{8})_(?P<end>\d{8})\.zip$",
    re.IGNORECASE,
)

MPS_TO_KT = 1.9438444924406
MPS_TO_FPM = 196.8503937008
FT_TO_M = 0.3048

SKIP_MEMBERS = {"airspace.json", "grib_meteo.json"}

SCAT_COLUMNS = [
    "time",
    "timestamp_utc",
    "icao24",
    "scat_flight_id",
    "callsign",
    "lat",
    "lon",
    "latitude",
    "longitude",
    "baroaltitude",
    "baro_altitude",
    "baro_altitude_ft",
    "measured_flight_level",
    "velocity",
    "ground_speed_kt",
    "true_track",
    "heading",
    "vertical_rate",
    "vertical_rate_fpm",
    "vertrate",
    "on_ground",
    "lastcontact",
    "source_plot_time",
    "source_plot_epoch",
    "source_plot_offset_s",
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
]


def parse_timestamp(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).timestamp()


def iso_utc(epoch: float) -> str:
    return (
        dt.datetime.fromtimestamp(float(epoch), tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def latest_record(records: list[tuple[float, dict]], times: list[float], epoch: float) -> dict:
    if not records:
        return {}
    index = bisect.bisect_right(times, epoch) - 1
    return records[max(index, 0)][1]


def timeline_records(items: Iterable[dict] | None) -> tuple[list[tuple[float, dict]], list[float]]:
    records = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        epoch = parse_timestamp(item.get("time_stamp", item.get("start_time")))
        if epoch is not None:
            records.append((epoch, item))
    records.sort(key=lambda item: item[0])
    return records, [item[0] for item in records]


def route_summary(route: list[dict] | None) -> tuple[str, str, str]:
    if not isinstance(route, list):
        return "", "", ""
    valid = [item for item in route if isinstance(item, dict)]
    if not valid:
        return "", "", ""
    names = [str(item.get("fix_name", "")).strip() for item in valid]
    names = [name for name in names if name]
    next_fix = names[0] if names else ""
    next_eta = str(valid[0].get("eto", "") or "")
    return ">".join(names), next_fix, next_eta


def prepare_flight_context(flight: dict) -> dict:
    fpl = flight.get("fpl") or {}
    base_records, base_times = timeline_records(fpl.get("fpl_base"))
    plan_records, plan_times = timeline_records(fpl.get("fpl_plan_update"))
    clearance_records, clearance_times = timeline_records(fpl.get("fpl_clearance"))
    centre_records, centre_times = timeline_records(flight.get("centre_ctrl"))
    prediction_records, prediction_times = timeline_records(flight.get("predicted_trajectory"))

    base = base_records[0][1] if base_records else {}
    return {
        "base_records": (base_records, base_times),
        "plan_records": (plan_records, plan_times),
        "clearance_records": (clearance_records, clearance_times),
        "centre_records": (centre_records, centre_times),
        "prediction_records": (prediction_records, prediction_times),
        "base": base,
    }


def extract_kinematics(plot: dict) -> dict | None:
    position = plot.get("I062/105") or {}
    lat = finite_number(position.get("lat"))
    lon = finite_number(position.get("lon"))
    epoch = parse_timestamp(plot.get("time_of_track"))
    if None in (lat, lon, epoch):
        return None

    measured_fl = finite_number((plot.get("I062/136") or {}).get("measured_flight_level"))
    item_380 = plot.get("I062/380") or {}
    altitude_item = item_380.get("subitem6") or {}
    selected_altitude_item = item_380.get("subitem7") or {}
    selected_altitude = finite_number(selected_altitude_item.get("altitude"))
    altitude_ft = (
        measured_fl * 100.0
        if measured_fl is not None
        else selected_altitude
    )
    if altitude_ft is None:
        altitude_ft = finite_number(altitude_item.get("altitude"))

    velocity_item = plot.get("I062/185") or {}
    vx = finite_number(velocity_item.get("vx"))
    vy = finite_number(velocity_item.get("vy"))
    speed_ms = None
    if vx is not None and vy is not None:
        speed_ms = math.hypot(vx, vy)

    heading = finite_number((item_380.get("subitem3") or {}).get("mag_hdg"))
    if heading is None and vx is not None and vy is not None and speed_ms:
        heading = math.degrees(math.atan2(vx, vy)) % 360.0

    rate_fpm = finite_number((plot.get("I062/220") or {}).get("rocd"))
    if rate_fpm is None:
        rate_fpm = finite_number((item_380.get("subitem13") or {}).get("baro_vert_rate"))
    rate_ms = None if rate_fpm is None else rate_fpm / MPS_TO_FPM

    if altitude_ft is None or not math.isfinite(altitude_ft):
        return None

    return {
        "epoch": epoch,
        "lat": lat,
        "lon": lon,
        "measured_flight_level": measured_fl,
        "altitude_ft": altitude_ft,
        "speed_ms": speed_ms,
        "heading": heading,
        "rate_fpm": rate_fpm,
        "rate_ms": rate_ms,
    }


def attach_context(context: dict, epoch: float) -> dict:
    base = context["base"]
    plan_records, plan_times = context["plan_records"]
    clearance_records, clearance_times = context["clearance_records"]
    centre_records, centre_times = context["centre_records"]
    prediction_records, prediction_times = context["prediction_records"]

    plan = latest_record(plan_records, plan_times, epoch)
    clearance = latest_record(clearance_records, clearance_times, epoch)
    centre = latest_record(centre_records, centre_times, epoch)
    prediction = latest_record(prediction_records, prediction_times, epoch)
    route, next_fix, next_eta = route_summary(prediction.get("route"))

    return {
        "callsign": str(base.get("callsign", "") or "").strip(),
        "fpl_adep": base.get("adep", ""),
        "fpl_ades": base.get("ades", ""),
        "fpl_aircraft_type": base.get("aircraft_type", ""),
        "fpl_wtc": base.get("wtc", ""),
        "fpl_flight_rules": base.get("flight_rules", ""),
        "fpl_route": plan.get("icao_route", ""),
        "fpl_rfl": plan.get("rfl_string", ""),
        "fpl_tas": plan.get("tas_string", ""),
        "fpl_copn": plan.get("copn", ""),
        "fpl_copn_pel": plan.get("copn_pel", ""),
        "fpl_copx": plan.get("copx", ""),
        "fpl_copx_pel": plan.get("copx_pel", ""),
        "fpl_update_timestamp": plan.get("time_stamp", ""),
        "clearance_cfl": clearance.get("cfl", ""),
        "clearance_cfl_unit": clearance.get("cfl_unit", ""),
        "clearance_heading_deg": clearance.get("assigned_heading_val", ""),
        "clearance_heading_beacon": clearance.get("assign_heading_beacon", ""),
        "clearance_speed": clearance.get("assigned_speed_val", ""),
        "clearance_speed_unit": clearance.get("assigned_speed_unit", ""),
        "clearance_timestamp": clearance.get("time_stamp", ""),
        "active_centre_id": centre.get("centre_id", ""),
        "centre_since": centre.get("start_time", ""),
        "predicted_route_timestamp": prediction.get("time_stamp", ""),
        "predicted_next_fix": next_fix,
        "predicted_next_fix_eta": next_eta,
        "predicted_route": route,
    }


def build_grid_rows(flight: dict, area: dict, grid_seconds: int) -> list[dict]:
    context = prepare_flight_context(flight)
    candidates: dict[int, tuple[tuple[int, float, float], dict]] = {}
    for plot in flight.get("plots") or []:
        if not isinstance(plot, dict):
            continue
        kinematics = extract_kinematics(plot)
        if kinematics is None:
            continue
        if not point_is_inside_area(kinematics["lat"], kinematics["lon"], area):
            continue

        grid_time = int(math.floor(kinematics["epoch"] / grid_seconds) * grid_seconds)
        offset = grid_time - kinematics["epoch"]
        # Prefer the latest observation at or before the grid.  If no such
        # observation exists for the first grid instant, use the nearest
        # following observation without inventing an interpolated position.
        direction = 0 if offset >= 0 else 1
        score = (direction, abs(offset), -kinematics["epoch"] if direction == 0 else kinematics["epoch"])
        if grid_time in candidates and candidates[grid_time][0] <= score:
            continue

        context_row = attach_context(context, kinematics["epoch"])
        altitude_ft = kinematics["altitude_ft"]
        altitude_m = altitude_ft * FT_TO_M
        flight_id = str(flight.get("id", ""))
        row = {
            "time": grid_time,
            "timestamp_utc": iso_utc(grid_time),
            "icao24": f"SCAT:{flight_id}",
            "scat_flight_id": flight.get("id", ""),
            "callsign": context_row.pop("callsign", "") or f"SCAT-{flight_id}",
            "lat": kinematics["lat"],
            "lon": kinematics["lon"],
            "latitude": kinematics["lat"],
            "longitude": kinematics["lon"],
            "baroaltitude": altitude_m,
            "baro_altitude": altitude_m,
            "baro_altitude_ft": altitude_ft,
            "measured_flight_level": kinematics["measured_flight_level"],
            "velocity": kinematics["speed_ms"],
            "ground_speed_kt": (
                None if kinematics["speed_ms"] is None
                else kinematics["speed_ms"] * MPS_TO_KT
            ),
            "true_track": kinematics["heading"],
            "heading": kinematics["heading"],
            "vertical_rate": kinematics["rate_ms"],
            "vertical_rate_fpm": kinematics["rate_fpm"],
            "vertrate": kinematics["rate_ms"],
            "on_ground": False,
            "lastcontact": grid_time,
            "source_plot_time": iso_utc(kinematics["epoch"]),
            "source_plot_epoch": kinematics["epoch"],
            "source_plot_offset_s": kinematics["epoch"] - grid_time,
            **context_row,
        }
        candidates[grid_time] = (score, row)

    return [candidates[key][1] for key in sorted(candidates)]


def select_archives(input_root: Path, weeks: list[str] | None) -> list[Path]:
    archives = sorted(input_root.glob("scat*.zip"))
    if not archives:
        raise FileNotFoundError(f"No SCAT ZIP archives found under {input_root}")
    if not weeks:
        return archives
    wanted = {week.replace("scat", "").replace(".zip", "") for week in weeks}
    selected = []
    for path in archives:
        match = ARCHIVE_RE.match(path.name)
        if match and f"{match.group('start')}_{match.group('end')}" in wanted:
            selected.append(path)
    if not selected:
        raise FileNotFoundError(
            "None of the requested SCAT weeks were found. "
            f"Available: {[path.name for path in archives]}"
        )
    return selected


def write_part_rows(
    archive: Path,
    area_key: str,
    area: dict,
    work_root: Path,
    grid_seconds: int,
    max_flights: int,
) -> tuple[dict, list[Path]]:
    handles: dict[str, tuple[object, csv.DictWriter]] = {}
    stats = {
        "archive": archive.name,
        "flights_seen": 0,
        "flights_with_rows": 0,
        "rows_written": 0,
        "invalid_json": 0,
        "invalid_plots": 0,
    }
    part_paths: list[Path] = []

    def get_writer(day: str) -> csv.DictWriter:
        if day in handles:
            return handles[day][1]
        target = work_root / archive.stem / f"date={day}" / "part.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=SCAT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        handles[day] = (handle, writer)
        part_paths.append(target)
        return writer

    try:
        with zipfile.ZipFile(archive) as zipped:
            flight_members = [
                name for name in zipped.namelist()
                if name.lower().endswith(".json")
                and Path(name).name.lower() not in SKIP_MEMBERS
            ]
            for member_index, member in enumerate(flight_members, start=1):
                if max_flights and member_index > max_flights:
                    break
                stats["flights_seen"] += 1
                try:
                    with zipped.open(member) as stream:
                        flight = json.load(stream)
                except (OSError, ValueError, json.JSONDecodeError):
                    stats["invalid_json"] += 1
                    continue
                rows = build_grid_rows(flight, area, grid_seconds)
                if not rows:
                    continue
                stats["flights_with_rows"] += 1
                for row in rows:
                    day = dt.datetime.fromtimestamp(
                        float(row["time"]), tz=dt.timezone.utc
                    ).strftime("%Y-%m-%d")
                    get_writer(day).writerow(row)
                    stats["rows_written"] += 1
    finally:
        for handle, _ in handles.values():
            handle.close()

    return stats, part_paths


def sort_part_csv(part_path: Path, output_path: Path, chunksize: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_path.parent / f".{output_path.stem}_sort_chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    sorted_chunks: list[Path] = []
    rows = 0
    try:
        for index, chunk in enumerate(
            pd.read_csv(part_path, chunksize=chunksize, low_memory=False),
        ):
            if chunk.empty:
                continue
            chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
            chunk = chunk.loc[chunk["time"].notna()].copy()
            chunk.sort_values(["time", "icao24"], kind="mergesort", inplace=True)
            target = chunk_dir / f"chunk_{index:06d}.csv"
            chunk.to_csv(target, index=False)
            sorted_chunks.append(target)

        with output_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(SCAT_COLUMNS)
            readers = []
            heap = []
            time_index = SCAT_COLUMNS.index("time")
            id_index = SCAT_COLUMNS.index("icao24")
            for index, chunk_path in enumerate(sorted_chunks):
                handle = chunk_path.open("r", encoding="utf-8", newline="")
                reader = csv.reader(handle)
                next(reader, None)
                readers.append((handle, reader))
                try:
                    row = next(reader)
                except StopIteration:
                    continue
                heap.append((float(row[time_index]), row[id_index], index, row))
            import heapq

            heapq.heapify(heap)
            while heap:
                _, _, reader_index, row = heapq.heappop(heap)
                writer.writerow(row)
                rows += 1
                try:
                    next_row = next(readers[reader_index][1])
                except StopIteration:
                    continue
                heapq.heappush(
                    heap,
                    (
                        float(next_row[time_index]),
                        next_row[id_index],
                        reader_index,
                        next_row,
                    ),
                )
            for handle, _ in readers:
                handle.close()
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chronological 10-second SCAT snapshots with FPL and clearance context."
    )
    parser.add_argument("--area", default=DEFAULT_AREA, help="bbox_presets.json key")
    parser.add_argument("--week", action="append", help="YYYYMMDD_YYYYMMDD; repeatable")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--grid-seconds", type=int, default=SNAPSHOT_STEP_SECONDS)
    parser.add_argument("--sort-chunksize", type=int, default=DEFAULT_SORT_CHUNKSIZE)
    parser.add_argument("--max-flights", type=int, default=0, help="0 means all flights")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep unsorted per-archive parts for audit/debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_seconds <= 0 or args.sort_chunksize <= 0 or args.max_flights < 0:
        raise SystemExit("grid-seconds and sort-chunksize must be positive; max-flights cannot be negative")

    presets = load_area_presets(BBOX_PRESETS_FILE)
    area = resolve_area_preset(args.area, presets)
    archives = select_archives(args.input_root, args.week)
    area_output = args.output_root / args.area
    work_root = area_output / ".scat_parts"
    if args.overwrite and work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"SCAT preprocessing | area={args.area} | archives={len(archives)}")
    print(f"Input: {args.input_root}")
    print(f"Output: {area_output}")

    all_stats = []
    part_paths: list[Path] = []
    for archive in archives:
        print(f"Reading {archive.name} ...", flush=True)
        stats, paths = write_part_rows(
            archive,
            args.area,
            area,
            work_root,
            args.grid_seconds,
            args.max_flights,
        )
        all_stats.append(stats)
        part_paths.extend(paths)
        print(
            f"  flights={stats['flights_seen']:,} | "
            f"flights_in_area={stats['flights_with_rows']:,} | "
            f"rows={stats['rows_written']:,}",
            flush=True,
        )

    output_files = []
    for part_path in sorted(part_paths):
        day = part_path.parent.name.removeprefix("date=")
        output_path = area_output / f"date={day}" / f"scat_{day}_10s.csv"
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing {output_path}")
            continue
        count = sort_part_csv(part_path, output_path, args.sort_chunksize)
        output_files.append({"date": day, "path": str(output_path), "rows": count})
        print(f"  sorted {day}: {count:,} rows", flush=True)

    manifest = {
        "dataset": "SCAT",
        "dataset_doi": "10.17632/8yn985bwz5.1",
        "area": args.area,
        "area_preset": area,
        "grid_seconds": args.grid_seconds,
        "columns": SCAT_COLUMNS,
        "archives": [str(path) for path in archives],
        "archive_stats": all_stats,
        "output_files": output_files,
    }
    area_output.mkdir(parents=True, exist_ok=True)
    (area_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.keep_intermediate:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"COMPLETE: {len(output_files)} daily files")
    print(f"Manifest: {area_output / 'manifest.json'}")


if __name__ == "__main__":
    main()
