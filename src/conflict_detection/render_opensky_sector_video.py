#!/usr/bin/env python3
"""Render local OpenSky traffic as 10-second PNG frames and an MP4.

Envelope mode reconstructs snapshots from downloaded raw CSVs without adding
a polygon or flight-level filter. Existing download filters still apply.
No Trino query is made.
Uses the shared ENROUTE classifier and map renderer.
The four projected trajectory segments cover 0–2, 2–5, 5–10 and 10–20 min.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ACQUISITION_DIR = SCRIPT_DIR.parent / "acquisition"
PREPROCESSING_DIR = SCRIPT_DIR.parent / "preprocessing"
sys.path.insert(0, str(ACQUISITION_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))

from area_presets import (  # noqa: E402
    filter_dataframe_to_area, load_area_presets, resolve_area_preset,
    load_waypoints_for_area, load_corridors_for_area,
)
from opensky_plot import build_shared_validation_render_config, render_map  # noqa: E402
from opensky_safety import (  # noqa: E402
    STATE_CONFLICT, STATE_HIGH_RISK, STATE_NORMAL, STATE_PROXIMITY,
    analyse_aircraft_snapshot, enabled_states, load_safety_profile,
    relevant_pair_results, state_colours, state_display_names,
)
from generate_random_validation_snapshots import prepare_snapshot_for_plot  # noqa: E402
from preprocess_opensky_10s import (  # noqa: E402
    GEOD, geodesic_interpolate, normalize_raw_chunk, regularize_aircraft_to_grid,
)

HORIZONS_MINUTES = (2, 5, 10, 20)
PREDICTION_STEP_SECONDS = 5
PROCESSED_INPUT_ROOT = PROJECT_ROOT / "data/processed/opensky_10s"
RAW_INPUT_ROOT = PROJECT_ROOT / "data/raw/opensky_sectors"
OUTPUT_ROOT = PROJECT_ROOT / "results/opensky_sector_video"
PRESETS = ACQUISITION_DIR / "bbox_presets.json"
PROFILES = ACQUISITION_DIR / "separation_profiles.json"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area", default="LECMSAN_ENVELOPE", help="Map preset and input area key")
    p.add_argument("--start", required=True, help="Inclusive UTC timestamp, e.g. 2019-01-15 00:00:00")
    p.add_argument("--end", required=True, help="Exclusive UTC timestamp, e.g. 2019-01-16 00:00:00")
    p.add_argument("--source", choices=("auto", "raw", "processed"), default="auto",
                   help="auto uses the area preset source, then *_ENVELOPE=raw or processed otherwise")
    p.add_argument("--input-root", type=Path, help="Override the chosen source root")
    p.add_argument("--out", type=Path, help="Output directory; default results/opensky_sector_video/AREA/RANGE")
    p.add_argument("--profile", default="ENROUTE")
    p.add_argument("--frame-step", type=int, default=10, help="Seconds between PNG frames")
    p.add_argument("--fps", type=int, default=10, help="MP4 playback frames per second")
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--width", type=int, help="Even PNG width; default map preset width")
    p.add_argument("--height", type=int, help="Even PNG height; default map preset height")
    p.add_argument("--no-basemap", action="store_true", help="Avoid network map tiles")
    p.add_argument("--show-corridors", action="store_true",
                   help="Draw ATS corridors; hidden by default to keep waypoints legible")
    p.add_argument("--verbose-render", action="store_true", help="Print per-frame map diagnostics")
    p.add_argument("--frames-only", action="store_true", help="Generate PNGs without MP4")
    p.add_argument("--max-frames", type=int, help="Limit frames for a quick local trial")
    p.add_argument("--resume", action="store_true", help="Reuse PNGs from an identical run")
    p.add_argument("--log-every", type=int, default=25, help="Report PNG/MP4 progress every N frames")
    p.add_argument("--max-interpolation-gap-seconds", "--bridge-gap-seconds",
                   dest="max_interpolation_gap_seconds", type=int, default=1800,
                   help="Maximum bounded same-flight gap to interpolate (default: 1800 s)")
    p.add_argument("--audit-only", action="store_true",
                   help="Write traffic coverage and source-filter audit without rendering")
    args = p.parse_args(argv)
    for name in ("frame_step", "fps", "chunksize", "log_every"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.frame_step % 10:
        p.error("--frame-step must be a multiple of the source's 10-second grid")
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")
    if args.max_interpolation_gap_seconds < 0 or args.max_interpolation_gap_seconds % args.frame_step:
        p.error("--max-interpolation-gap-seconds must be non-negative and a multiple of --frame-step")
    return args


def progress(phase, done, total, started, detail=""):
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed
    remaining = (total - done) / rate if rate else 0
    suffix = f" | {detail}" if detail else ""
    print(
        f"{phase} {done:,}/{total:,} ({done / total:.1%}) | "
        f"elapsed={timedelta(seconds=int(elapsed))} | "
        f"rate={rate:.2f} frames/s | "
        f"ETA={timedelta(seconds=int(remaining))}{suffix}",
        flush=True,
    )


def utc_epoch(value):
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    if timestamp.nanosecond or timestamp.microsecond:
        raise ValueError("Start and end must be whole UTC seconds")
    return int(timestamp.timestamp())


def processed_month_files(root, area, start, end, context_seconds=0):
    current = pd.Timestamp(start - context_seconds, unit="s", tz="UTC").replace(
        day=1, hour=0, minute=0, second=0)
    files = []
    while int(current.timestamp()) < end + context_seconds:
        month = current.strftime("%Y-%m")
        directory = root / area / f"year={current.year}" / f"month={current.month:02d}"
        matches = sorted(directory.glob(f"*_{month}_10s.csv"))
        required = int(current.timestamp()) < end and int((current + pd.DateOffset(months=1)).timestamp()) > start
        if len(matches) != 1 and required:
            raise FileNotFoundError(
                f"Expected one processed 10-second CSV for {month} in {directory}; found {len(matches)}. "
                "Run src/preprocessing/preprocess_opensky_10s.py for missing months."
            )
        if len(matches) == 1:
            files.extend(matches)
        current += pd.DateOffset(months=1)
    return files


def raw_input_files(root, area, start, end, context_seconds=20):
    """Prefer complete local hourly cache; fall back to canonical monthly CSVs."""
    context = max(20, context_seconds)
    first_hour = pd.Timestamp(start - context, unit="s", tz="UTC").floor("h")
    last_hour = pd.Timestamp(end + context - 1, unit="s", tz="UTC").floor("h")
    hours = pd.date_range(first_hour, last_hour, freq="h")
    hourly = []
    for hour in hours:
        following = hour + pd.Timedelta(hours=1)
        filename = f"window_{hour:%Y%m%dT%H%M%S}Z_{following:%Y%m%dT%H%M%S}Z.csv"
        hourly.append(root / area / f"year={hour.year}" / f"month={hour.month:02d}" /
                      "_window_cache" / filename)
    if hourly and all(path.is_file() for path in hourly):
        return hourly
    monthly = []
    current = first_hour.replace(day=1, hour=0)
    while current <= last_hour:
        month = current.strftime("%Y-%m")
        directory = root / area / f"year={current.year}" / f"month={current.month:02d}"
        matches = sorted(directory.glob(f"*_{month}.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one raw monthly CSV for {month} in {directory}; "
                                    f"found {len(matches)}")
        monthly.extend(matches)
        current += pd.DateOffset(months=1)
    return monthly


def read_processed_window(files, start, end, chunksize, context_seconds=0):
    parts = []
    for path in files:
        selected = 0
        scanned = 0
        began = time.monotonic()
        print(f"CSV | reading {path} ({path.stat().st_size / 1_000_000:.1f} MB)", flush=True)
        for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False), 1):
            scanned += len(chunk)
            if "time" not in chunk or "icao24" not in chunk:
                raise ValueError(f"Missing time or icao24 in {path}")
            epoch = pd.to_numeric(chunk.time, errors="coerce")
            mask = (epoch.ge(start - context_seconds) & epoch.lt(end + context_seconds)
                    & epoch.mod(10).abs().lt(0.001))
            if mask.any():
                part = chunk.loc[mask].copy()
                part["frame_epoch"] = epoch.loc[mask].round().astype("int64")
                selected += len(part)
                parts.append(part)
            print(f"CSV | chunk={chunk_number:,} | scanned={scanned:,} | "
                  f"selected={selected:,} | elapsed={timedelta(seconds=int(time.monotonic() - began))}",
                  flush=True)
        print(f"CSV | complete {path.name}: {selected:,} rows in requested interval", flush=True)
    if not parts:
        raise ValueError("No local 10-second states exist in this UTC interval")
    states = pd.concat(parts, ignore_index=True)
    if "lastcontact" in states:
        states["lastcontact"] = pd.to_numeric(states.lastcontact, errors="coerce")
        states.sort_values(["frame_epoch", "icao24", "lastcontact"], inplace=True, kind="mergesort")
    states.drop_duplicates(["frame_epoch", "icao24"], keep="last", inplace=True)
    return {int(t): group.drop(columns="frame_epoch").reset_index(drop=True)
            for t, group in states.groupby("frame_epoch", sort=False)}


def read_raw_window(files, start, end, chunksize, context_seconds=20):
    """Reconstruct 10-second positions from all envelope traffic and altitudes."""
    context = max(20, context_seconds)
    parts = []
    for path in files:
        selected = scanned = 0
        began = time.monotonic()
        print(f"CSV | reading raw {path} ({path.stat().st_size / 1_000_000:.1f} MB)", flush=True)
        for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False), 1):
            scanned += len(chunk)
            chunk = normalize_raw_chunk(chunk)
            chosen = chunk.loc[chunk.time.ge(start - context) & chunk.time.lt(end + context)]
            if not chosen.empty:
                parts.append(chosen.copy())
                selected += len(chosen)
            print(f"CSV | chunk={chunk_number:,} | scanned={scanned:,} | "
                  f"selected={selected:,} | elapsed={timedelta(seconds=int(time.monotonic() - began))}",
                  flush=True)
        print(f"CSV | complete {path.name}: {selected:,} relevant raw rows", flush=True)
    if not parts:
        raise ValueError("No raw OpenSky states exist in this UTC interval")
    raw = pd.concat(parts, ignore_index=True)
    raw.sort_values(["time", "icao24", "lastcontact"], inplace=True, kind="mergesort")
    raw.drop_duplicates(["time", "icao24"], keep="last", inplace=True)
    began = time.monotonic()
    print(f"GRID | reconstructing 10-second states from {len(raw):,} raw rows, "
          f"{raw.icao24.nunique():,} aircraft", flush=True)
    records = []
    groups = raw.groupby("icao24", sort=False)
    for i, (_, group) in enumerate(groups, 1):
        records.extend(regularize_aircraft_to_grid(
            group, float(start - context), float(end + context)))
        if i == 1 or i % 100 == 0 or i == groups.ngroups:
            print(f"GRID | aircraft={i:,}/{groups.ngroups:,} | states={len(records):,} | "
                  f"elapsed={timedelta(seconds=int(time.monotonic() - began))}", flush=True)
    if not records:
        raise ValueError("Raw data yielded no usable 10-second states")
    states = pd.DataFrame.from_records(records)
    states["frame_epoch"] = states.time.astype("int64")
    if "lastcontact" in states:
        states.sort_values(["frame_epoch", "icao24", "lastcontact"], inplace=True, kind="mergesort")
    states.drop_duplicates(["frame_epoch", "icao24"], keep="last", inplace=True)
    return {int(t): group.drop(columns="frame_epoch").reset_index(drop=True)
            for t, group in states.groupby("frame_epoch", sort=False)}


def source_filters(files, source):
    manifests = []
    for path in files:
        directory = path.parent.parent if "_window_cache" in path.parts else path.parent
        for manifest in sorted(directory.glob("*.manifest.json")):
            if manifest not in manifests:
                manifests.append(manifest)
    records = []
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        records.append({"manifest": str(manifest), "filters": data.get("filters", {}),
                        "filter_fl_min": data.get("filter_fl_min"),
                        "filter_fl_max": data.get("filter_fl_max")})
    if source == "raw":
        for record in records:
            filters = record["filters"]
            if filters.get("filter_fl_band"):
                print(f"SOURCE FILTER | {record['manifest']} includes FL"
                      f"{filters.get('fl_min')}–{filters.get('fl_max')} only", flush=True)
            if filters.get("keep_only_airborne"):
                print("SOURCE FILTER | ground traffic was excluded during download", flush=True)
    return records


def track_gaps_and_estimates(snapshots, timestamps, max_gap_seconds, bbox):
    """Interpolate bounded gaps of the same flight into complete aircraft states."""
    by_aircraft = {}
    visible = set(timestamps)
    for timestamp in sorted(snapshots):
        frame = snapshots.get(timestamp)
        if frame is None:
            continue
        for _, row in frame.iterrows():
            by_aircraft.setdefault(str(row.icao24), []).append((timestamp, row))

    step = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 10
    end = timestamps[-1] + step if timestamps else 0
    estimates = {}
    gaps = []

    def inside_margin(row):
        return (bbox["lamin"] + .2 <= float(row.lat) <= bbox["lamax"] - .2
                and bbox["lomin"] + .2 <= float(row.lon) <= bbox["lomax"] - .2)

    def label(row, icao):
        callsign = row.get("callsign")
        return str(callsign).strip() if pd.notna(callsign) and str(callsign).strip() else icao

    for icao, points in by_aircraft.items():
        for (before, previous), (after, following) in zip(points[:-1], points[1:]):
            if after - before <= step:
                continue
            missing = (after - before) // step - 1
            target_times = [t for t in range(before + step, after, step) if t in visible]
            if not target_times:
                continue
            before_callsign = label(previous, icao)
            after_callsign = label(following, icao)
            same_flight = (before_callsign == after_callsign or before_callsign == icao
                           or after_callsign == icao)
            altitude_before = pd.to_numeric(previous.get("baroaltitude"), errors="coerce")
            altitude_after = pd.to_numeric(following.get("baroaltitude"), errors="coerce")
            azimuth, _, distance_m = GEOD.inv(
                float(previous.lon), float(previous.lat), float(following.lon), float(following.lat))
            elapsed = after - before
            speed_ms = distance_m / elapsed
            plausible_motion = (np.isfinite(speed_ms) and speed_ms <= 400
                                and not (distance_m < 100 and altitude_before > 3000))
            can_estimate = (0 < max_gap_seconds >= elapsed and same_flight
                            and np.isfinite(altitude_before) and np.isfinite(altitude_after)
                            and plausible_motion)
            gaps.append({"icao24": icao, "callsign": label(previous, icao),
                         "kind": "between_observations", "last_utc": pd.Timestamp(before, unit="s", tz="UTC").isoformat(),
                         "next_utc": pd.Timestamp(after, unit="s", tz="UTC").isoformat(),
                         "missing_frames": missing, "gap_seconds": after - before - step,
                         "last_lat": float(previous.lat), "last_lon": float(previous.lon),
                         "last_altitude_ft": float(previous.baroaltitude) / .3048,
                         "last_well_inside_envelope": inside_margin(previous),
                         "estimated_frames": len(target_times) if can_estimate else 0})
            if not can_estimate:
                continue
            for timestamp in target_times:
                alpha = (timestamp - before) / (after - before)
                lat, lon = geodesic_interpolate(
                    float(previous.lat), float(previous.lon),
                    float(following.lat), float(following.lon), alpha)
                state = previous.to_dict()
                state.update({"time": float(timestamp), "lat": lat, "lon": lon,
                              "baroaltitude": float(altitude_before +
                                                     (altitude_after - altitude_before) * alpha),
                              "velocity": float(speed_ms),
                              "heading": float(azimuth % 360),
                              "vertrate": float((altitude_after - altitude_before) / elapsed),
                              "state_source": "GAP_ESTIMATE",
                              "state_age_s": float(timestamp - before),
                              "position_age_s": float(timestamp - before)})
                geo_before = pd.to_numeric(previous.get("geoaltitude"), errors="coerce")
                geo_after = pd.to_numeric(following.get("geoaltitude"), errors="coerce")
                if np.isfinite(geo_before) and np.isfinite(geo_after):
                    state["geoaltitude"] = float(geo_before + (geo_after - geo_before) * alpha)
                estimates.setdefault(timestamp, []).append(state)
        last_time, last_row = points[-1]
        if timestamps and timestamps[0] <= last_time < end and last_time + step < end:
            gaps.append({"icao24": icao, "callsign": label(last_row, icao),
                         "kind": "after_last_observation", "last_utc": pd.Timestamp(last_time, unit="s", tz="UTC").isoformat(),
                         "next_utc": "", "missing_frames": (end - last_time) // step - 1,
                         "gap_seconds": end - last_time - step,
                         "last_lat": float(last_row.lat), "last_lon": float(last_row.lon),
                         "last_altitude_ft": float(last_row.baroaltitude) / .3048,
                         "last_well_inside_envelope": inside_margin(last_row),
                         "estimated_frames": 0})
    return pd.DataFrame(gaps), estimates


def add_estimates_to_snapshots(snapshots, estimates):
    """Put estimated rows in the same frames used by safety analysis and drawing."""
    for timestamp, rows in estimates.items():
        added = pd.DataFrame.from_records(rows)
        existing = snapshots.get(timestamp)
        snapshots[timestamp] = (added if existing is None else
                                pd.concat((existing, added), ignore_index=True))


def traffic_coverage(snapshots, timestamps):
    """Count observed and estimated aircraft used in each frame."""
    rows = []
    for timestamp in timestamps:
        frame = snapshots.get(timestamp)
        sources = (frame.state_source.value_counts() if frame is not None and
                   "state_source" in frame else pd.Series(dtype="int64"))
        estimated = int(sources.get("GAP_ESTIMATE", 0))
        rows.append({"time_utc": pd.Timestamp(timestamp, unit="s", tz="UTC").isoformat(),
                     "aircraft_live": 0 if frame is None else len(frame) - estimated,
                     "aircraft_estimated_between_observations": estimated,
                     "real_position": int(sources.get("REAL_POSITION", 0)),
                     "interpolated_position": int(sources.get("INTERPOLATED_POSITION", 0)),
                     "extrapolated_position": int(sources.get("EXTRAPOLATED_POSITION", 0))})
    return pd.DataFrame(rows)


def output_configuration(args, start, end, area, width, height):
    return {"area": args.area, "start_utc": pd.Timestamp(start, unit="s", tz="UTC").isoformat(),
            "end_utc_exclusive": pd.Timestamp(end, unit="s", tz="UTC").isoformat(),
            "source": args.source,
            "input_root": str(args.input_root.resolve()),
            "profile": args.profile, "frame_step_seconds": args.frame_step, "fps": args.fps,
            "width_px": width, "height_px": height,
            "lookahead_minutes": list(HORIZONS_MINUTES),
            "prediction_step_seconds": PREDICTION_STEP_SECONDS,
            "max_interpolation_gap_seconds": args.max_interpolation_gap_seconds,
            "gap_estimator": "same-flight-geodesic-linear-altitude-v2-context",
            "show_corridors": args.show_corridors,
            "basemap": not args.no_basemap, "area_name": area["name"]}


def write_video(paths, target, fps, log_every):
    began = time.monotonic()
    print(f"MP4 | encoding {len(paths):,} PNGs at {fps} fps → {target}", flush=True)
    try:
        writer = imageio.get_writer(target, fps=fps, codec="libx264", format="FFMPEG",
                                     pixelformat="yuv420p", macro_block_size=None)
    except (ImportError, ValueError) as exc:
        raise RuntimeError("MP4 encoding needs imageio-ffmpeg") from exc
    expected = None
    with writer:
        for i, path in enumerate(paths, 1):
            pixels = imageio.imread(path)
            if pixels.ndim != 3 or pixels.shape[2] < 3:
                raise ValueError(f"Unexpected PNG shape: {path}: {pixels.shape}")
            rgb = pixels[:, :, :3]
            if expected is None:
                expected = rgb.shape
                if expected[0] % 2 or expected[1] % 2:
                    raise ValueError("H.264 yuv420p requires even frame dimensions")
            elif rgb.shape != expected:
                raise ValueError(f"PNG dimensions differ: {path}: {rgb.shape} versus {expected}")
            writer.append_data(rgb)
            if i == 1 or i % log_every == 0 or i == len(paths):
                progress("MP4", i, len(paths), began, path.name)


def main(argv=None):
    run_began = time.monotonic()
    args = arguments(argv)
    presets = load_area_presets(PRESETS)
    if args.area not in presets:
        raise ValueError(f"Unknown map preset: {args.area}")
    area = resolve_area_preset(args.area, presets)
    if args.source == "auto":
        args.source = area.get("source") or ("raw" if args.area.endswith("_ENVELOPE")
                                              else "processed")
    if args.input_root is None:
        args.input_root = RAW_INPUT_ROOT if args.source == "raw" else PROCESSED_INPUT_ROOT
    start, end = utc_epoch(args.start), utc_epoch(args.end)
    if end <= start:
        raise ValueError("--end must follow --start")
    if start % args.frame_step or end % args.frame_step:
        raise ValueError("UTC start/end must align with --frame-step")
    width = args.width or int(area.get("render", {}).get("width_px", 3840))
    height = args.height or int(area.get("render", {}).get("height_px", 2160))
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("PNG width and height must be positive even integers")
    run_name = (f"{pd.Timestamp(start, unit='s', tz='UTC'):%Y%m%dT%H%M%S}_"
                f"{pd.Timestamp(end, unit='s', tz='UTC'):%Y%m%dT%H%M%S}")
    out = args.out or OUTPUT_ROOT / args.area / run_name
    frames_dir = out / "frames"
    config_path = out / "run_config.json"
    config = output_configuration(args, start, end, area, width, height)
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {out}; pass --resume for an identical run")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing run_config.json differs; use another --out")
    elif args.resume and frames_dir.exists() and any(frames_dir.glob("*.png")):
        raise ValueError("Existing frames lack run_config.json; use another --out")
    frames_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    profile = load_safety_profile(PROFILES, args.profile)
    rules = profile["states"]
    red = rules[STATE_CONFLICT]
    colours = state_colours(rules)
    names = state_display_names(rules)
    map_config = build_shared_validation_render_config(
        area, width_px=width, height_px=height, dpi=160,
        title_prefix=("OpenSky envelope traffic" if args.source == "raw"
                      else "OpenSky sector traffic"),
        show_basemap=not args.no_basemap,
        overrides={"SEPARATION_PROFILE": args.profile, "LOOKAHEAD_MINUTES": list(HORIZONS_MINUTES),
                   "ENABLED_SAFETY_STATES": enabled_states(rules),
                   "SAFETY_STATE_COLORS": colours, "SAFETY_STATE_DISPLAY_NAMES": names,
                   "NORMAL_AIRCRAFT_COLOR": colours[STATE_NORMAL],
                   "PROXIMITY_AIRCRAFT_COLOR": colours[STATE_PROXIMITY],
                   "HIGH_RISK_AIRCRAFT_COLOR": colours[STATE_HIGH_RISK],
                   "CONFLICT_AIRCRAFT_COLOR": colours[STATE_CONFLICT],
                   "WARNING_AIRCRAFT_COLOR": colours[STATE_PROXIMITY],
                   "MIN_HORIZONTAL_SEPARATION_NM": float(red["horizontal_nm"]),
                   "MIN_VERTICAL_SEPARATION_FT": float(red["vertical_ft"]),
                   "SHOW_WAYPOINTS": True, "SHOW_WAYPOINT_LABELS": True,
                   "SHOW_CORRIDORS": args.show_corridors,
                   "SHOW_CORRIDOR_LABELS": args.show_corridors},
    )
    waypoints = load_waypoints_for_area(ACQUISITION_DIR / "waypoints.json", args.area, area)
    corridors = (load_corridors_for_area(ACQUISITION_DIR / "corridors.json", args.area, area)
                 if args.show_corridors else [])
    timestamps = list(range(start, end, args.frame_step))
    if args.max_frames is not None:
        timestamps = timestamps[:args.max_frames]
    paths = [frames_dir / f"frame_{i:06d}_{pd.Timestamp(t, unit='s', tz='UTC'):%Y%m%dT%H%M%SZ}.png"
             for i, t in enumerate(timestamps)]
    missing = [(i, t, path) for i, (t, path) in enumerate(zip(timestamps, paths), 1)
               if not (args.resume and path.exists())]
    print(f"RUN | area={args.area} | source={args.source} | UTC={config['start_utc']} to "
          f"{config['end_utc_exclusive']} (end exclusive) | "
          f"size={width}x{height} | interval={args.frame_step}s | fps={args.fps} | "
          f"horizons={list(HORIZONS_MINUTES)} min", flush=True)
    print(f"PNG | total={len(paths):,} | already present={len(paths) - len(missing):,} | "
          f"to render={len(missing):,}", flush=True)
    context_seconds = max(20, args.max_interpolation_gap_seconds)
    files = (raw_input_files(args.input_root, args.area, start, end, context_seconds)
             if args.source == "raw" else
             processed_month_files(args.input_root, args.area, start, end, context_seconds))
    input_filters = source_filters(files, args.source)
    snapshots = (read_raw_window(files, start, end, args.chunksize, context_seconds)
                 if args.source == "raw" else
                 read_processed_window(files, start, end, args.chunksize, context_seconds))
    print(f"CSV | loaded {len(snapshots):,} occupied 10-second instants; "
          f"elapsed={timedelta(seconds=int(time.monotonic() - run_began))}", flush=True)
    gaps, estimates = track_gaps_and_estimates(
        snapshots, timestamps, args.max_interpolation_gap_seconds, area["bbox"])
    gap_file = out / "track_gaps.csv"
    gaps.to_csv(gap_file, index=False)
    add_estimates_to_snapshots(snapshots, estimates)
    coverage = traffic_coverage(snapshots, timestamps)
    coverage_file = out / "traffic_coverage.csv"
    coverage.to_csv(coverage_file, index=False)
    empty_live = int(coverage.aircraft_live.eq(0).sum())
    empty_visible = int((coverage.aircraft_live + coverage.aircraft_estimated_between_observations).eq(0).sum())
    estimate_rows = int(coverage.aircraft_estimated_between_observations.sum())
    print(f"COVERAGE | no observed aircraft: {empty_live:,}/{len(timestamps):,} frames | "
          f"no observed or estimated aircraft: {empty_visible:,}/{len(timestamps):,} frames | "
          f"estimated states={estimate_rows:,} across {len(estimates):,} frames | "
          f"audit={coverage_file} | gaps={gap_file}", flush=True)
    if args.audit_only:
        print("AUDIT COMPLETE | no PNGs or MP4 rendered", flush=True)
        return
    empty = next(iter(snapshots.values())).iloc[0:0] if snapshots else None
    blank_count = 0
    fallback_file = out / "label_fallback_frames.json"
    label_fallbacks = json.loads(fallback_file.read_text()) if args.resume and fallback_file.exists() else []
    render_began = time.monotonic()
    for rendered, (ordinal, timestamp, path) in enumerate(missing, 1):
        raw = snapshots.get(timestamp, empty)
        if raw is None:
            raise ValueError("No source snapshots were loaded")
        if raw.empty:
            blank_count += 1
        traffic = prepare_snapshot_for_plot(raw)
        traffic = filter_dataframe_to_area(traffic, area, "latitude", "longitude")
        traffic.reset_index(drop=True, inplace=True)
        pairs, current, intervals, _ = analyse_aircraft_snapshot(
            traffic, min_horizontal_nm=float(red["horizontal_nm"]),
            min_vertical_ft=float(red["vertical_ft"]), state_rules=rules,
            lookahead_minutes=HORIZONS_MINUTES,
            prediction_step_seconds=PREDICTION_STEP_SECONDS,
            check_ground_aircraft=False,
        )
        relevant = relevant_pair_results(pairs)
        warnings = [p for p in relevant if p["state_current"] in (STATE_PROXIMITY, STATE_HIGH_RISK)]
        conflicts = [p for p in relevant if p["state_current"] == STATE_CONFLICT]
        map_config["LOOKAHEAD_INTERVAL_STATES"] = intervals
        output = contextlib.nullcontext() if args.verbose_render else contextlib.redirect_stdout(io.StringIO())
        frame_kwargs = dict(
            aircraft=traffic, area=area, bbox=area["bbox"], aircraft_states=current,
            warnings=warnings, conflicts=conflicts,
            snapshot_time=pd.Timestamp(timestamp, unit="s", tz="UTC").to_pydatetime(),
            map_file=path, waypoints=waypoints, corridors=corridors,
            show=False, save=False,
        )
        used_label_fallback = False
        try:
            with output:
                fig, _ = render_map(config=map_config, **frame_kwargs)
        except RuntimeError as exc:
            if "zero-intersection label layout" not in str(exc):
                raise
            plt.close("all")
            fallback_config = {**map_config, "SHOW_LABELS": False,
                               "SHOW_WAYPOINT_LABELS": False,
                               "SHOW_CORRIDOR_LABELS": False}
            label = pd.Timestamp(timestamp, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"PNG | label layout failed at {label}; retrying without text labels", flush=True)
            with output:
                fig, ax = render_map(config=fallback_config, **frame_kwargs)
            used_label_fallback = True
        else:
            ax = fig.axes[0]
        if not args.no_basemap and not ax.images:
            plt.close(fig)
            raise RuntimeError(
                "Satellite basemap tiles were unavailable; no frame was saved. "
                "Restore tile access, or pass --no-basemap for a dark background."
            )
        if traffic.empty:
            ax.text(.5, .5, "No usable aircraft positions in the source at this time",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    color="white", bbox={"facecolor": "#1f2937", "alpha": .8,
                                         "edgecolor": "none", "pad": 5})
        fig.savefig(path, dpi=160, facecolor="white", pil_kwargs={"compress_level": 1})
        plt.close(fig)
        if used_label_fallback:
            label_fallbacks.append({"frame": ordinal, "utc": label, "png": path.name})
            fallback_file.write_text(json.dumps(label_fallbacks, indent=2) + "\n")
        if rendered == 1 or rendered % args.log_every == 0 or rendered == len(missing):
            label = pd.Timestamp(timestamp, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
            progress("PNG", rendered, len(missing), render_began,
                     f"overall={ordinal:,}/{len(paths):,} | {label} | aircraft={len(traffic)} | "
                     f"estimated={int(coverage.iloc[ordinal - 1].aircraft_estimated_between_observations)} | "
                     f"pairs={len(pairs)} | empty={blank_count:,}")
    if not args.frames_only:
        write_video(paths, out / "video.mp4", args.fps, args.log_every)
    summary = {**config, "input_files": [str(p) for p in files],
               "frames": len(paths), "new_frames": len(missing), "blank_frames_generated": blank_count,
               "frames_without_live_aircraft": empty_live,
               "frames_without_live_or_estimated_aircraft": empty_visible,
               "track_gap_events": len(gaps), "estimated_states": estimate_rows,
               "estimated_frames": len(estimates),
               "track_gaps_csv": str(gap_file),
               "source_filters": input_filters, "traffic_coverage_csv": str(coverage_file),
               "label_fallback_frames": len(label_fallbacks),
               "video": None if args.frames_only else str(out / "video.mp4"),
               "elapsed_seconds": round(time.monotonic() - run_began, 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"frames": len(paths), "output": str(out), "video": summary["video"]}, indent=2))


if __name__ == "__main__":
    main()
