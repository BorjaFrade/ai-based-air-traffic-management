#!/usr/bin/env python3
"""Build continuous interaction features around intervention/control windows."""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import DEFAULT_OUTPUT_ROOT, parse_timestamp, update_read_me_first


STATE_COLUMNS = [
    "flight_key", "flight_id", "archive_week", "timestamp", "time_epoch", "callsign", "latitude", "longitude",
    "altitude_ft", "flight_level", "ground_speed_kt", "track_deg", "heading_deg", "vertical_rate_fpm",
    "flight_phase", "active_centre_id", "fpl_route", "predicted_next_fix", "predicted_route_json",
]
HORIZONS = (30, 60, 120)
DEFAULT_INPUT_BATCH_SIZE = 50_000
DEFAULT_TIME_BLOCK_HOURS = 1
TARGET_COLUMNS = [
    "window_id", "clearance_id", "control_for_clearance_id", "window_role", "ATC_INTERVENTION",
    "event_epoch", "event_timestamp", "archive_week", "relative_seconds", "is_event_time", "predictor_eligible",
    "flight_key", "flight_id", "timestamp", "time_epoch", "callsign", "latitude", "longitude",
    "altitude_ft", "flight_level", "ground_speed_kt", "track_deg", "heading_deg", "vertical_rate_fpm",
    "flight_phase", "active_centre_id", "fpl_route", "predicted_next_fix", "predicted_route_json",
]


def existing_columns(path: Path, wanted: list[str]) -> list[str]:
    available = set(pq.read_schema(path).names)
    return [column for column in wanted if column in available]


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def velocity_components(speed_kt, track_deg) -> tuple[float, float]:
    speed = finite(speed_kt)
    track = finite(track_deg)
    if speed is None or track is None:
        return np.nan, np.nan
    angle = math.radians(track)
    # x=east, y=north; kt is NM/hour, converted to NM/minute later.
    return speed * math.sin(angle), speed * math.cos(angle)


def horizontal_geometry(focal: pd.Series, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat0 = float(focal["latitude"])
    lon0 = float(focal["longitude"])
    lat1 = other["latitude"].to_numpy(dtype=float)
    lon1 = other["longitude"].to_numpy(dtype=float)
    mean_lat = np.radians((lat1 + lat0) / 2.0)
    dx_nm = (lon1 - lon0) * 60.0 * np.cos(mean_lat)
    dy_nm = (lat1 - lat0) * 60.0
    separation = np.hypot(dx_nm, dy_nm)
    bearing = (np.degrees(np.arctan2(dx_nm, dy_nm)) + 360.0) % 360.0
    return dx_nm, dy_nm, separation, bearing


@functools.lru_cache(maxsize=100_000)
def parse_route_points_raw(route_json: str) -> tuple[tuple[float, float, float], ...]:
    if not route_json:
        return ()
    try:
        route = json.loads(route_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(route, list):
        return ()
    points = []
    for item in route:
        if not isinstance(item, dict):
            continue
        lat = finite(item.get("lat"))
        lon = finite(item.get("lon"))
        eto = parse_timestamp(item.get("eto"))
        if None in (lat, lon, eto):
            continue
        points.append((float(eto), float(lat), float(lon)))
    return tuple(points)


def parse_route_points(route_json: str, current_epoch: float) -> list[tuple[float, float, float]]:
    return [point for point in parse_route_points_raw(route_json) if point[0] >= current_epoch - 30]


def route_distance_nm(point_a, point_b) -> float:
    _, lat_a, lon_a = point_a
    _, lat_b, lon_b = point_b
    mean_lat = math.radians((lat_a + lat_b) / 2.0)
    dx = (lon_b - lon_a) * 60.0 * math.cos(mean_lat)
    dy = (lat_b - lat_a) * 60.0
    return math.hypot(dx, dy)


@functools.lru_cache(maxsize=200_000)
def topsky_proxy(route_a_json: str, route_b_json: str, current_epoch_rounded: int) -> tuple[float, float]:
    """Compare future TopSky points with a time-consistency guard.

    This is intentionally named a proxy: the public SCAT export provides
    predicted points, not a full conflict-assessment result.  We therefore
    keep the continuous minimum predicted point distance and time-to-minimum,
    without converting them into a conflict label.
    """

    current_epoch = float(current_epoch_rounded)
    route_a = parse_route_points(route_a_json, current_epoch)
    route_b = parse_route_points(route_b_json, current_epoch)
    if not route_a or not route_b:
        return np.nan, np.nan
    best_distance = math.inf
    best_time = math.nan
    for point_a in route_a:
        for point_b in route_b:
            if abs(point_a[0] - point_b[0]) > 120:
                continue
            distance = route_distance_nm(point_a, point_b)
            if distance < best_distance:
                best_distance = distance
                best_time = max(0.0, min(point_a[0], point_b[0]) - current_epoch)
    if not math.isfinite(best_distance):
        return np.nan, np.nan
    return best_distance, best_time


@functools.lru_cache(maxsize=100_000)
def route_token_set(value: str) -> frozenset[str]:
    return frozenset(token for token in str(value or "").replace(">", " ").split() if token)


def pair_features(
    focal: pd.Series,
    others: pd.DataFrame,
    radius_nm: float,
    max_neighbours: int | None = None,
) -> pd.DataFrame:
    dx_nm, dy_nm, separation, bearing = horizontal_geometry(focal, others)
    traffic_density = len(others) + 1
    nearby_count = int((separation <= radius_nm).astype(int).sum())
    if max_neighbours is not None and len(others) > max_neighbours:
        nearest = np.argsort(separation, kind="stable")[:max_neighbours]
        others = others.iloc[nearest].copy()
        dx_nm = dx_nm[nearest]
        dy_nm = dy_nm[nearest]
        separation = separation[nearest]
        bearing = bearing[nearest]
    focal_vx, focal_vy = velocity_components(focal.get("ground_speed_kt"), focal.get("track_deg"))
    other_vectors = np.array(
        [velocity_components(speed, track) for speed, track in zip(others["ground_speed_kt"], others["track_deg"])],
        dtype=float,
    )
    relative_vx_kt = other_vectors[:, 0] - focal_vx
    relative_vy_kt = other_vectors[:, 1] - focal_vy
    relative_speed_kt = np.hypot(relative_vx_kt, relative_vy_kt)
    relative_vx_nm_min = relative_vx_kt / 60.0
    relative_vy_nm_min = relative_vy_kt / 60.0
    velocity_norm_sq = relative_vx_nm_min ** 2 + relative_vy_nm_min ** 2
    dot = dx_nm * relative_vx_nm_min + dy_nm * relative_vy_nm_min
    tcpa_min = np.where(velocity_norm_sq > 1e-12, -dot / velocity_norm_sq, np.nan)
    cpa_dx = dx_nm + relative_vx_nm_min * np.maximum(tcpa_min, 0)
    cpa_dy = dy_nm + relative_vy_nm_min * np.maximum(tcpa_min, 0)
    cpa_nm = np.hypot(cpa_dx, cpa_dy)
    focal_alt = float(focal["altitude_ft"])
    other_alt = others["altitude_ft"].to_numpy(dtype=float)
    signed_vertical = other_alt - focal_alt
    vertical_sep = np.abs(signed_vertical)
    focal_vr = finite(focal.get("vertical_rate_fpm"))
    focal_vr = 0.0 if focal_vr is None else focal_vr
    other_vr = pd.to_numeric(others["vertical_rate_fpm"], errors="coerce").fillna(0).to_numpy(dtype=float)
    relative_vertical_rate = other_vr - focal_vr
    vertical_closing_rate = -np.sign(signed_vertical) * relative_vertical_rate
    predicted_at_tcpa = np.abs(signed_vertical + relative_vertical_rate * np.maximum(tcpa_min, 0))
    focal_heading = finite(focal.get("heading_deg"))
    other_heading = pd.to_numeric(others["heading_deg"], errors="coerce").to_numpy(dtype=float)
    relative_heading = ((other_heading - (focal_heading or 0.0) + 180.0) % 360.0) - 180.0
    focal_track = finite(focal.get("track_deg"))
    other_track = pd.to_numeric(others["track_deg"], errors="coerce").to_numpy(dtype=float)
    crossing_angle = np.abs(((other_track - (focal_track or 0.0) + 180.0) % 360.0) - 180.0)
    shared_fix = []
    route_overlap = []
    focal_route_tokens = route_token_set(focal.get("fpl_route", ""))
    for _, other in others.iterrows():
        other_tokens = route_token_set(other.get("fpl_route", ""))
        overlap = focal_route_tokens & other_tokens
        route_overlap.append(len(overlap))
        shared_fix.append(bool(focal.get("predicted_next_fix") and focal.get("predicted_next_fix") == other.get("predicted_next_fix")))

    topsky_min = []
    topsky_time = []
    current_epoch = int(round(float(focal["time_epoch"])))
    focal_route_json = str(focal.get("predicted_route_json", "") or "")
    for _, other in others.iterrows():
        distance, seconds = topsky_proxy(
            focal_route_json,
            str(other.get("predicted_route_json", "") or ""),
            current_epoch,
        )
        topsky_min.append(distance)
        topsky_time.append(seconds)

    result = pd.DataFrame(
        {
            "other_flight_key": others["flight_key"].astype(str).to_numpy(),
            "other_callsign": others.get("callsign", pd.Series("", index=others.index)).astype(str).to_numpy(),
            "horizontal_separation_nm": separation,
            "vertical_separation_ft": vertical_sep,
            "relative_velocity_kt": relative_speed_kt,
            "horizontal_closing_rate_nm_min": -dot / np.maximum(separation, 1e-12),
            "vertical_closing_rate_fpm": vertical_closing_rate,
            "relative_heading_deg": relative_heading,
            "crossing_angle_deg": crossing_angle,
            "relative_bearing_deg": bearing,
            "cpa_nm": cpa_nm,
            "tcpa_min": tcpa_min,
            "predicted_min_horizontal_sep_nm": cpa_nm,
            "predicted_min_vertical_sep_ft": predicted_at_tcpa,
            "time_to_predicted_minimum_s": np.maximum(tcpa_min, 0) * 60.0,
            "traffic_density": traffic_density,
            "nearby_aircraft_count_30nm": nearby_count,
            "other_altitude_ft": other_alt,
            "other_flight_level": pd.to_numeric(others["flight_level"], errors="coerce").to_numpy(dtype=float),
            "other_ground_speed_kt": pd.to_numeric(others["ground_speed_kt"], errors="coerce").to_numpy(dtype=float),
            "other_vertical_rate_fpm": other_vr,
            "shared_predicted_next_fix": shared_fix,
            "shared_fpl_route_token_count": route_overlap,
            "topsky_predicted_min_sep_nm_proxy": topsky_min,
            "topsky_predicted_time_to_min_s_proxy": topsky_time,
            "topsky_route_pair_available": [bool(focal_route_json and str(value or "")) for value in others["predicted_route_json"]],
            "other_sector": others.get("active_centre_id", pd.Series("", index=others.index)).astype(str).to_numpy(),
        }
    )
    return result


def add_trends(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features.sort_values(["window_id", "focal_flight_key", "other_flight_key", "time_epoch"], inplace=True, kind="mergesort")
    # groupby().groups contains index labels.  Reset after sorting so those
    # labels are also positional indices when values are assigned via iloc.
    features.reset_index(drop=True, inplace=True)
    for horizon in HORIZONS:
        values = np.full(len(features), np.nan, dtype=float)
        for _, indices in features.groupby(["window_id", "focal_flight_key", "other_flight_key"], sort=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            group = features.iloc[positions]
            times = group["time_epoch"].to_numpy(dtype=float)
            separations = group["horizontal_separation_nm"].to_numpy(dtype=float)
            past = np.interp(times - horizon, times, separations, left=np.nan, right=np.nan)
            values[positions] = separations - past
        features[f"separation_trend_{horizon}s"] = values
    return features


class ParquetAppender:
    """Small streaming Parquet writer used for temporary input partitions."""

    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        elif not table.schema.equals(self.writer.schema, check_metadata=False):
            table = table.cast(self.writer.schema, safe=False)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def week_row_groups(path: Path, week: str) -> tuple[pq.ParquetFile, list[int]]:
    parquet = pq.ParquetFile(path)
    archive_index = parquet.schema.names.index("archive_week")
    row_groups = []
    for index in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(index).column(archive_index).statistics
        if statistics is None or statistics.min <= week <= statistics.max:
            row_groups.append(index)
    if not row_groups:
        raise RuntimeError(f"No state row groups found for archive week {week}")
    return parquet, row_groups


def clean_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def partition_targets(
    paths: list[Path],
    week: str,
    target_dir: Path,
    block_seconds: int,
    batch_size: int,
    max_rows: int,
) -> tuple[set[int], int]:
    """Partition window rows by hour without materialising the full windows."""

    appenders: dict[int, ParquetAppender] = {}
    block_ids: set[int] = set()
    target_rows = 0
    try:
        for path in paths:
            if not path.exists() or (max_rows and target_rows >= max_rows):
                continue
            parquet = pq.ParquetFile(path)
            columns = existing_columns(path, TARGET_COLUMNS)
            for record_batch in parquet.iter_batches(
                batch_size=batch_size, columns=columns, use_threads=True
            ):
                frame = record_batch.to_pandas()
                if frame.empty:
                    continue
                if "archive_week" in frame:
                    frame = frame.loc[frame["archive_week"].astype(str) == week].copy()
                if frame.empty:
                    continue
                frame["time_epoch"] = pd.to_numeric(frame["time_epoch"], errors="coerce")
                for column in ("latitude", "longitude", "altitude_ft"):
                    if column in frame:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce")
                required = ["window_id", "flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"]
                frame.dropna(subset=[column for column in required if column in frame], inplace=True)
                if frame.empty:
                    continue
                frame.drop_duplicates(["window_id", "flight_key", "time_epoch"], inplace=True)
                if max_rows:
                    remaining = max_rows - target_rows
                    frame = frame.head(remaining)
                if frame.empty:
                    continue
                frame["_time_block"] = np.floor(
                    frame["time_epoch"].to_numpy(dtype=float) / block_seconds
                ).astype("int64")
                for block, group in frame.groupby("_time_block", sort=False):
                    block = int(block)
                    block_ids.add(block)
                    appender = appenders.setdefault(
                        block,
                        ParquetAppender(target_dir / f"targets_{week}_{block}.parquet"),
                    )
                    appender.append(group.drop(columns=["_time_block"]))
                target_rows += len(frame)
                del frame
                if max_rows and target_rows >= max_rows:
                    break
    finally:
        for appender in appenders.values():
            appender.close()
    return block_ids, target_rows


def partition_states(
    states_path: Path,
    week: str,
    block_ids: set[int],
    state_dir: Path,
    block_seconds: int,
    batch_size: int,
) -> int:
    """Partition one archive week of states into only the needed time blocks."""

    parquet, row_groups = week_row_groups(states_path, week)
    columns = existing_columns(states_path, STATE_COLUMNS)
    appenders: dict[int, ParquetAppender] = {}
    row_count = 0
    try:
        for record_batch in parquet.iter_batches(
            batch_size=batch_size, row_groups=row_groups, columns=columns, use_threads=True
        ):
            frame = record_batch.to_pandas()
            if frame.empty:
                continue
            frame["time_epoch"] = pd.to_numeric(frame["time_epoch"], errors="coerce")
            for column in ("latitude", "longitude", "altitude_ft"):
                if column in frame:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame.dropna(subset=["flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"], inplace=True)
            if frame.empty:
                continue
            frame["_time_block"] = np.floor(
                frame["time_epoch"].to_numpy(dtype=float) / block_seconds
            ).astype("int64")
            frame = frame.loc[frame["_time_block"].isin(block_ids)]
            if frame.empty:
                continue
            for block, group in frame.groupby("_time_block", sort=False):
                block = int(block)
                appender = appenders.setdefault(
                    block,
                    ParquetAppender(state_dir / f"states_{week}_{block}.parquet"),
                )
                appender.append(group.drop(columns=["_time_block"]))
                row_count += len(group)
            del frame
    finally:
        for appender in appenders.values():
            appender.close()
    return row_count


def prepare_partitions(args: argparse.Namespace, partition_root: Path, weeks: list[str]) -> dict:
    manifest_path = partition_root / "manifest.json"
    target_dir = partition_root / "targets"
    state_dir = partition_root / "states"
    block_seconds = args.time_block_hours * 3600
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("max_target_rows") != args.max_target_rows:
            raise RuntimeError("Checkpoint was created with a different --max-target-rows")
        return manifest
    clean_directory(target_dir)
    clean_directory(state_dir)
    manifest = {
        "max_target_rows": args.max_target_rows,
        "time_block_hours": args.time_block_hours,
        "weeks": {},
    }
    window_checkpoint_dir = args.intervention_windows.parent / "checkpoints"
    total_target_rows = 0
    for week in weeks:
        intervention_shard = window_checkpoint_dir / f"intervention_{week}.parquet"
        control_shard = window_checkpoint_dir / f"control_{week}.parquet"
        target_paths = [path for path in (intervention_shard, control_shard) if path.exists()]
        if not target_paths:
            target_paths = [args.intervention_windows, args.control_windows]
        remaining = max(0, args.max_target_rows - total_target_rows) if args.max_target_rows else 0
        block_ids, target_rows = partition_targets(
            target_paths,
            week,
            target_dir,
            block_seconds,
            args.input_batch_size,
            remaining,
        )
        total_target_rows += target_rows
        if block_ids:
            state_rows = partition_states(
                args.states_path,
                week,
                block_ids,
                state_dir,
                block_seconds,
                args.input_batch_size,
            )
        else:
            state_rows = 0
        manifest["weeks"][week] = {
            "blocks": sorted(block_ids),
            "target_rows": target_rows,
            "state_rows": state_rows,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Partitioned {week}: targets={target_rows}, states={state_rows}, blocks={len(block_ids)}",
            flush=True,
        )
        if args.max_target_rows and total_target_rows >= args.max_target_rows:
            break
    manifest["partitions_ready"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def decorate_pair(pair: pd.DataFrame, target) -> pd.DataFrame:
    metadata = {
        "window_id": target.window_id,
        "clearance_id": getattr(target, "clearance_id", ""),
        "control_for_clearance_id": getattr(target, "control_for_clearance_id", ""),
        "window_role": target.window_role,
        "event_timestamp": getattr(target, "event_timestamp", ""),
        "archive_week": getattr(target, "archive_week", ""),
        "focal_flight_key": target.flight_key,
        "focal_flight_id": getattr(target, "flight_id", ""),
        "focal_callsign": getattr(target, "callsign", ""),
        "focal_sector": getattr(target, "active_centre_id", ""),
        "focal_flight_phase": getattr(target, "flight_phase", ""),
        "ATC_INTERVENTION": getattr(target, "ATC_INTERVENTION", 0),
        "event_epoch": getattr(target, "event_epoch", np.nan),
        "time_epoch": target.time_epoch,
        "timestamp": getattr(target, "timestamp", ""),
        "relative_seconds": getattr(target, "relative_seconds", np.nan),
        "is_event_time": getattr(target, "is_event_time", False),
        "predictor_eligible": getattr(target, "predictor_eligible", False),
        "focal_altitude_ft": target.altitude_ft,
        "flight_level": getattr(target, "flight_level", np.nan),
        "focal_ground_speed_kt": getattr(target, "ground_speed_kt", np.nan),
        "focal_vertical_rate_fpm": getattr(target, "vertical_rate_fpm", np.nan),
        "focal_latitude": target.latitude,
        "focal_longitude": target.longitude,
    }
    pair = pair.copy()
    for column, value in metadata.items():
        pair[column] = value
    return pair


def update_streaming_trends(pair: pd.DataFrame, history: dict[tuple, list[tuple[float, float]]]) -> pd.DataFrame:
    """Add trend features while rows are processed in chronological order."""

    trend_values = {horizon: np.full(len(pair), np.nan, dtype=float) for horizon in HORIZONS}
    for position, row in pair.reset_index(drop=True).iterrows():
        key = (str(row["window_id"]), str(row["focal_flight_key"]), str(row["other_flight_key"]))
        timestamp = float(row["time_epoch"])
        separation = float(row["horizontal_separation_nm"])
        previous = history.setdefault(key, [])
        if previous:
            times = np.asarray([item[0] for item in previous] + [timestamp], dtype=float)
            values = np.asarray([item[1] for item in previous] + [separation], dtype=float)
            for horizon in HORIZONS:
                requested = timestamp - horizon
                if times[0] <= requested <= times[-1]:
                    trend_values[horizon][position] = separation - float(np.interp(requested, times, values))
        previous.append((timestamp, separation))
        cutoff = timestamp - max(HORIZONS)
        while previous and previous[0][0] < cutoff:
            previous.pop(0)
    for horizon, values in trend_values.items():
        pair[f"separation_trend_{horizon}s"] = values
    return pair


def process_target_block(
    target_path: Path,
    state_path: Path,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    checkpoint_path: Path,
    week: str,
    block: int,
    total_blocks: int,
    completed_blocks: set[str],
    history: dict[tuple, list[tuple[float, float]]],
) -> int:
    targets = pd.read_parquet(target_path)
    targets["time_epoch"] = pd.to_numeric(targets["time_epoch"], errors="coerce")
    targets.dropna(subset=["window_id", "flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"], inplace=True)
    targets.drop_duplicates(["window_id", "flight_key", "time_epoch"], inplace=True)
    targets.sort_values(["time_epoch", "window_id", "flight_key"], inplace=True, kind="mergesort")
    states = pd.read_parquet(state_path, columns=existing_columns(state_path, STATE_COLUMNS))
    for column in ("latitude", "longitude", "altitude_ft", "time_epoch"):
        states[column] = pd.to_numeric(states[column], errors="coerce")
    states.dropna(subset=["flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"], inplace=True)
    states["time_bucket_5s"] = np.rint(states["time_epoch"].to_numpy(dtype=float) / 5.0).astype("int64")
    states.sort_values(["time_bucket_5s", "flight_key"], inplace=True, kind="mergesort")
    states.reset_index(drop=True, inplace=True)
    bucket_values = states["time_bucket_5s"].to_numpy(dtype="int64")
    target_count = len(targets)
    shard_count = 0
    for chunk_start in range(0, target_count, args.target_chunk_size):
        chunk_end = min(chunk_start + args.target_chunk_size, target_count)
        feature_rows = []
        for target in targets.iloc[chunk_start:chunk_end].itertuples(index=False):
            bucket = int(round(float(target.time_epoch) / 5.0))
            left = int(np.searchsorted(bucket_values, bucket, side="left"))
            right = int(np.searchsorted(bucket_values, bucket, side="right"))
            if left == right:
                continue
            group = states.iloc[left:right]
            focal_rows = group.loc[group["flight_key"].astype(str) == str(target.flight_key)]
            if focal_rows.empty:
                continue
            others = group.loc[group["flight_key"].astype(str) != str(target.flight_key)].copy()
            if others.empty:
                continue
            pair = pair_features(
                focal_rows.iloc[0],
                others,
                args.nearby_radius_nm,
                max_neighbours=args.max_neighbours,
            )
            pair.sort_values("horizontal_separation_nm", inplace=True, kind="mergesort")
            pair = decorate_pair(pair, target)
            pair = update_streaming_trends(pair, history)
            feature_rows.append(pair)
        if feature_rows:
            feature_shard = checkpoint_dir / f"raw_features_{week}_{block}_{chunk_start:012d}_{chunk_end:012d}.parquet"
            pd.concat(feature_rows, ignore_index=True).to_parquet(feature_shard, index=False, compression="zstd")
            shard_count += 1
        checkpoint_path.write_text(
            json.dumps({
                "completed": False,
                "completed_blocks": sorted(completed_blocks),
                "current_week": week,
                "current_block": block,
                "processed_in_block": chunk_end,
                "total_in_block": target_count,
                "total_blocks": total_blocks,
                "target_chunk_size": args.target_chunk_size,
                "raw_feature_shards": len(list(checkpoint_dir.glob("raw_features_*.parquet"))),
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"Target chunk completed: {week} block {block} {chunk_end}/{target_count}",
            flush=True,
        )
        del feature_rows
        gc.collect()
    cutoff = (block + 1) * args.time_block_hours * 3600 - max(HORIZONS)
    for key in list(history):
        if not history[key] or history[key][-1][0] < cutoff:
            del history[key]
    del targets, states
    gc.collect()
    return shard_count


def merge_feature_shards(shards: list[Path], output_path: Path) -> dict:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = None
    quality = {
        "rows": 0,
        "windows": set(),
        "intervention_rows": 0,
        "control_rows": 0,
        "pairs_with_topsky_routes": 0,
        "missing_cpa": 0,
        "missing_tcpa": 0,
    }
    try:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for record_batch in parquet.iter_batches(batch_size=DEFAULT_INPUT_BATCH_SIZE):
                frame = record_batch.to_pandas()
                if "distance_to_sector_boundary_nm" not in frame:
                    frame["distance_to_sector_boundary_nm"] = np.nan
                quality["rows"] += len(frame)
                quality["windows"].update(frame["window_id"].astype(str).unique())
                quality["intervention_rows"] += int((pd.to_numeric(frame["ATC_INTERVENTION"], errors="coerce") == 1).sum())
                quality["control_rows"] += int((pd.to_numeric(frame["ATC_INTERVENTION"], errors="coerce") == 0).sum())
                quality["pairs_with_topsky_routes"] += int(frame["topsky_route_pair_available"].fillna(False).astype(bool).sum())
                quality["missing_cpa"] += int(frame["cpa_nm"].isna().sum())
                quality["missing_tcpa"] += int(frame["tcpa_min"].isna().sum())
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
                elif not table.schema.equals(writer.schema, check_metadata=False):
                    table = table.cast(writer.schema, safe=False)
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No feature shards were produced")
    temporary_path.replace(output_path)
    quality["windows"] = len(quality["windows"])
    return quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "02_traffic" / "scat_states.parquet")
    parser.add_argument("--intervention-windows", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows" / "intervention_windows.parquet")
    parser.add_argument("--control-windows", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows" / "control_windows.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=1_000,
        help="Focal window-state rows processed before writing a checkpoint shard.",
    )
    parser.add_argument("--max-neighbours", type=int, default=10)
    parser.add_argument("--nearby-radius-nm", type=float, default=30.0)
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=DEFAULT_INPUT_BATCH_SIZE,
        help="Rows read at a time while partitioning the large Parquet inputs.",
    )
    parser.add_argument(
        "--time-block-hours",
        type=int,
        default=DEFAULT_TIME_BLOCK_HOURS,
        help="Temporal partition size used to bound state and target memory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume pair generation from checkpoint shards.")
    parser.add_argument(
        "--stop-after-week",
        default="",
        help="Finalize the feature table after completing this archive week; useful for staged analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.target_chunk_size <= 0
        or args.max_neighbours <= 0
        or args.nearby_radius_nm <= 0
        or args.input_batch_size <= 0
        or args.time_block_hours <= 0
    ):
        raise SystemExit("chunk sizes, time-block-hours, max-neighbours and nearby-radius-nm must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "interaction_features.parquet"
    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_path = args.output_root / "checkpoint.json"
    partition_root = args.output_root / "input_partitions"
    if args.resume and args.overwrite:
        raise SystemExit("Use either --resume or --overwrite, not both")
    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}; use --overwrite")
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    if not args.resume:
        clean_directory(checkpoint_dir)
        clean_directory(partition_root / "targets")
        clean_directory(partition_root / "states")
        manifest = partition_root / "manifest.json"
        if manifest.exists():
            manifest.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        checkpoint = {"completed_blocks": []}
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not args.states_path.exists() or not args.intervention_windows.exists():
        raise FileNotFoundError("Run stages 02 and 04 before building interaction features")
    window_checkpoint_dir = args.intervention_windows.parent / "checkpoints"
    intervention_shards = sorted(window_checkpoint_dir.glob("intervention_*.parquet"))
    if intervention_shards:
        weeks = [path.stem.removeprefix("intervention_") for path in intervention_shards]
    else:
        raise FileNotFoundError("Stage 04 per-week checkpoint shards are required for memory-safe stage 05")
    stop_after_week = args.stop_after_week.strip()
    if stop_after_week and stop_after_week not in weeks:
        raise SystemExit(f"Unknown --stop-after-week {stop_after_week}; available weeks: {', '.join(weeks)}")
    partition_manifest = prepare_partitions(args, partition_root, weeks)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_blocks = set(checkpoint.get("completed_blocks", []))
    history: dict[tuple, list[tuple[float, float]]] = {}
    total_blocks = sum(len(info.get("blocks", [])) for info in partition_manifest["weeks"].values())
    completed_count = len(completed_blocks)
    stopped_after_week = False
    for week in weeks:
        week_info = partition_manifest["weeks"].get(week, {})
        for block in week_info.get("blocks", []):
            block_key = f"{week}:{block}"
            if block_key in completed_blocks:
                print(f"Block already completed: {block_key}", flush=True)
                continue
            raw_prefix = f"raw_features_{week}_{block}_"
            for shard in checkpoint_dir.glob(f"{raw_prefix}*.parquet"):
                shard.unlink()
            target_path = partition_root / "targets" / f"targets_{week}_{block}.parquet"
            state_path = partition_root / "states" / f"states_{week}_{block}.parquet"
            if not target_path.exists() or not state_path.exists():
                raise FileNotFoundError(f"Missing input partition for {block_key}")
            print(f"Processing time block {completed_count + 1}/{total_blocks}: {block_key}", flush=True)
            process_target_block(
                target_path,
                state_path,
                args,
                checkpoint_dir,
                checkpoint_path,
                week,
                int(block),
                total_blocks,
                completed_blocks,
                history,
            )
            completed_blocks.add(block_key)
            completed_count += 1
            checkpoint = {
                "completed": False,
                "completed_blocks": sorted(completed_blocks),
                "total_blocks": total_blocks,
                "target_chunk_size": args.target_chunk_size,
                "input_batch_size": args.input_batch_size,
                "time_block_hours": args.time_block_hours,
                "raw_feature_shards": len(list(checkpoint_dir.glob("raw_features_*.parquet"))),
            }
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            print(f"Time block completed: {completed_count}/{total_blocks} ({block_key})", flush=True)
            if stop_after_week == week and all(
                f"{week}:{week_block}" in completed_blocks
                for week_block in week_info.get("blocks", [])
            ):
                stopped_after_week = True
                print(
                    f"Archive week completed: {week}; finalizing scoped feature table and stopping.",
                    flush=True,
                )
                break
        if stopped_after_week:
            break

    feature_shards = sorted(checkpoint_dir.glob("raw_features_*.parquet"))
    if not feature_shards:
        raise RuntimeError("No focal/nearby pairs could be computed")
    quality = merge_feature_shards(feature_shards, output_path)
    dictionary = [
        ("horizontal_separation_nm", "Current horizontal separation in nautical miles."),
        ("vertical_separation_ft", "Absolute current vertical separation in feet."),
        ("relative_velocity_kt", "Magnitude of relative horizontal velocity."),
        ("horizontal_closing_rate_nm_min", "Positive values indicate decreasing horizontal separation."),
        ("vertical_closing_rate_fpm", "Positive values indicate decreasing vertical separation."),
        ("cpa_nm", "Kinematic horizontal closest-point distance."),
        ("tcpa_min", "Signed kinematic time to horizontal CPA; negative means CPA is behind the observation."),
        ("predicted_min_horizontal_sep_nm", "Future kinematic minimum horizontal separation using max(TCPA, 0)."),
        ("predicted_min_vertical_sep_ft", "Vertical separation at the future kinematic CPA time."),
        ("topsky_predicted_min_sep_nm_proxy", "Minimum time-consistent separation between published TopSky predicted points; proxy only."),
        ("separation_trend_30s/60s/120s", "Current separation minus interpolated separation that many seconds earlier; negative means closing."),
        ("traffic_density", "Number of tracked aircraft in the same 5-second surveillance bucket."),
        ("nearby_aircraft_count_30nm", "Number of other tracked aircraft within the configured 30 NM radius."),
        ("ATC_INTERVENTION", "1 for intervention windows and 0 for matched controls; never a conflict label."),
    ]
    pd.DataFrame(dictionary, columns=["feature", "definition"]).to_csv(args.output_root / "feature_dictionary.csv", index=False)
    config = {
        "states_path": str(args.states_path),
        "max_target_rows": args.max_target_rows,
        "target_chunk_size": args.target_chunk_size,
        "input_batch_size": args.input_batch_size,
        "time_block_hours": args.time_block_hours,
        "max_neighbours": args.max_neighbours,
        "nearby_radius_nm": args.nearby_radius_nm,
        "stop_after_week": stop_after_week or None,
        "processed_scope": stop_after_week or "all_weeks",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": str(checkpoint_dir),
        "time_alignment": "nearest native states within a common 5-second bucket; no interpolation of positions",
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_root / "feature_quality_report.md").write_text(
        "# SCAT interaction feature quality\n\n" + "\n".join(f"- {key}: **{value}**" for key, value in quality.items()) + "\n\n"
        "All separation, CPA/TCPA and closing-rate variables remain continuous. No 5 NM/1000 ft rule, conflict target or OpenSky colour state is applied.\n",
        encoding="utf-8",
    )
    checkpoint_path.write_text(
        json.dumps({
            "completed": True,
            "completed_blocks": sorted(completed_blocks),
            "total_blocks": total_blocks,
            "target_chunk_size": args.target_chunk_size,
            "input_batch_size": args.input_batch_size,
            "time_block_hours": args.time_block_hours,
            "stop_after_week": stop_after_week or None,
            "processed_scope": stop_after_week or "all_weeks",
            "raw_feature_shards": len(feature_shards),
        }, indent=2),
        encoding="utf-8",
    )
    update_read_me_first(args.output_root.parent)
    print(f"Interaction features complete: {args.output_root}")


if __name__ == "__main__":
    main()

