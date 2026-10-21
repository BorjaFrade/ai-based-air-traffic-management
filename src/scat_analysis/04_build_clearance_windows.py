#!/usr/bin/env python3
"""Build intervention and matched-control windows without predictor leakage."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from common import DEFAULT_OUTPUT_ROOT, update_read_me_first


WINDOW_SECONDS = 600
DEFAULT_MAX_CONTROL_TIME_DIFFERENCE_S = 3600.0
LOCAL_RADIUS_NM = 30.0
DEFAULT_STATE_BATCH_SIZE = 50_000
DEFAULT_OUTPUT_CHUNK_ROWS = 100_000
STATE_COLUMNS = [
    "flight_key", "flight_id", "archive_week", "source_archive", "source_member", "plot_index",
    "timestamp", "time_epoch", "callsign", "latitude", "longitude", "altitude_ft", "flight_level",
    "selected_altitude_ft", "ground_speed_ms", "ground_speed_kt", "track_deg", "heading_deg",
    "vertical_rate_fpm", "vertical_rate_ms", "flight_phase", "fpl_aircraft_type", "fpl_wtc",
    "fpl_flight_rules", "fpl_route", "fpl_rfl", "fpl_tas", "fpl_copn", "fpl_copx",
    "clearance_cfl", "clearance_cfl_unit", "clearance_heading_deg", "clearance_heading_beacon",
    "clearance_speed", "clearance_speed_unit", "active_centre_id", "predicted_next_fix",
    "predicted_next_fix_eta", "predicted_route", "predicted_route_json",
]
MATCH_STATE_COLUMNS = [
    "flight_key", "archive_week", "timestamp", "time_epoch", "callsign",
    "latitude", "longitude", "altitude_ft", "flight_level", "flight_phase",
    "fpl_aircraft_type", "active_centre_id",
]


def existing_columns(path: Path, wanted: list[str]) -> list[str]:
    available = set(pq.read_schema(path).names)
    return [column for column in wanted if column in available]


def week_row_groups(path: Path, week: str) -> tuple[pq.ParquetFile, list[int]]:
    """Return only row groups whose min/max archive-week statistics match."""

    parquet = pq.ParquetFile(path)
    archive_index = parquet.schema.names.index("archive_week")
    row_groups = []
    for index in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(index).column(archive_index).statistics
        if statistics is None or statistics.min <= week <= statistics.max:
            row_groups.append(index)
    if not row_groups:
        raise RuntimeError(f"No Parquet row groups found for archive week {week}")
    return parquet, row_groups


def iter_state_batches(
    path: Path,
    week: str,
    columns: list[str],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    parquet, row_groups = week_row_groups(path, week)
    selected_columns = existing_columns(path, columns)
    for record_batch in parquet.iter_batches(
        batch_size=batch_size,
        row_groups=row_groups,
        columns=selected_columns,
        use_threads=True,
    ):
        frame = record_batch.to_pandas()
        if not frame.empty:
            yield frame


def density_bins(values: pd.Series) -> pd.Series:
    # These are local-traffic matching strata, not safety thresholds.
    return pd.cut(
        values,
        bins=[-np.inf, 0, 1, 3, 6, 10, np.inf],
        labels=["0", "1", "2_3", "4_6", "7_10", "11_plus"],
    ).astype("string")


def add_local_density(states: pd.DataFrame, radius_nm: float = LOCAL_RADIUS_NM) -> pd.DataFrame:
    """Count other aircraft within ``radius_nm`` in each common 5 s bucket.

    A small latitude/longitude grid avoids an all-pairs matrix for busy
    buckets.  The neighbouring cells cover the configured radius, and the
    final haversine-like local planar distance is still checked exactly in NM.
    """

    states = states.reset_index(drop=True)
    time_values = pd.to_numeric(states["time_epoch"], errors="coerce").to_numpy(dtype=float)
    buckets = np.rint(time_values / 5.0).astype("int64")
    counts = np.zeros(len(states), dtype=np.int32)
    order = np.argsort(buckets, kind="mergesort")
    sorted_buckets = buckets[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_buckets)) + 1]
    ends = np.r_[starts[1:], len(order)]
    latitudes = pd.to_numeric(states["latitude"], errors="coerce").to_numpy(dtype=float)
    longitudes = pd.to_numeric(states["longitude"], errors="coerce").to_numpy(dtype=float)
    for start, end in zip(starts, ends):
        positions = order[start:end]
        if len(positions) <= 1:
            continue
        lat = latitudes[positions]
        lon = longitudes[positions]
        finite = np.isfinite(lat) & np.isfinite(lon)
        if finite.sum() <= 1:
            continue
        valid_positions = positions[finite]
        lat = lat[finite]
        lon = lon[finite]
        mean_lat = np.radians(float(np.nanmean(lat)))
        coordinates = np.column_stack((lat * 60.0, lon * 60.0 * np.cos(mean_lat)))
        # ``return_length`` avoids materialising a Python list of neighbour
        # indices for every aircraft in every time bucket.
        counts[valid_positions] = (
            cKDTree(coordinates).query_ball_point(
                coordinates, r=radius_nm, return_length=True, workers=-1
            ).astype(np.int32) - 1
        )
    states["time_bucket_5s"] = buckets
    states["nearby_aircraft_count_30nm"] = counts
    states["local_traffic_density"] = states["nearby_aircraft_count_30nm"].astype(float)
    states["local_traffic_density_bin"] = density_bins(states["local_traffic_density"])
    return states


def add_matching_columns(states: pd.DataFrame) -> pd.DataFrame:
    states = states.copy()
    states["time_epoch"] = pd.to_numeric(states["time_epoch"], errors="coerce")
    states["flight_level"] = pd.to_numeric(states.get("flight_level"), errors="coerce")
    states["flight_level_bin"] = (states["flight_level"] // 10 * 10).astype("Float64")
    timestamps = pd.to_datetime(states["timestamp"], utc=True, errors="coerce")
    states["hour_utc"] = timestamps.dt.hour.astype("Int64")
    aircraft_type = states["fpl_aircraft_type"] if "fpl_aircraft_type" in states else pd.Series("", index=states.index)
    centre = states["active_centre_id"] if "active_centre_id" in states else pd.Series("", index=states.index)
    phase = states["flight_phase"] if "flight_phase" in states else pd.Series("UNKNOWN", index=states.index)
    states["aircraft_type_match"] = aircraft_type.fillna("").astype(str)
    states["centre_match"] = centre.fillna("").astype(str)
    states["phase_match"] = phase.fillna("UNKNOWN").astype(str)
    states["archive_week_match"] = states["archive_week"].fillna("").astype(str)
    return states


def clearance_intervals(clearances: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, group in clearances.groupby("flight_key", sort=False):
        result[key] = np.sort(group["clearance_epoch"].to_numpy(dtype=float))
    return result


def mark_clearance_free(states: pd.DataFrame, clearances: pd.DataFrame) -> pd.Series:
    intervals = clearance_intervals(clearances)
    eligible = np.ones(len(states), dtype=bool)
    for key, indices in states.groupby("flight_key", sort=False).groups.items():
        times = states.loc[indices, "time_epoch"].to_numpy(dtype=float)
        blocked = np.zeros(len(times), dtype=bool)
        for event_time in intervals.get(key, np.array([], dtype=float)):
            left = np.searchsorted(times, event_time - WINDOW_SECONDS, side="left")
            right = np.searchsorted(times, event_time + WINDOW_SECONDS, side="right")
            blocked[left:right] = True
        eligible[np.asarray(list(indices), dtype=int)] = ~blocked
    return pd.Series(eligible, index=states.index)


def row_value(row, key: str, default=""):
    if hasattr(row, "_asdict"):
        return getattr(row, key, default)
    if isinstance(row, dict):
        return row.get(key, default)
    return row.get(key, default)


def match_key(row, include_type=True, include_density=True):
    values = [
        str(row_value(row, "archive_week_match", "")),
        str(row_value(row, "centre_match", "")),
        str(row_value(row, "phase_match", "UNKNOWN")),
        str(row_value(row, "flight_level_bin", "")),
        str(row_value(row, "hour_utc", "")),
    ]
    if include_density:
        values.append(str(row_value(row, "local_traffic_density_bin", "")))
    if include_type:
        values.append(str(row_value(row, "aircraft_type_match", "")))
    return tuple(values)


def control_match(
    intervention,
    candidate_groups: dict[tuple, list[int]],
    candidate_records: list[dict],
    used: set[tuple[str, float]],
    rng: np.random.Generator,
    max_time_difference_s: float,
):
    levels = [
        ("exact", True, True),
        ("without_aircraft_type", False, True),
        ("without_density", True, False),
        ("centre_phase_level_hour", False, False),
    ]
    for level, include_type, include_density in levels:
        key = match_key(intervention, include_type=include_type, include_density=include_density)
        indices = candidate_groups.get((include_type, include_density, key), [])
        if not indices:
            continue
        candidates = []
        for state_index in indices:
            candidate = candidate_records[state_index]
            identity = (str(candidate["flight_key"]), float(candidate["time_epoch"]))
            if identity in used:
                continue
            if identity[0] == str(intervention.flight_key):
                continue
            time_difference = abs(float(candidate["time_epoch"]) - float(intervention.clearance_epoch))
            if time_difference > max_time_difference_s:
                continue
            candidates.append((time_difference, float(candidate["time_epoch"]), state_index, candidate))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], str(item[3]["flight_key"])))
            best_distance = candidates[0][0]
            tied = [item for item in candidates if item[0] == best_distance]
            selected = tied[int(rng.integers(len(tied)))]
            identity = (str(selected[3]["flight_key"]), float(selected[3]["time_epoch"]))
            used.add(identity)
            relaxations = []
            if not include_type:
                relaxations.append("aircraft_type")
            if not include_density:
                relaxations.append("local_density")
            return selected[3], level, len(indices), "+".join(relaxations) or "none"
    return None, "unmatched", 0, "no_valid_candidate"


def build_matching_balance(quality: pd.DataFrame) -> pd.DataFrame:
    """Summarise covariate balance for matched intervention/control anchors."""

    columns = [
        "covariate", "covariate_type", "category", "intervention_n", "control_n",
        "intervention_value", "control_value", "absolute_difference",
        "standardized_mean_difference",
    ]
    if quality.empty or "matched" not in quality:
        return pd.DataFrame(columns=columns)
    matched = quality.loc[quality["matched"].astype(bool)].copy()
    if matched.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    numeric = [
        ("flight_level", "intervention_flight_level", "control_flight_level"),
        ("local_traffic_density", "intervention_local_traffic_density", "control_local_traffic_density"),
        ("hour_utc", "intervention_hour_utc", "control_hour_utc"),
    ]
    for name, left, right in numeric:
        if left not in matched or right not in matched:
            continue
        intervention = pd.to_numeric(matched[left], errors="coerce").dropna()
        control = pd.to_numeric(matched[right], errors="coerce").dropna()
        if intervention.empty or control.empty:
            continue
        left_mean = float(intervention.mean())
        right_mean = float(control.mean())
        pooled = float(np.sqrt((intervention.var(ddof=1) + control.var(ddof=1)) / 2.0))
        smd = (left_mean - right_mean) / pooled if pooled > 0 else (0.0 if left_mean == right_mean else np.inf)
        rows.append({
            "covariate": name,
            "covariate_type": "numeric",
            "category": "",
            "intervention_n": len(intervention),
            "control_n": len(control),
            "intervention_value": left_mean,
            "control_value": right_mean,
            "absolute_difference": abs(left_mean - right_mean),
            "standardized_mean_difference": smd,
        })
    categorical = [
        ("archive_week", "intervention_archive_week", "control_archive_week"),
        ("centre", "intervention_centre", "control_centre"),
        ("phase", "intervention_phase", "control_phase"),
        ("aircraft_type", "intervention_aircraft_type", "control_aircraft_type"),
        ("local_density_bin", "intervention_local_density_bin", "control_local_density_bin"),
    ]
    for name, left, right in categorical:
        if left not in matched or right not in matched:
            continue
        left_values = matched[left].fillna("").astype(str)
        right_values = matched[right].fillna("").astype(str)
        categories = sorted(set(left_values) | set(right_values))
        for category in categories:
            left_prop = float((left_values == category).mean())
            right_prop = float((right_values == category).mean())
            rows.append({
                "covariate": name,
                "covariate_type": "categorical",
                "category": category,
                "intervention_n": int((left_values == category).sum()),
                "control_n": int((right_values == category).sum()),
                "intervention_value": left_prop,
                "control_value": right_prop,
                "absolute_difference": abs(left_prop - right_prop),
                "standardized_mean_difference": np.nan,
            })
    return pd.DataFrame(rows, columns=columns)


def build_matching_balance_from_csv(path: Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Build the balance table without loading the full quality CSV."""

    columns = [
        "covariate", "covariate_type", "category", "intervention_n", "control_n",
        "intervention_value", "control_value", "absolute_difference",
        "standardized_mean_difference",
    ]
    numeric_names = [
        ("flight_level", "intervention_flight_level", "control_flight_level"),
        ("local_traffic_density", "intervention_local_traffic_density", "control_local_traffic_density"),
        ("hour_utc", "intervention_hour_utc", "control_hour_utc"),
    ]
    categorical_names = [
        ("archive_week", "intervention_archive_week", "control_archive_week"),
        ("centre", "intervention_centre", "control_centre"),
        ("phase", "intervention_phase", "control_phase"),
        ("aircraft_type", "intervention_aircraft_type", "control_aircraft_type"),
        ("local_density_bin", "intervention_local_density_bin", "control_local_density_bin"),
    ]
    numeric_stats = {
        name: {"intervention": [0, 0.0, 0.0], "control": [0, 0.0, 0.0]}
        for name, _, _ in numeric_names
    }
    categorical_counts = {
        name: {"intervention": Counter(), "control": Counter()}
        for name, _, _ in categorical_names
    }
    matched_total = 0
    if not path.exists():
        return pd.DataFrame(columns=columns)
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if "matched" not in chunk:
            continue
        matched_mask = chunk["matched"].astype(str).str.lower().isin({"true", "1", "yes"})
        matched = chunk.loc[matched_mask]
        if matched.empty:
            continue
        matched_total += len(matched)
        for name, left, right in numeric_names:
            for group, source in (("intervention", left), ("control", right)):
                if source not in matched:
                    continue
                values = pd.to_numeric(matched[source], errors="coerce").dropna().to_numpy(dtype=float)
                if values.size:
                    count, total, squares = numeric_stats[name][group]
                    numeric_stats[name][group] = [
                        count + int(values.size),
                        total + float(values.sum()),
                        squares + float(np.square(values).sum()),
                    ]
        for name, left, right in categorical_names:
            for group, source in (("intervention", left), ("control", right)):
                if source in matched:
                    categorical_counts[name][group].update(matched[source].fillna("").astype(str))

    rows = []
    for name, _, _ in numeric_names:
        intervention_n, intervention_sum, intervention_squares = numeric_stats[name]["intervention"]
        control_n, control_sum, control_squares = numeric_stats[name]["control"]
        if not intervention_n or not control_n:
            continue
        intervention_mean = intervention_sum / intervention_n
        control_mean = control_sum / control_n
        intervention_var = (
            max(0.0, (intervention_squares - intervention_sum ** 2 / intervention_n) / (intervention_n - 1))
            if intervention_n > 1 else 0.0
        )
        control_var = (
            max(0.0, (control_squares - control_sum ** 2 / control_n) / (control_n - 1))
            if control_n > 1 else 0.0
        )
        pooled = float(np.sqrt((intervention_var + control_var) / 2.0))
        smd = ((intervention_mean - control_mean) / pooled
               if pooled > 0 else (0.0 if intervention_mean == control_mean else np.inf))
        rows.append({
            "covariate": name,
            "covariate_type": "numeric",
            "category": "",
            "intervention_n": intervention_n,
            "control_n": control_n,
            "intervention_value": intervention_mean,
            "control_value": control_mean,
            "absolute_difference": abs(intervention_mean - control_mean),
            "standardized_mean_difference": smd,
        })
    for name, _, _ in categorical_names:
        left_counts = categorical_counts[name]["intervention"]
        right_counts = categorical_counts[name]["control"]
        for category in sorted(set(left_counts) | set(right_counts)):
            left_n = int(left_counts.get(category, 0))
            right_n = int(right_counts.get(category, 0))
            left_prop = left_n / matched_total if matched_total else 0.0
            right_prop = right_n / matched_total if matched_total else 0.0
            rows.append({
                "covariate": name,
                "covariate_type": "categorical",
                "category": category,
                "intervention_n": left_n,
                "control_n": right_n,
                "intervention_value": left_prop,
                "control_value": right_prop,
                "absolute_difference": abs(left_prop - right_prop),
                "standardized_mean_difference": np.nan,
            })
    return pd.DataFrame(rows, columns=columns)


def make_window_rows(
    states: pd.DataFrame,
    flight_slices: dict[str, tuple[int, int]],
    anchor: dict,
    event_meta: dict,
    role: str,
) -> list[dict]:
    bounds = flight_slices.get(str(anchor["flight_key"]))
    if bounds is None:
        return []
    group = states.iloc[bounds[0]:bounds[1]]
    if group.empty:
        return []
    event_epoch = float(event_meta["event_epoch"])
    mask = group["time_epoch"].between(event_epoch - WINDOW_SECONDS, event_epoch + WINDOW_SECONDS)
    selected = group.loc[mask]
    rows = []
    for _, state in selected.iterrows():
        row = state.to_dict()
        row.update(
            {
                "window_id": event_meta["window_id"],
                "clearance_id": event_meta.get("clearance_id", ""),
                "control_for_clearance_id": event_meta.get("control_for_clearance_id", ""),
                "window_role": role,
                "ATC_INTERVENTION": 1 if role == "intervention" else 0,
                "event_epoch": event_epoch,
                "event_timestamp": event_meta["event_timestamp"],
                "relative_seconds": float(state["time_epoch"]) - event_epoch,
                "is_event_time": abs(float(state["time_epoch"]) - event_epoch) <= 5,
                "predictor_eligible": float(state["time_epoch"]) <= event_epoch,
                "target_name": "ATC_INTERVENTION",
            }
        )
        rows.append(row)
    return rows


def build_partition(
    clearances: pd.DataFrame,
    all_clearances: pd.DataFrame,
    states: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Build windows for one archive week without loading other weeks."""

    states = add_matching_columns(states)
    states = states.dropna(subset=["flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"]).copy()
    states.sort_values(["flight_key", "time_epoch"], inplace=True, kind="mergesort")
    # Density is needed only at possible control anchors and at the
    # previous-surveillance state of an intervention.  The windows themselves
    # still retain every native state row.
    states["_time_bucket_5s"] = (states["time_epoch"] / 5.0).round().astype("int64")
    event_buckets = set(
        (pd.to_numeric(clearances.get("matched_state_epoch"), errors="coerce") / 5.0)
        .round()
        .dropna()
        .astype("int64")
    )
    stride_buckets = ((states["_time_bucket_5s"] * 5) % max(args.control_stride_s, 1)) == 0
    density_mask = stride_buckets | states["_time_bucket_5s"].isin(event_buckets)
    density_states = states.loc[density_mask].copy()
    density_states["_original_state_index"] = density_states.index
    if not density_states.empty:
        density_states = add_local_density(density_states, radius_nm=args.local_radius_nm)
        for column in ("nearby_aircraft_count_30nm", "local_traffic_density", "local_traffic_density_bin"):
            states[column] = np.nan if column != "local_traffic_density_bin" else ""
            states.loc[
                density_states["_original_state_index"].to_numpy(dtype=int), column
            ] = density_states[column].to_numpy()
    else:
        states["nearby_aircraft_count_30nm"] = 0
        states["local_traffic_density"] = 0.0
        states["local_traffic_density_bin"] = density_bins(states["local_traffic_density"])
    states.drop(columns=["_time_bucket_5s"], inplace=True)
    states.reset_index(drop=True, inplace=True)
    flight_slices = {}
    for key, indices in states.groupby("flight_key", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        if positions.size:
            flight_slices[str(key)] = (int(positions[0]), int(positions[-1]) + 1)

    matched_state_rows = []
    for row in clearances.itertuples(index=False):
        bounds = flight_slices.get(str(row.flight_key))
        if bounds is None:
            matched_state_rows.append({})
            continue
        group = states.iloc[bounds[0]:bounds[1]]
        times = group["time_epoch"].to_numpy(dtype=float)
        idx = int(np.searchsorted(times, row.clearance_epoch, side="right")) - 1
        if idx < 0:
            matched_state_rows.append({})
            continue
        assert times[idx] <= row.clearance_epoch
        matched_state_rows.append(group.iloc[idx].to_dict())
    matched = pd.DataFrame(matched_state_rows)
    for column in (
        "archive_week_match", "centre_match", "phase_match", "flight_level", "flight_level_bin", "hour_utc",
        "local_traffic_density", "local_traffic_density_bin", "aircraft_type_match",
    ):
        clearances[column] = matched[column].to_numpy() if column in matched else ""
    clearances["event_timestamp"] = clearances["clearance_timestamp"].astype(str)

    eligible = mark_clearance_free(states, all_clearances)
    states["clearance_free"] = eligible.to_numpy()
    candidates = states.loc[states["clearance_free"]].copy()
    if args.control_stride_s > 1:
        candidates = candidates.loc[
            (candidates["time_epoch"].round().astype("int64") % args.control_stride_s) == 0
        ]
    candidate_groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in candidates.iterrows():
        for include_type, include_density in ((True, True), (False, True), (True, False), (False, False)):
            candidate_groups[match_key(row, include_type, include_density)].append(index)

    used_controls: set[tuple[str, float]] = set()
    intervention_rows = []
    control_rows = []
    quality_rows = []
    for index, intervention in enumerate(clearances.itertuples(index=False), start=1):
        intervention_meta = {
            "window_id": f"I:{intervention.clearance_id}",
            "clearance_id": intervention.clearance_id,
            "event_epoch": intervention.clearance_epoch,
            "event_timestamp": intervention.event_timestamp,
        }
        intervention_rows.extend(
            make_window_rows(states, flight_slices, {"flight_key": intervention.flight_key}, intervention_meta, "intervention")
        )
        control, level, candidate_count, relaxations = control_match(
            intervention, candidate_groups, states, used_controls, rng,
            args.max_control_time_difference_s,
        )
        control_time_difference = (
            abs(float(control["time_epoch"]) - float(intervention.clearance_epoch))
            if control is not None else np.nan
        )
        quality_rows.append({
            "clearance_id": intervention.clearance_id,
            "intervention_window_id": intervention_meta["window_id"],
            "matched": control is not None,
            "matching_level": level,
            "matching_relaxations": relaxations,
            "candidate_count": candidate_count,
            "intervention_flight_key": intervention.flight_key,
            "intervention_epoch": intervention.clearance_epoch,
            "control_flight_key": control.get("flight_key", "") if control is not None else "",
            "control_epoch": control.get("time_epoch", np.nan) if control is not None else np.nan,
            "control_time_difference_s": control_time_difference,
            "intervention_archive_week": str(getattr(intervention, "archive_week_match", "")),
            "control_archive_week": str(control.get("archive_week_match", "")) if control is not None else "",
            "intervention_centre": str(getattr(intervention, "centre_match", "")),
            "control_centre": str(control.get("centre_match", "")) if control is not None else "",
            "intervention_phase": str(getattr(intervention, "phase_match", "")),
            "control_phase": str(control.get("phase_match", "")) if control is not None else "",
            "intervention_aircraft_type": str(getattr(intervention, "aircraft_type_match", "")),
            "control_aircraft_type": str(control.get("aircraft_type_match", "")) if control is not None else "",
            "intervention_flight_level": getattr(intervention, "flight_level", np.nan),
            "control_flight_level": control.get("flight_level", np.nan) if control is not None else np.nan,
            "intervention_hour_utc": getattr(intervention, "hour_utc", np.nan),
            "control_hour_utc": control.get("hour_utc", np.nan) if control is not None else np.nan,
            "intervention_local_traffic_density": getattr(intervention, "local_traffic_density", np.nan),
            "control_local_traffic_density": control.get("local_traffic_density", np.nan) if control is not None else np.nan,
            "intervention_local_density_bin": str(getattr(intervention, "local_traffic_density_bin", "")),
            "control_local_density_bin": str(control.get("local_traffic_density_bin", "")) if control is not None else "",
        })
        if control is not None:
            control_meta = {
                "window_id": f"C:{intervention.clearance_id}",
                "control_for_clearance_id": intervention.clearance_id,
                "event_epoch": float(control["time_epoch"]),
                "event_timestamp": str(control.get("timestamp", "")),
            }
            control_rows.extend(
                make_window_rows(states, flight_slices, {"flight_key": control["flight_key"]}, control_meta, "control")
            )
        if index % 1000 == 0:
            print(f"Processed {index}/{len(clearances)} interventions in {clearances['archive_week'].iloc[0]}", flush=True)
        if index % args.event_chunk_size == 0:
            yield pd.DataFrame(intervention_rows), pd.DataFrame(control_rows), pd.DataFrame(quality_rows)
            intervention_rows = []
            control_rows = []
            quality_rows = []
            gc.collect()
    if intervention_rows or control_rows or quality_rows:
        yield pd.DataFrame(intervention_rows), pd.DataFrame(control_rows), pd.DataFrame(quality_rows)


def build_candidate_groups(candidates: pd.DataFrame) -> dict[tuple[bool, bool, tuple], list[int]]:
    """Index candidates by the exact and relaxed matching strata."""

    base_columns = [
        "archive_week_match", "centre_match", "phase_match", "flight_level_bin", "hour_utc",
    ]
    groups: dict[tuple[bool, bool, tuple], list[int]] = {}
    for include_type, include_density in (
        (True, True), (False, True), (True, False), (False, False)
    ):
        columns = list(base_columns)
        if include_density:
            columns.append("local_traffic_density_bin")
        if include_type:
            columns.append("aircraft_type_match")
        for key, indices in candidates.groupby(columns, dropna=False, sort=False).groups.items():
            if not isinstance(key, tuple):
                key = (key,)
            normalised = tuple(str(value) for value in key)
            groups[(include_type, include_density, normalised)] = list(indices)
    return groups


def decorate_window_frame(
    selected: pd.DataFrame,
    event_meta: dict,
    role: str,
) -> pd.DataFrame:
    # Arrow-backed string columns can fail when pandas inserts scalar columns
    # one by one (notably on later Parquet batches).  Build the metadata block
    # separately and concatenate it once, keeping the streamed state frame
    # untouched and avoiding the ``KeyError: clearance_id`` insertion bug.
    selected = selected.copy().reset_index(drop=True)
    event_epoch = float(event_meta["event_epoch"])
    relative_seconds = pd.to_numeric(selected["time_epoch"], errors="coerce") - event_epoch
    metadata = pd.DataFrame(
        {
            "window_id": [event_meta["window_id"]] * len(selected),
            "clearance_id": [event_meta.get("clearance_id", "")] * len(selected),
            "control_for_clearance_id": [event_meta.get("control_for_clearance_id", "")] * len(selected),
            "window_role": [role] * len(selected),
            "ATC_INTERVENTION": [1 if role == "intervention" else 0] * len(selected),
            "event_epoch": [event_epoch] * len(selected),
            "event_timestamp": [event_meta.get("event_timestamp", "")] * len(selected),
            "relative_seconds": relative_seconds.to_numpy(),
            "is_event_time": (relative_seconds.abs() <= 5).to_numpy(),
            "predictor_eligible": (relative_seconds <= 0).to_numpy(),
            "target_name": ["ATC_INTERVENTION"] * len(selected),
        }
    )
    return pd.concat([selected, metadata], axis=1)


def iter_window_frames(
    states_path: Path,
    week: str,
    state_columns: list[str],
    intervention_meta_by_flight: dict[str, list[dict]],
    control_meta_by_flight: dict[str, list[dict]],
    batch_size: int,
    output_chunk_rows: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Stream full state rows and emit only the selected intervention/control windows."""

    intervention_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    intervention_rows = 0
    control_rows = 0
    required = ["flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"]
    for raw in iter_state_batches(states_path, week, state_columns, batch_size):
        raw = raw.dropna(subset=[column for column in required if column in raw]).copy()
        if raw.empty:
            continue
        raw["time_epoch"] = pd.to_numeric(raw["time_epoch"], errors="coerce")
        raw = raw.dropna(subset=["time_epoch"])
        if raw.empty:
            continue
        for flight_key, group in raw.groupby("flight_key", sort=False):
            key = str(flight_key)
            intervention_meta = intervention_meta_by_flight.get(key, [])
            control_meta = control_meta_by_flight.get(key, [])
            if not intervention_meta and not control_meta:
                continue
            group = group.sort_values("time_epoch", kind="mergesort")
            times = group["time_epoch"].to_numpy(dtype=float)
            for event_meta in intervention_meta:
                event_epoch = float(event_meta["event_epoch"])
                selected = group.loc[(times >= event_epoch - WINDOW_SECONDS) & (times <= event_epoch + WINDOW_SECONDS)]
                if not selected.empty:
                    frame = decorate_window_frame(selected, event_meta, "intervention")
                    intervention_frames.append(frame)
                    intervention_rows += len(frame)
            for event_meta in control_meta:
                event_epoch = float(event_meta["event_epoch"])
                selected = group.loc[(times >= event_epoch - WINDOW_SECONDS) & (times <= event_epoch + WINDOW_SECONDS)]
                if not selected.empty:
                    frame = decorate_window_frame(selected, event_meta, "control")
                    control_frames.append(frame)
                    control_rows += len(frame)
        if intervention_rows >= output_chunk_rows or control_rows >= output_chunk_rows:
            yield (
                pd.concat(intervention_frames, ignore_index=True) if intervention_frames else pd.DataFrame(),
                pd.concat(control_frames, ignore_index=True) if control_frames else pd.DataFrame(),
            )
            intervention_frames = []
            control_frames = []
            intervention_rows = 0
            control_rows = 0
            gc.collect()
    if intervention_frames or control_frames:
        yield (
            pd.concat(intervention_frames, ignore_index=True) if intervention_frames else pd.DataFrame(),
            pd.concat(control_frames, ignore_index=True) if control_frames else pd.DataFrame(),
        )


def build_partition_streaming(
    clearances: pd.DataFrame,
    all_clearances: pd.DataFrame,
    states_path: Path,
    week: str,
    state_columns: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, list[dict]], dict[str, list[dict]]]:
    """Find anchors from minimal state columns without loading a whole week."""

    event_lookup = clearances[
        ["clearance_id", "flight_key", "matched_state_epoch", "clearance_epoch"]
    ].copy()
    event_lookup["matched_state_epoch"] = pd.to_numeric(
        event_lookup["matched_state_epoch"], errors="coerce"
    )
    event_lookup = event_lookup.dropna(subset=["matched_state_epoch"])
    event_lookup["_time_bucket_5s"] = np.rint(
        event_lookup["matched_state_epoch"].to_numpy(dtype=float) / 5.0
    ).astype("int64")
    event_buckets = np.unique(event_lookup["_time_bucket_5s"].to_numpy(dtype="int64"))
    match_columns = existing_columns(states_path, MATCH_STATE_COLUMNS)
    matching_frames = []
    for raw in iter_state_batches(states_path, week, match_columns, args.state_batch_size):
        required = ["flight_key", "time_epoch", "latitude", "longitude", "altitude_ft"]
        raw = raw.dropna(subset=[column for column in required if column in raw]).copy()
        if raw.empty:
            continue
        times = pd.to_numeric(raw["time_epoch"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times)
        if not valid.any():
            continue
        raw = raw.loc[valid].reset_index(drop=True)
        times = times[valid]
        buckets = np.rint(times / 5.0).astype("int64")
        stride_mask = ((buckets * 5) % max(args.control_stride_s, 1)) == 0
        selection = stride_mask | np.isin(buckets, event_buckets)
        if not selection.any():
            continue
        selected = raw.loc[selection].copy()
        selected["time_epoch"] = times[selection]
        selected["_time_bucket_5s"] = buckets[selection]
        selected["_is_candidate"] = stride_mask[selection]
        matching_frames.append(selected)
        del raw, selected
    if not matching_frames:
        raise RuntimeError(f"No matching state rows found for archive week {week}")
    anchors = pd.concat(matching_frames, ignore_index=True)
    del matching_frames
    anchors = add_matching_columns(anchors)
    anchors = add_local_density(anchors, radius_nm=args.local_radius_nm)

    anchors["_event_epoch_key"] = np.rint(
        pd.to_numeric(anchors["time_epoch"], errors="coerce").to_numpy(dtype=float) * 1000.0
    ).astype("int64")
    event_lookup["_event_epoch_key"] = np.rint(
        event_lookup["matched_state_epoch"].to_numpy(dtype=float) * 1000.0
    ).astype("int64")
    event_candidates = anchors.merge(
        event_lookup,
        left_on=["flight_key", "_event_epoch_key"],
        right_on=["flight_key", "_event_epoch_key"],
        how="inner",
        suffixes=("", "_event"),
    )
    event_candidates["_event_time_difference"] = 0.0
    missing_ids = event_lookup.loc[
        ~event_lookup["clearance_id"].isin(event_candidates["clearance_id"]),
        "clearance_id",
    ]
    if not missing_ids.empty:
        fallback_lookup = event_lookup.loc[event_lookup["clearance_id"].isin(missing_ids)]
        fallback_candidates = anchors.merge(
            fallback_lookup,
            left_on=["flight_key", "_time_bucket_5s"],
            right_on=["flight_key", "_time_bucket_5s"],
            how="inner",
            suffixes=("", "_event"),
        )
        fallback_candidates["_event_time_difference"] = (
            pd.to_numeric(fallback_candidates["time_epoch"], errors="coerce")
            - pd.to_numeric(fallback_candidates["matched_state_epoch"], errors="coerce")
        ).abs()
        event_candidates = pd.concat([event_candidates, fallback_candidates], ignore_index=True)
    event_rows = (
        event_candidates.sort_values(
            ["clearance_id", "_event_time_difference", "time_epoch"],
            kind="mergesort",
        )
        .drop_duplicates("clearance_id", keep="first")
    )
    event_index = event_rows.set_index("clearance_id")
    matching_columns = [
        "archive_week_match", "centre_match", "phase_match", "flight_level",
        "flight_level_bin", "hour_utc", "local_traffic_density",
        "local_traffic_density_bin", "aircraft_type_match",
    ]
    for column in matching_columns:
        if column in event_index:
            clearances[column] = clearances["clearance_id"].map(event_index[column])
        else:
            clearances[column] = ""
    clearances["event_timestamp"] = clearances["clearance_timestamp"].astype(str)

    candidates = anchors.loc[anchors["_is_candidate"]].copy()
    candidates.sort_values(["flight_key", "time_epoch"], inplace=True, kind="mergesort")
    candidates.reset_index(drop=True, inplace=True)
    candidates["clearance_free"] = mark_clearance_free(candidates, all_clearances).to_numpy()
    candidates = candidates.loc[candidates["clearance_free"]].copy().reset_index(drop=True)
    candidate_groups = build_candidate_groups(candidates)
    candidate_record_columns = [
        "flight_key", "time_epoch", "timestamp", "archive_week_match", "centre_match",
        "phase_match", "aircraft_type_match", "flight_level", "hour_utc",
        "local_traffic_density", "local_traffic_density_bin",
    ]
    candidate_records = candidates[candidate_record_columns].to_dict("records")

    used_controls: set[tuple[str, float]] = set()
    quality_rows = []
    intervention_meta_by_flight: dict[str, list[dict]] = defaultdict(list)
    control_meta_by_flight: dict[str, list[dict]] = defaultdict(list)
    for index, intervention in enumerate(clearances.itertuples(index=False), start=1):
        intervention_meta = {
            "window_id": f"I:{intervention.clearance_id}",
            "clearance_id": intervention.clearance_id,
            "event_epoch": intervention.clearance_epoch,
            "event_timestamp": intervention.event_timestamp,
        }
        intervention_meta_by_flight[str(intervention.flight_key)].append(intervention_meta)
        control, level, candidate_count, relaxations = control_match(
            intervention, candidate_groups, candidate_records, used_controls, rng,
            args.max_control_time_difference_s,
        )
        control_time_difference = (
            abs(float(control["time_epoch"]) - float(intervention.clearance_epoch))
            if control is not None else np.nan
        )
        quality_rows.append({
            "clearance_id": intervention.clearance_id,
            "intervention_window_id": intervention_meta["window_id"],
            "matched": control is not None,
            "matching_level": level,
            "matching_relaxations": relaxations,
            "candidate_count": candidate_count,
            "intervention_flight_key": intervention.flight_key,
            "intervention_epoch": intervention.clearance_epoch,
            "control_flight_key": control.get("flight_key", "") if control is not None else "",
            "control_epoch": control.get("time_epoch", np.nan) if control is not None else np.nan,
            "control_time_difference_s": control_time_difference,
            "intervention_archive_week": str(getattr(intervention, "archive_week_match", "")),
            "control_archive_week": str(control.get("archive_week_match", "")) if control is not None else "",
            "intervention_centre": str(getattr(intervention, "centre_match", "")),
            "control_centre": str(control.get("centre_match", "")) if control is not None else "",
            "intervention_phase": str(getattr(intervention, "phase_match", "")),
            "control_phase": str(control.get("phase_match", "")) if control is not None else "",
            "intervention_aircraft_type": str(getattr(intervention, "aircraft_type_match", "")),
            "control_aircraft_type": str(control.get("aircraft_type_match", "")) if control is not None else "",
            "intervention_flight_level": getattr(intervention, "flight_level", np.nan),
            "control_flight_level": control.get("flight_level", np.nan) if control is not None else np.nan,
            "intervention_hour_utc": getattr(intervention, "hour_utc", np.nan),
            "control_hour_utc": control.get("hour_utc", np.nan) if control is not None else np.nan,
            "intervention_local_traffic_density": getattr(intervention, "local_traffic_density", np.nan),
            "control_local_traffic_density": control.get("local_traffic_density", np.nan) if control is not None else np.nan,
            "intervention_local_density_bin": str(getattr(intervention, "local_traffic_density_bin", "")),
            "control_local_density_bin": str(control.get("local_traffic_density_bin", "")) if control is not None else "",
        })
        if control is not None:
            control_meta_by_flight[str(control["flight_key"])].append({
                "window_id": f"C:{intervention.clearance_id}",
                "control_for_clearance_id": intervention.clearance_id,
                "event_epoch": float(control["time_epoch"]),
                "event_timestamp": str(control.get("timestamp", "")),
            })
        if index % 1000 == 0:
            print(f"Matched {index}/{len(clearances)} interventions in {week}", flush=True)

    del anchors, event_candidates, event_rows, candidates, candidate_groups, candidate_records
    gc.collect()
    return pd.DataFrame(quality_rows), intervention_meta_by_flight, control_meta_by_flight


def append_parquet_frame(
    frame: pd.DataFrame,
    path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter | None:
    """Append one in-memory batch to a Parquet stream and release the batch."""

    if frame.empty:
        return writer
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    else:
        target_schema = writer.schema
        if not table.schema.equals(target_schema, check_metadata=False):
            table = table.cast(target_schema, safe=False)
    writer.write_table(table)
    return writer


def merge_parquet_shards(shards: list[Path], output_path: Path, empty_columns: list[str] | None = None) -> None:
    """Merge checkpoint Parquet shards without materialising the full table."""

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = None
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for record_batch in parquet.iter_batches(batch_size=DEFAULT_OUTPUT_CHUNK_ROWS):
            table = pa.Table.from_batches([record_batch])
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            else:
                target_schema = writer.schema
                if not table.schema.equals(target_schema, check_metadata=False):
                    table = table.cast(target_schema, safe=False)
            writer.write_table(table)
    if writer is not None:
        writer.close()
    elif empty_columns is not None:
        empty = pa.Table.from_pandas(pd.DataFrame(columns=empty_columns), preserve_index=False)
        pq.write_table(empty, temporary_path, compression="zstd")
    else:
        raise RuntimeError(f"No shards available for {output_path}")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "02_traffic" / "scat_states.parquet")
    parser.add_argument("--clearances-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "03_clearances" / "scat_clearances.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows")
    parser.add_argument("--max-interventions", type=int, default=0)
    parser.add_argument(
        "--event-chunk-size",
        type=int,
        default=500,
        help="Retained for compatibility; anchor metadata is kept per archive week.",
    )
    parser.add_argument(
        "--state-batch-size",
        type=int,
        default=DEFAULT_STATE_BATCH_SIZE,
        help="State rows read from Parquet at a time.",
    )
    parser.add_argument(
        "--output-chunk-rows",
        type=int,
        default=DEFAULT_OUTPUT_CHUNK_ROWS,
        help="Window rows buffered before being appended to Parquet.",
    )
    parser.add_argument("--control-stride-s", type=int, default=30)
    parser.add_argument(
        "--max-control-time-difference-s",
        type=float,
        default=DEFAULT_MAX_CONTROL_TIME_DIFFERENCE_S,
        help="Maximum absolute time distance between intervention and control.",
    )
    parser.add_argument(
        "--local-radius-nm",
        type=float,
        default=LOCAL_RADIUS_NM,
        help="Radius for local traffic matching; not a conflict threshold.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed per-week checkpoint shards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.max_control_time_difference_s < 0
        or args.local_radius_nm <= 0
        or args.event_chunk_size <= 0
        or args.state_batch_size <= 0
        or args.output_chunk_rows <= 0
    ):
        raise SystemExit("Invalid control time distance, local radius or chunk size")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_root / "intervention_windows.parquet",
        args.output_root / "control_windows.parquet",
    ]
    checkpoint_root = args.output_root / "checkpoints"
    checkpoint_path = args.output_root / "checkpoint.json"
    if args.resume and args.overwrite:
        raise SystemExit("Use either --resume or --overwrite, not both")
    if not args.resume:
        if any(path.exists() for path in output_paths) and not args.overwrite:
            raise SystemExit(f"Window outputs already exist under {args.output_root}; use --overwrite")
        for path in output_paths:
            if path.exists():
                path.unlink()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for path in checkpoint_root.iterdir():
            if path.is_file():
                path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        completed_weeks: set[str] = set()
        totals = {
            "intervention_windows": 0,
            "control_windows": 0,
            "intervention_rows": 0,
            "control_rows": 0,
            "quality_rows": 0,
            "matched_controls": 0,
        }
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_weeks = set(checkpoint.get("completed_weeks", []))
        totals = {
            "intervention_windows": int(checkpoint.get("totals", {}).get("intervention_windows", 0)),
            "control_windows": int(checkpoint.get("totals", {}).get("control_windows", 0)),
            "intervention_rows": int(checkpoint.get("totals", {}).get("intervention_rows", 0)),
            "control_rows": int(checkpoint.get("totals", {}).get("control_rows", 0)),
            "quality_rows": int(checkpoint.get("totals", {}).get("quality_rows", 0)),
            "matched_controls": int(checkpoint.get("totals", {}).get("matched_controls", 0)),
        }
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not args.states_path.exists() or not args.clearances_path.exists():
        raise FileNotFoundError("Run stages 02 and 03 before building windows")

    clearance_columns = None
    clearances = pd.read_parquet(args.clearances_path, columns=clearance_columns)
    if clearances.empty:
        raise RuntimeError("No clearances available for window construction")
    clearances["clearance_epoch"] = pd.to_numeric(clearances["clearance_epoch"], errors="coerce")
    clearances = clearances.dropna(subset=["clearance_epoch"]).sort_values("clearance_epoch", kind="mergesort")
    all_clearances = clearances.copy()
    if "surveillance_match_valid" in clearances.columns:
        clearances["surveillance_match_valid"] = clearances["surveillance_match_valid"].fillna(False).astype(bool)
        clearances = clearances.loc[clearances["surveillance_match_valid"]].copy()
    if clearances.empty:
        raise RuntimeError("No actionable clearances have a valid previous-surveillance match")
    if args.max_interventions:
        clearances = clearances.head(args.max_interventions).copy()

    rng = np.random.default_rng(args.random_seed)
    state_columns = existing_columns(args.states_path, STATE_COLUMNS)
    weeks = sorted(clearances["archive_week"].fillna("").astype(str).unique())
    if args.resume:
        missing_checkpoint_shards = [
            week for week in completed_weeks
            if not (checkpoint_root / f"intervention_{week}.parquet").exists()
            or not (checkpoint_root / f"matching_quality_{week}.csv").exists()
        ]
        if missing_checkpoint_shards:
            raise FileNotFoundError(
                "Checkpoint is incomplete for weeks: " + ", ".join(sorted(missing_checkpoint_shards))
            )
    for week_index, week in enumerate(weeks, start=1):
        if week in completed_weeks:
            print(f"[{week_index}/{len(weeks)}] Semana {week_index}/{len(weeks)} ya estaba completada: {week}", flush=True)
            continue
        week_clearances = clearances.loc[clearances["archive_week"].astype(str) == week].copy()
        week_all_clearances = all_clearances.loc[all_clearances["archive_week"].astype(str) == week].copy()
        if not week_clearances.empty:
            intervention_shard = checkpoint_root / f"intervention_{week}.parquet"
            control_shard = checkpoint_root / f"control_{week}.parquet"
            quality_shard = checkpoint_root / f"matching_quality_{week}.csv"
            for path in (intervention_shard, control_shard, quality_shard):
                if path.exists():
                    path.unlink()
            print(f"[{week_index}/{len(weeks)}] Finding anchors for archive week {week}", flush=True)
            quality_frame, intervention_meta_by_flight, control_meta_by_flight = build_partition_streaming(
                week_clearances,
                week_all_clearances,
                args.states_path,
                week,
                state_columns,
                args,
                rng,
            )
            quality_frame.to_csv(quality_shard, index=False)
            intervention_writer = None
            control_writer = None
            intervention_window_ids: set[str] = set()
            control_window_ids: set[str] = set()
            intervention_row_count = 0
            control_row_count = 0
            print(f"[{week_index}/{len(weeks)}] Writing windows for archive week {week}", flush=True)
            for intervention_frame, control_frame in iter_window_frames(
                args.states_path,
                week,
                state_columns,
                intervention_meta_by_flight,
                control_meta_by_flight,
                args.state_batch_size,
                args.output_chunk_rows,
            ):
                intervention_writer = append_parquet_frame(
                    intervention_frame, intervention_shard, intervention_writer
                )
                control_writer = append_parquet_frame(
                    control_frame, control_shard, control_writer
                )
                if not intervention_frame.empty:
                    intervention_window_ids.update(intervention_frame["window_id"].astype(str).unique())
                if not control_frame.empty:
                    control_window_ids.update(control_frame["window_id"].astype(str).unique())
                intervention_row_count += len(intervention_frame)
                control_row_count += len(control_frame)
                del intervention_frame, control_frame
                gc.collect()
            if intervention_writer is not None:
                intervention_writer.close()
            if control_writer is not None:
                control_writer.close()
            if intervention_writer is None or not intervention_window_ids:
                raise RuntimeError(f"No intervention windows could be built for archive week {week}")
            if not quality_frame.empty and "matched" in quality_frame:
                matched_week = int(quality_frame["matched"].fillna(False).astype(bool).sum())
            else:
                matched_week = 0
            totals["intervention_windows"] += len(intervention_window_ids)
            totals["control_windows"] += len(control_window_ids)
            totals["intervention_rows"] += intervention_row_count
            totals["control_rows"] += control_row_count
            totals["quality_rows"] += len(quality_frame)
            totals["matched_controls"] += matched_week
            completed_weeks.add(week)
            checkpoint_path.write_text(
                json.dumps({
                    "completed_weeks": sorted(completed_weeks),
                    "total_weeks": len(weeks),
                    "totals": totals,
                    "state_batch_size": args.state_batch_size,
                    "output_chunk_rows": args.output_chunk_rows,
                    "control_stride_s": args.control_stride_s,
                    "random_seed": args.random_seed,
                }, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{week_index}/{len(weeks)}] Semana {week_index}/{len(weeks)} completada: {week} "
                f"(intervención={len(intervention_window_ids)}, control={len(control_window_ids)})",
                flush=True,
            )
            del quality_frame, intervention_meta_by_flight, control_meta_by_flight
            gc.collect()

    if set(weeks) - completed_weeks:
        raise RuntimeError("Not all archive weeks were completed; use --resume to continue")
    intervention_shards = sorted(checkpoint_root.glob("intervention_*.parquet"))
    control_shards = sorted(checkpoint_root.glob("control_*.parquet"))
    quality_shards = sorted(checkpoint_root.glob("matching_quality_*.csv"))
    if not intervention_shards:
        raise RuntimeError("No intervention checkpoint shards were created")
    merge_parquet_shards(
        intervention_shards,
        output_paths[0],
        empty_columns=existing_columns(intervention_shards[0], STATE_COLUMNS),
    )
    merge_parquet_shards(
        control_shards,
        output_paths[1],
        empty_columns=pq.read_schema(intervention_shards[0]).names,
    )
    quality_path = args.output_root / "matching_quality.csv"
    with quality_path.open("w", encoding="utf-8", newline="") as destination:
        for shard_index, shard in enumerate(quality_shards):
            with shard.open("r", encoding="utf-8") as source:
                header = source.readline()
                if shard_index == 0:
                    destination.write(header)
                for line in source:
                    destination.write(line)
    build_matching_balance_from_csv(quality_path).to_csv(
        args.output_root / "matching_balance.csv", index=False
    )

    config = {
        "states_path": str(args.states_path),
        "clearances_path": str(args.clearances_path),
        "checkpoint_path": str(checkpoint_path),
        "completed_weeks": sorted(completed_weeks),
        "window_seconds_each_side": WINDOW_SECONDS,
        "max_interventions": args.max_interventions,
        "state_batch_size": args.state_batch_size,
        "output_chunk_rows": args.output_chunk_rows,
        "control_stride_s": args.control_stride_s,
        "max_control_time_difference_s": args.max_control_time_difference_s,
        "local_radius_nm": args.local_radius_nm,
        "clearance_exclusion": "all original clearance timestamps, regardless of max-interventions or surveillance-match validity",
        "matching_strata": ["archive_week", "centre", "phase", "flight_level_bin", "hour_utc", "local_density_bin", "aircraft_type"],
        "random_seed": args.random_seed,
        "predictor_rule": "Only relative_seconds <= 0 is predictor-eligible; post-clearance rows are retained for retrospective analysis.",
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = [
        "# SCAT clearance windows",
        "",
        f"- Intervention windows: **{totals['intervention_windows']}**",
        f"- Control windows: **{totals['control_windows']}**",
        f"- Matched controls: **{totals['matched_controls']}/{totals['quality_rows']}**",
        f"- Intervention rows: **{totals['intervention_rows']}**",
        f"- Control rows: **{totals['control_rows']}**",
        "",
        "The windows contain ten minutes before and after each event when surveillance data is available. `predictor_eligible` is false after the event to prevent leakage in predictive models.",
        f"Control candidates use the same archive week, centre, phase, flight-level bin, UTC hour and local-density stratum; aircraft type is relaxed only when needed. The maximum intervention/control time distance is {args.max_control_time_difference_s:g} s and the nearest temporal candidate is selected.",
        "matching_quality.csv records the level and relaxation used for every intervention; matching_balance.csv compares matched anchor covariates.",
        f"State rows are streamed in batches of {args.state_batch_size}; window rows are written in chunks of {args.output_chunk_rows}. Per-week checkpoint shards are kept under `{checkpoint_root}` and can be resumed with `--resume`.",
        "The target is `ATC_INTERVENTION`; no conflict or safety category is created.",
    ]
    (args.output_root / "window_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Window construction complete: {args.output_root}")


if __name__ == "__main__":
    main()

