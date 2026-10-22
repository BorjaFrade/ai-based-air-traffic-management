#!/usr/bin/env python3
"""Conservative first-week FL290–410 kinematic/operational Stage 09 screen.

COUNTERFACTUAL_SCREEN_NOT_CAUSAL: no conflict label, safety minimum or
Stage-07 model score is used to construct the provisional Stage-09 labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK
from cruise_counterfactual_kinematics import (
    angular_difference_deg, circular_median_deg, finite_cpa, haversine_nm,
    nearest_within_radius, next_clearance_censors, pair_series, propagate,
    conservative_label, joint_separation_nm, interaction_agreement_for_clearance,
)


ROOT = DEFAULT_OUTPUT_ROOT
OUT = ROOT / "09_counterfactual_cruise"
REPORT = ROOT / "PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md"
BAND = "CRUISE_FL290_410"
SCOPE = "FIRST_WEEK_PILOT"
COHORT = "CRUISE_MATCHING_V2"
METHOD = "COUNTERFACTUAL_SCREEN_NOT_CAUSAL"
HORIZONS = (60, 120, 300, 600)
MAIN_HORIZONS = (60, 120, 300)
STEP = 5
ANCHOR_MAX_AGE_S = 6.0
OBS_MAX_AGE_S = 3.5
MIN_OBS_COVERAGE = .70
BLOCK_GROUPS = 50
N_PARTNERS = 3
DAY_ALGORITHM_VERSION = 4
FIRST_DAY = pd.Timestamp("2016-10-15", tz="UTC")
DAYS = [FIRST_DAY + pd.Timedelta(days=i) for i in range(7)]
STATE_COLS = ["flight_key", "archive_week", "time_epoch", "latitude", "longitude",
              "altitude_ft", "ground_speed_kt", "track_deg", "heading_deg",
              "vertical_rate_fpm"]
STATE_VARS = [column for column in STATE_COLS if column != "archive_week"]
ALLOWED_LABELS = ("LIKELY_INTERACTION_RELATED", "LIKELY_NON_INTERACTION_OPERATIONAL", "UNKNOWN")


def read_inputs():
    pairs = pd.read_parquet(ROOT / "04_windows/cruise_matching_v2.parquet").reset_index(drop=True)
    windows = pd.read_parquet(ROOT / "05_features/window_features_cruise_matching_v2.parquet")
    pair_split = pd.read_parquet(ROOT / "07_explainability_cruise/pair_split_assignments.parquet")
    assignments = pd.read_parquet(ROOT / "07_explainability_cruise/split_assignments.parquet")
    assert len(pairs) == len(pair_split) == 4098 and len(windows) == len(assignments) == 8196
    assert pair_split.matched_pair_id.tolist() == list(range(4098))
    assert np.array_equal(pairs.intervention_window_id.to_numpy(), pair_split.intervention_window_id.to_numpy())
    assert np.array_equal(pairs.control_window_id.to_numpy(), pair_split.control_window_id.to_numpy())
    assert windows.window_id.is_unique and assignments.window_id.is_unique
    assert set(windows.window_id) == set(assignments.window_id)
    assert set(windows.archive_week) == set(pairs.archive_week) == {PILOT_WEEK}
    assert set(windows.altitude_band) == set(pairs.altitude_band) == {BAND}
    assert windows.analysis_altitude_ft.between(29000, 41000).all()
    assert windows.ATC_INTERVENTION.value_counts().to_dict() == {0: 4098, 1: 4098}
    assert set(windows.loc[windows.ATC_INTERVENTION == 1, "window_id"]) == set(pairs.intervention_window_id)
    assert set(windows.loc[windows.ATC_INTERVENTION == 0, "window_id"]) == set(pairs.control_window_id)
    assert pair_split.split.value_counts().to_dict() == {"train": 2842, "validation": 580,
                                                           "test": 592, "excluded_cut_or_embargo": 84}
    assert pairs.control_clearance_free_verified.all()
    window_context = windows[["window_id", "archive_week", "analysis_altitude_ft", "flight_phase_at_anchor",
                              "centre_at_anchor", "min_topsky_predicted_sep_nm_proxy",
                              "min_horizontal_sep_within_30nm", "vertical_sep_at_event",
                              "min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
                              "max_vertical_closing_within_30nm", "relative_bearing_sin", "relative_bearing_cos",
                              "median_crossing_angle", "median_traffic_density"]]
    event_rows = []
    for side, label in (("intervention", 1), ("control", 0)):
        prefix = side + "_"
        item = pairs[[prefix + "window_id", prefix + "focal_flight_key", prefix + "event_epoch"]].copy()
        item.columns = ["window_id", "flight_key", "event_epoch"]
        item["matched_pair_id"] = np.arange(4098, dtype=np.int32)
        item["ATC_INTERVENTION"] = label
        event_rows.append(item)
    events = pd.concat(event_rows, ignore_index=True)
    events = events.merge(window_context, on="window_id", validate="one_to_one")
    events = events.merge(assignments[["window_id", "split", "connected_group_id"]],
                          on="window_id", validate="one_to_one")
    assert events.window_id.is_unique and len(events) == 8196
    assert events.loc[events.ATC_INTERVENTION == 1, "matched_pair_id"].nunique() == 4098
    assert events.loc[events.ATC_INTERVENTION == 0, "matched_pair_id"].nunique() == 4098
    clear_cols = ["clearance_id", "flight_key", "archive_week", "clearance_epoch", "clearance_type",
                  "changed_fields", "previous_value", "new_value", "delta_value", "clearance_cfl",
                  "clearance_cfl_unit", "clearance_heading_deg", "clearance_speed", "clearance_speed_unit"]
    clearances = pd.read_parquet(ROOT / "03_clearances/scat_clearances.parquet", columns=clear_cols)
    clearances = clearances.loc[clearances.archive_week == PILOT_WEEK].copy()
    assert clearances.clearance_id.is_unique
    intervention_meta = clearances.loc[clearances.clearance_id.isin(pairs.clearance_id)].copy()
    assert len(intervention_meta) == 4098 and intervention_meta.clearance_id.is_unique
    interventions = events.loc[events.ATC_INTERVENTION == 1].copy()
    interventions["clearance_id"] = interventions.matched_pair_id.map(pairs.clearance_id)
    interventions = interventions.merge(intervention_meta.drop(columns=["flight_key", "archive_week", "clearance_epoch"]),
                                        on="clearance_id", validate="one_to_one")
    assert (interventions.window_id == pairs.intervention_window_id).all()
    assert (interventions.flight_key == pairs.intervention_focal_flight_key).all()
    assert np.allclose(interventions.event_epoch, pairs.intervention_event_epoch)
    assert interventions.clearance_type.notna().all()
    controls = events.loc[events.ATC_INTERVENTION == 0].copy()
    controls["clearance_id"] = ""
    controls["clearance_type"] = "CONTROL"
    combined = pd.concat([interventions, controls], ignore_index=True, sort=False)
    assert len(combined) == 8196
    next_by_flight = {key: np.sort(group.clearance_epoch.to_numpy(dtype=float))
                      for key, group in clearances.groupby("flight_key", sort=False)}
    return pairs, combined, clearances, next_by_flight


def json_field(value, key):
    try:
        item = json.loads(value) if isinstance(value, str) else {}
        return item.get(key) if isinstance(item, dict) else None
    except (ValueError, TypeError):
        return None


def next_clearance(next_by_flight: dict, flight: str, anchor: float) -> float:
    epochs = next_by_flight.get(flight)
    if epochs is None:
        return np.nan
    index = np.searchsorted(epochs, anchor + .001, side="right")
    return float(epochs[index]) if index < len(epochs) else np.nan


def cache_first_week_states() -> dict:
    """Checkpointed minimal-column scan; preserves Stage 02 and other weeks."""
    source = pq.ParquetFile(ROOT / "02_traffic/scat_states.parquet")
    week_column = source.schema.names.index("archive_week")
    week_groups = []
    for i in range(source.num_row_groups):
        stats = source.metadata.row_group(i).column(week_column).statistics
        if stats.min == PILOT_WEEK and stats.max == PILOT_WEEK:
            week_groups.append(i)
    expected_rows = sum(source.metadata.row_group(i).num_rows for i in week_groups)
    assert expected_rows == 5495000 and len(week_groups) == 1099
    cache = OUT / "state_cache"
    cache.mkdir(parents=True, exist_ok=True)
    expected_blocks = math.ceil(len(week_groups) / BLOCK_GROUPS)
    for block in range(expected_blocks):
        group_ids = week_groups[block * BLOCK_GROUPS:(block + 1) * BLOCK_GROUPS]
        folder = cache / f"block_{block:03d}"
        marker = folder / "complete.json"
        if marker.exists():
            data = json.loads(marker.read_text())
            assert data["row_groups"] == group_ids
            assert sum(data["day_rows"].values()) == data["nominal_day_rows"] <= data["rows"]
            assert all((folder / name).exists() for name in data["day_rows"])
            continue
        folder.mkdir(exist_ok=True)
        table = source.read_row_groups(group_ids, columns=STATE_COLS)
        assert table.num_rows == sum(source.metadata.row_group(i).num_rows for i in group_ids)
        assert pc.all(pc.equal(table["archive_week"], PILOT_WEEK)).as_py()
        epochs = table["time_epoch"]
        counts = {}
        for day in DAYS:
            lo, hi = day.timestamp(), (day + pd.Timedelta(days=1)).timestamp()
            mask = pc.and_(pc.greater_equal(epochs, lo), pc.less(epochs, hi))
            part = table.filter(mask).drop(["archive_week"])
            if not part.num_rows:
                continue
            name = f"day_{day.strftime('%Y%m%d')}.parquet"
            tmp = folder / (name + ".tmp")
            pq.write_table(part, tmp, compression="zstd", row_group_size=50000)
            tmp.replace(folder / name)
            counts[name] = part.num_rows
        # Stage-02 has a few boundary states outside nominal week; keep count explicit.
        marker.write_text(json.dumps({"row_groups": group_ids, "rows": table.num_rows,
                                      "day_rows": counts, "nominal_day_rows": sum(counts.values())}, indent=2))
        print(f"Stage 09 native-state cache: block {block+1}/{expected_blocks}, "
              f"{table.num_rows:,} first-week rows, {sum(counts.values()):,} in seven nominal days", flush=True)
    markers = [cache / f"block_{i:03d}" / "complete.json" for i in range(expected_blocks)]
    assert all(path.exists() for path in markers)
    manifests = [json.loads(path.read_text()) for path in markers]
    total = sum(item["rows"] for item in manifests)
    nominal = sum(item["nominal_day_rows"] for item in manifests)
    assert total == expected_rows
    result = {"source_rows_first_week": total, "cached_rows_nominal_days": nominal,
              "blocks": expected_blocks, "source_path": str(ROOT / "02_traffic/scat_states.parquet"),
              "columns": STATE_VARS, "other_weeks_read": False}
    (cache / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def load_day_states(day: pd.Timestamp) -> pd.DataFrame:
    cache = OUT / "state_cache"
    paths = sorted(cache.glob(f"block_*/day_{day.strftime('%Y%m%d')}.parquet"))
    assert paths, f"Missing cached states for {day}"
    frames = [pd.read_parquet(path) for path in paths]
    states = pd.concat(frames, ignore_index=True)
    # Cross-midnight windows need only 60 s before and 600 s after.
    lo, hi = day.timestamp(), (day + pd.Timedelta(days=1)).timestamp()
    for adjacent, start, end in ((day - pd.Timedelta(days=1), lo - 65, lo),
                                 (day + pd.Timedelta(days=1), hi, hi + 605)):
        other = sorted(cache.glob(f"block_*/day_{adjacent.strftime('%Y%m%d')}.parquet"))
        if other:
            import pyarrow.dataset as ds
            small = ds.dataset([str(path) for path in other], format="parquet").to_table(
                filter=(ds.field("time_epoch") >= start) & (ds.field("time_epoch") < end)).to_pandas()
            if len(small):
                states = pd.concat([states, small], ignore_index=True)
    states = states.loc[states.latitude.between(-90, 90) & states.longitude.between(-180, 180) &
                        np.isfinite(states.time_epoch) & np.isfinite(states.altitude_ft)].copy()
    states = states.sort_values("time_epoch", kind="mergesort").reset_index(drop=True)
    assert not states.empty and states.time_epoch.is_monotonic_increasing
    return states


class NativeStateIndex:
    def __init__(self, states: pd.DataFrame):
        self.states = states
        self.times = states.time_epoch.to_numpy(dtype=float)
        self.by_flight = states.groupby("flight_key", sort=False).indices

    def last_before(self, flight: str, epoch: float, max_age: float = ANCHOR_MAX_AGE_S):
        positions = self.by_flight.get(flight)
        if positions is None:
            return None
        times = self.times[positions]
        j = np.searchsorted(times, epoch, side="right") - 1
        if j < 0 or epoch - times[j] > max_age or epoch < times[j]:
            return None
        result = self.states.iloc[positions[j]].to_dict()
        result["state_age_s"] = float(epoch - times[j])
        result["position_age_s"] = float(epoch - times[j])
        return result

    def robust_state(self, instant: dict, epoch: float):
        positions = self.by_flight.get(instant["flight_key"])
        if positions is None:
            return instant.copy(), False, 0
        times = self.times[positions]
        lo = np.searchsorted(times, epoch - 60, side="left")
        hi = np.searchsorted(times, epoch, side="right")
        samples = self.states.iloc[positions[lo:hi]]
        good = samples[["ground_speed_kt", "track_deg", "vertical_rate_fpm"]].notna().all(axis=1)
        samples = samples.loc[good]
        if len(samples) < 3:
            return instant.copy(), False, len(samples)
        result = instant.copy()
        result["ground_speed_kt"] = float(samples.ground_speed_kt.median())
        result["track_deg"] = circular_median_deg(samples.track_deg)
        result["vertical_rate_fpm"] = float(samples.vertical_rate_fpm.median())
        return result, True, len(samples)

    def snapshot(self, epoch: float, max_age: float = ANCHOR_MAX_AGE_S):
        lo = np.searchsorted(self.times, epoch - max_age, side="left")
        hi = np.searchsorted(self.times, epoch, side="right")
        slice_frame = self.states.iloc[lo:hi]
        # At most the latest *pre-anchor* state per flight.
        latest = slice_frame.drop_duplicates("flight_key", keep="last")
        latest = latest.loc[latest.ground_speed_kt.notna() & latest.track_deg.notna() &
                            latest.vertical_rate_fpm.notna()]
        return latest.to_dict("records")

    def sampled_trajectory(self, flight: str, epoch: float, grid: np.ndarray, anchor: dict):
        positions = self.by_flight.get(flight)
        fields = ("latitude", "longitude", "altitude_ft", "ground_speed_kt", "track_deg", "heading_deg")
        output = {column: np.full(len(grid), np.nan) for column in fields}
        ages = np.full(len(grid), np.nan)
        for column in fields:
            output[column][0] = anchor.get(column, np.nan)
        ages[0] = anchor["state_age_s"]
        if positions is None:
            return output, ages
        times = self.times[positions]
        target = epoch + grid[1:]
        right = np.searchsorted(times, target, side="left")
        left = np.maximum(right - 1, 0)
        right = np.minimum(right, len(times) - 1)
        choose_right = np.abs(times[right] - target) < np.abs(times[left] - target)
        chosen = np.where(choose_right, right, left)
        deltas = np.abs(times[chosen] - target)
        valid = (deltas <= OBS_MAX_AGE_S) & (times[chosen] > epoch)
        rows = self.states.iloc[positions[chosen]].reset_index(drop=True)
        for column in fields:
            output[column][1:] = np.where(valid, pd.to_numeric(rows[column], errors="coerce"), np.nan)
        ages[1:] = np.where(valid, deltas, np.nan)
        return output, ages


def valid_kinematic_state(state: dict) -> bool:
    return all(np.isfinite(float(state.get(column, np.nan))) for column in
               ("latitude", "longitude", "altitude_ft", "ground_speed_kt", "track_deg", "vertical_rate_fpm"))


def nearest_grid_value(values: np.ndarray, position: int):
    return float(values[position]) if position < len(values) and np.isfinite(values[position]) else np.nan


def process_event(event: dict, index: NativeStateIndex, next_by_flight: dict):
    anchor = float(event["event_epoch"])
    flight = event["flight_key"]
    grid = np.arange(0, max(HORIZONS) + STEP, STEP, dtype=float)
    base = {"window_id": event["window_id"], "matched_pair_id": int(event["matched_pair_id"]),
            "clearance_id": event.get("clearance_id", ""), "flight_key": flight,
            "event_epoch": anchor, "archive_week": PILOT_WEEK,
            "analysis_altitude_ft": float(event["analysis_altitude_ft"]),
            "flight_phase": event["flight_phase_at_anchor"], "centre": event["centre_at_anchor"],
            "clearance_type": event["clearance_type"], "ATC_INTERVENTION": int(event["ATC_INTERVENTION"]),
            "split": event["split"], "connected_group_id": int(event["connected_group_id"]),
            "previous_value": event.get("previous_value", ""), "new_value": event.get("new_value", ""),
            "delta_value": event.get("delta_value", ""),
            "assigned_level": event.get("clearance_cfl", np.nan),
            "assigned_level_unit": event.get("clearance_cfl_unit", ""),
            "assigned_heading": event.get("clearance_heading_deg", np.nan),
            "assigned_speed": event.get("clearance_speed", np.nan),
            "assigned_speed_unit": event.get("clearance_speed_unit", ""),
            "candidate_partner_count_30nm": 0, "candidate_partner_count_50nm": 0,
            "focal_state_age_s": np.nan, "focal_position_age_s": np.nan,
            "focal_robust_state_available": False, "focal_pre_samples": 0,
            "focal_next_clearance_epoch": next_clearance(next_by_flight, flight, anchor),
            "quality_code": "UNKNOWN"}
    focal = index.last_before(flight, anchor)
    if focal is None:
        stale = index.last_before(flight, anchor, max_age=15)
        base["quality_code"] = "STALE_STATE" if stale is not None else "NO_FOCAL_PRE_ANCHOR_STATE"
        return base, [], [], []
    base["focal_state_age_s"] = focal["state_age_s"]
    base["focal_position_age_s"] = focal["position_age_s"]
    base["anchor_altitude_ft"] = focal["altitude_ft"]
    base["anchor_ground_speed_kt"] = focal["ground_speed_kt"]
    base["anchor_heading_deg"] = focal["heading_deg"]
    base["anchor_track_deg"] = focal["track_deg"]
    base["anchor_vertical_rate_fpm"] = focal["vertical_rate_fpm"]
    if not valid_kinematic_state(focal):
        base["quality_code"] = "INVALID_FOCAL_PRE_ANCHOR_KINEMATICS"
        return base, [], [], []
    focal_robust, robust_ok, n_pre = index.robust_state(focal, anchor)
    base["focal_robust_state_available"] = robust_ok
    base["focal_pre_samples"] = n_pre
    base["quality_code"] = "OK" if robust_ok else "INSUFFICIENT_PRE_EVENT_HISTORY"
    observed_focal, focal_ages = index.sampled_trajectory(flight, anchor, grid, focal)
    focal_lat, focal_lon, focal_alt = propagate(focal_robust, grid, anchor)
    focal_instant_lat, focal_instant_lon, focal_instant_alt = propagate(focal, grid, anchor)
    for horizon in HORIZONS:
        j = horizon // STEP
        base[f"observed_focal_altitude_{horizon}s"] = nearest_grid_value(observed_focal["altitude_ft"], j)
        base[f"observed_focal_ground_speed_{horizon}s"] = nearest_grid_value(observed_focal["ground_speed_kt"], j)
        base[f"observed_focal_heading_{horizon}s"] = nearest_grid_value(observed_focal["heading_deg"], j)
        base[f"observed_focal_track_{horizon}s"] = nearest_grid_value(observed_focal["track_deg"], j)
        base[f"observed_focal_state_age_{horizon}s"] = nearest_grid_value(focal_ages, j)
        base[f"focal_next_clearance_within_{horizon}s"] = next_clearance_censors(
            base["focal_next_clearance_epoch"], anchor, horizon)
    snapshot = index.snapshot(anchor)
    nearby = nearest_within_radius(focal, snapshot, 50.0)
    wider = nearest_within_radius(focal, index.snapshot(anchor, max_age=15), 50.0)
    base["stale_neighbours_within_50nm"] = sum(anchor - item["time_epoch"] > ANCHOR_MAX_AGE_S for item in wider)
    base["candidate_partner_count_50nm"] = len(nearby)
    base["candidate_partner_count_30nm"] = sum(item["current_horizontal_sep_nm"] <= 30 for item in nearby)
    if not nearby:
        base["quality_code"] = "NO_NEIGHBOUR_WITHIN_50NM"
        return base, [], [], []
    candidates = []
    for other in nearby:
        if not valid_kinematic_state(other):
            continue
        other["state_age_s"] = anchor - float(other["time_epoch"])
        other["position_age_s"] = other["state_age_s"]
        other_robust, other_ok, other_pre = index.robust_state(other, anchor)
        cpa = {h: finite_cpa(focal_robust, other_robust, anchor, h) for h in MAIN_HORIZONS}
        current_v = abs(float(focal["altitude_ft"]) - float(other["altitude_ft"]))
        from cruise_counterfactual_kinematics import xy_from_latlon
        dx, dy = xy_from_latlon(other["latitude"], other["longitude"], focal["latitude"], focal["longitude"])
        bearing = float(np.rad2deg(np.arctan2(dx, dy)) % 360.0)
        crossing = float(abs(angular_difference_deg(other["track_deg"], focal["track_deg"])))
        candidates.append({"state": other, "robust": other_robust, "robust_ok": other_ok,
                           "pre_samples": other_pre,
                           "record": {"window_id": event["window_id"], "matched_pair_id": int(event["matched_pair_id"]),
                                      "ATC_INTERVENTION": int(event["ATC_INTERVENTION"]),
                                      "event_epoch": anchor, "partner_id": other["flight_key"],
                                      "neighbourhood_nm": 30 if other["current_horizontal_sep_nm"] <= 30 else 50,
                                      "state_age_s": other["state_age_s"], "position_age_s": other["position_age_s"],
                                      "current_horizontal_sep_nm": other["current_horizontal_sep_nm"],
                                      "current_vertical_sep_ft": current_v,
                                      "relative_bearing_deg": bearing, "crossing_angle_deg": crossing,
                                      "relative_track_deg": float(angular_difference_deg(other["track_deg"], focal["track_deg"])),
                                      "horizontal_closing_kt": cpa[300]["horizontal_closing_kt"],
                                      "cpa_60s_nm": cpa[60]["cpa_nm"], "tcpa_60s_s": cpa[60]["tcpa_s"],
                                      "cpa_120s_nm": cpa[120]["cpa_nm"], "tcpa_120s_s": cpa[120]["tcpa_s"],
                                      "cpa_300s_nm": cpa[300]["cpa_nm"], "tcpa_300s_s": cpa[300]["tcpa_s"],
                                      "raw_tcpa_s": cpa[300]["raw_tcpa_s"],
                                      "neighbour_robust_state_available": other_ok,
                                      "neighbour_pre_samples": other_pre,
                                      "next_neighbour_clearance_epoch": next_clearance(next_by_flight, other["flight_key"], anchor)}})
    # The ranking uses only pre-event, finite-300-s geometry, not outcomes.
    candidates.sort(key=lambda row: (row["record"]["neighbourhood_nm"] != 30,
                                     row["record"]["cpa_300s_nm"],
                                     row["record"]["current_horizontal_sep_nm"],
                                     row["record"]["partner_id"]))
    ranked_50 = sorted(candidates, key=lambda row: (row["record"]["cpa_300s_nm"],
                                                    row["record"]["current_horizontal_sep_nm"],
                                                    row["record"]["partner_id"]))
    rank_50 = {candidate["record"]["partner_id"]: rank for rank, candidate in enumerate(ranked_50, 1)}
    for rank, candidate in enumerate(candidates, 1):
        candidate["record"]["pre_event_candidate_rank"] = rank
        candidate["record"]["pre_event_candidate_rank_50nm"] = rank_50[candidate["record"]["partner_id"]]
        candidate["record"]["detailed_trajectory_evaluated"] = (
            rank <= N_PARTNERS or candidate["record"]["pre_event_candidate_rank_50nm"] <= N_PARTNERS)
    candidate_rows = [candidate["record"] for candidate in candidates]
    pair_rows, calibration_rows = [], []
    for candidate in candidates:
        if not candidate["record"]["detailed_trajectory_evaluated"]:
            continue
        other = candidate["state"]
        other_robust = candidate["robust"]
        observed_other, other_ages = index.sampled_trajectory(other["flight_key"], anchor, grid, other)
        other_lat, other_lon, other_alt = propagate(other_robust, grid, anchor)
        other_instant_lat, other_instant_lon, other_instant_alt = propagate(other, grid, anchor)
        cf_h, cf_v = pair_series(focal_lat, focal_lon, focal_alt, other_lat, other_lon, other_alt)
        instant_h, instant_v = pair_series(focal_instant_lat, focal_instant_lon, focal_instant_alt,
                                          other_instant_lat, other_instant_lon, other_instant_alt)
        observed_h, observed_v = pair_series(observed_focal["latitude"], observed_focal["longitude"],
                                             observed_focal["altitude_ft"], observed_other["latitude"],
                                             observed_other["longitude"], observed_other["altitude_ft"])
        observed_neighbour_h, observed_neighbour_v = pair_series(
            focal_lat, focal_lon, focal_alt,
            observed_other["latitude"], observed_other["longitude"], observed_other["altitude_ft"])
        cf_joint = joint_separation_nm(cf_h, cf_v)
        instant_joint = joint_separation_nm(instant_h, instant_v)
        observed_joint = joint_separation_nm(observed_h, observed_v)
        observed_neighbour_joint = joint_separation_nm(observed_neighbour_h, observed_neighbour_v)
        for horizon in HORIZONS:
            count = horizon // STEP + 1
            end = count - 1
            cf_cpa = finite_cpa(focal_robust, other_robust, anchor, horizon)
            instant_cpa = finite_cpa(focal, other, anchor, horizon)
            valid = np.isfinite(observed_h[:count]) & np.isfinite(observed_v[:count])
            valid[0] = True
            coverage = float(valid[1:].mean())
            endpoint_valid = bool(valid[-1])
            focal_censored = next_clearance_censors(base["focal_next_clearance_epoch"], anchor, horizon)
            other_next = candidate["record"]["next_neighbour_clearance_epoch"]
            neighbour_censored = next_clearance_censors(other_next, anchor, horizon)
            cf_min_h, cf_min_v = float(np.min(cf_h[:count])), float(np.min(cf_v[:count]))
            inst_min_h, inst_min_v = float(np.min(instant_h[:count])), float(np.min(instant_v[:count]))
            cf_min_joint, inst_min_joint = float(np.min(cf_joint[:count])), float(np.min(instant_joint[:count]))
            if valid.sum() >= 2:
                obs_min_h = float(np.nanmin(observed_h[:count]))
                obs_min_v = float(np.nanmin(observed_v[:count]))
                obs_min_joint = float(np.nanmin(observed_joint[:count]))
                obs_tcpa = float(grid[np.nanargmin(observed_h[:count])])
            else:
                obs_min_h = obs_min_v = obs_min_joint = obs_tcpa = np.nan
            neighbour_valid = np.isfinite(observed_neighbour_h[:count])
            obsneigh_min_h = float(np.nanmin(observed_neighbour_h[:count])) if neighbour_valid.any() else np.nan
            obsneigh_min_v = float(np.nanmin(observed_neighbour_v[:count])) if neighbour_valid.any() else np.nan
            obsneigh_min_joint = float(np.nanmin(observed_neighbour_joint[:count])) if neighbour_valid.any() else np.nan
            obsneigh_tcpa = float(grid[np.nanargmin(observed_neighbour_h[:count])]) if neighbour_valid.any() else np.nan
            usable = (candidate["robust_ok"] and robust_ok and coverage >= MIN_OBS_COVERAGE and
                      endpoint_valid and not focal_censored and not neighbour_censored)
            if endpoint_valid and end >= 1 and valid[end - 1]:
                obs_closing = float((observed_h[end - 1] - observed_h[end]) * 3600 / STEP)
            else:
                obs_closing = np.nan
            row = {"window_id": event["window_id"], "matched_pair_id": int(event["matched_pair_id"]),
                   "ATC_INTERVENTION": int(event["ATC_INTERVENTION"]), "split": event["split"],
                   "partner_id": other["flight_key"], "candidate_rank": candidate["record"]["pre_event_candidate_rank"],
                   "neighbourhood_nm": candidate["record"]["neighbourhood_nm"],
                   "horizon_s": horizon, "state_age_s": other["state_age_s"],
                   "position_age_s": other["position_age_s"], "focal_state_age_s": focal["state_age_s"],
                   "STALE_STATE": bool(other["state_age_s"] > ANCHOR_MAX_AGE_S or focal["state_age_s"] > ANCHOR_MAX_AGE_S),
                   "FOCAL_NEXT_CLEARANCE": focal_censored,
                   "NEIGHBOUR_NEXT_CLEARANCE": neighbour_censored,
                   "observed_pair_coverage": coverage, "observed_endpoint_valid": endpoint_valid,
                   "robust_pre_state_available": bool(candidate["robust_ok"] and robust_ok),
                   "usable": bool(usable),
                   "current_horizontal_sep_nm": candidate["record"]["current_horizontal_sep_nm"],
                   "current_vertical_sep_ft": candidate["record"]["current_vertical_sep_ft"],
                   "counterfactual_min_horizontal_sep": cf_min_h,
                   "observed_min_horizontal_sep": obs_min_h,
                   "delta_horizontal_sep": obs_min_h - cf_min_h if np.isfinite(obs_min_h) else np.nan,
                   "counterfactual_min_vertical_sep": cf_min_v,
                   "observed_min_vertical_sep": obs_min_v,
                   "delta_vertical_sep": obs_min_v - cf_min_v if np.isfinite(obs_min_v) else np.nan,
                   "counterfactual_min_joint_sep_nm": cf_min_joint,
                   "observed_min_joint_sep_nm": obs_min_joint,
                   "delta_joint_sep_nm": obs_min_joint - cf_min_joint if np.isfinite(obs_min_joint) else np.nan,
                   "counterfactual_CPA": cf_cpa["cpa_nm"], "observed_CPA": obs_min_h,
                   "delta_CPA": obs_min_h - cf_cpa["cpa_nm"] if np.isfinite(obs_min_h) else np.nan,
                   "counterfactual_TCPA": cf_cpa["tcpa_s"], "observed_TCPA": obs_tcpa,
                   "delta_TCPA": obs_tcpa - cf_cpa["tcpa_s"] if np.isfinite(obs_tcpa) else np.nan,
                   "raw_counterfactual_TCPA": cf_cpa["raw_tcpa_s"],
                   "counterfactual_horizontal_closing_kt": cf_cpa["horizontal_closing_kt"],
                   "observed_endpoint_horizontal_closing_kt": obs_closing,
                   "delta_horizontal_closing_kt": cf_cpa["horizontal_closing_kt"] - obs_closing if np.isfinite(obs_closing) else np.nan,
                   "instantaneous_min_horizontal_sep": inst_min_h,
                   "instantaneous_min_vertical_sep": inst_min_v,
                   "instantaneous_min_joint_sep_nm": inst_min_joint,
                   "instantaneous_CPA": instant_cpa["cpa_nm"],
                   "instantaneous_TCPA": instant_cpa["tcpa_s"],
                   "observed_neighbour_min_horizontal_sep": obsneigh_min_h,
                   "observed_neighbour_min_vertical_sep": obsneigh_min_v,
                   "observed_neighbour_min_joint_sep_nm": obsneigh_min_joint,
                   "observed_neighbour_TCPA": obsneigh_tcpa,
                   "pre_event_bearing_deg": candidate["record"]["relative_bearing_deg"],
                   "pre_event_crossing_angle_deg": candidate["record"]["crossing_angle_deg"]}
            pair_rows.append(row)
            if int(event["ATC_INTERVENTION"]) == 0 and usable:
                fpos = float(haversine_nm(focal_lat[end], focal_lon[end],
                                          observed_focal["latitude"][end], observed_focal["longitude"][end]))
                calibration_rows.append({"window_id": event["window_id"], "matched_pair_id": int(event["matched_pair_id"]),
                                         "partner_id": other["flight_key"], "split": event["split"],
                                         "horizon_s": horizon, "horizontal_position_error_nm": fpos,
                                         "altitude_error_ft": abs(float(focal_alt[end]) - observed_focal["altitude_ft"][end]),
                                         "ground_speed_error_kt": abs(float(focal_robust["ground_speed_kt"]) - observed_focal["ground_speed_kt"][end]),
                                         "track_error_deg": abs(float(angular_difference_deg(focal_robust["track_deg"], observed_focal["track_deg"][end]))),
                                         "pair_min_horizontal_error_nm": abs(row["delta_horizontal_sep"]),
                                         "pair_min_vertical_error_ft": abs(row["delta_vertical_sep"]),
                                         "pair_min_joint_error_nm": abs(row["delta_joint_sep_nm"]),
                                         "pair_cpa_error_nm": abs(row["delta_CPA"]),
                                         "pair_tcpa_error_s": abs(row["delta_TCPA"]),
                                         "pair_closing_error_kt": abs(row["delta_horizontal_closing_kt"]) if np.isfinite(row["delta_horizontal_closing_kt"]) else np.nan})
    return base, candidate_rows, pair_rows, calibration_rows


def process_day(day: pd.Timestamp, events: pd.DataFrame, next_by_flight: dict):
    output = OUT / "day_checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    stamp = day.strftime("%Y%m%d")
    marker = output / f"day_{stamp}_complete.json"
    if marker.exists():
        data = json.loads(marker.read_text())
        if data.get("algorithm_version") == DAY_ALGORITHM_VERSION:
            assert data["event_count"] == len(events)
            assert all((output / filename).exists() for filename in data["files"])
            print(f"Stage 09: reused checked day {stamp}, {len(events)} anchors", flush=True)
            return
    states = load_day_states(day)
    index = NativeStateIndex(states)
    base_rows, candidate_rows, pair_rows, calibration_rows = [], [], [], []
    for item in events.to_dict("records"):
        base, candidates, pair, calibration = process_event(item, index, next_by_flight)
        base_rows.append(base)
        candidate_rows.extend(candidates)
        pair_rows.extend(pair)
        calibration_rows.extend(calibration)
    frames = {"events": pd.DataFrame(base_rows), "candidates": pd.DataFrame(candidate_rows),
              "pair_horizon": pd.DataFrame(pair_rows), "control_calibration": pd.DataFrame(calibration_rows)}
    filenames = []
    for name, frame in frames.items():
        filename = f"day_{stamp}_{name}.parquet"
        frame.to_parquet(output / filename, index=False, compression="zstd")
        filenames.append(filename)
    marker.write_text(json.dumps({"day": stamp, "algorithm_version": DAY_ALGORITHM_VERSION,
                                  "event_count": len(events),
                                  "candidate_count": len(candidate_rows), "pair_horizon_count": len(pair_rows),
                                  "control_calibration_count": len(calibration_rows), "files": filenames}, indent=2))
    print(f"Stage 09: completed {stamp}: {len(events)} anchors, {len(candidate_rows)} 50-NM neighbours, "
          f"{len(pair_rows)} detailed pair-horizon records; native rows in day index {len(states):,}", flush=True)


def collect_day_outputs():
    root = OUT / "day_checkpoints"
    frames = {}
    for kind in ("events", "candidates", "pair_horizon", "control_calibration"):
        paths = [root / f"day_{day.strftime('%Y%m%d')}_{kind}.parquet" for day in DAYS]
        assert all(path.exists() for path in paths), (kind, paths)
        frames[kind] = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    base = frames["events"]
    assert len(base) == 8196 and base.window_id.is_unique
    assert base.ATC_INTERVENTION.value_counts().to_dict() == {0: 4098, 1: 4098}
    assert base.matched_pair_id.nunique() == 4098 and set(base.archive_week) == {PILOT_WEEK}
    assert base.analysis_altitude_ft.between(29000, 41000).all()
    assert frames["pair_horizon"].horizon_s.isin(HORIZONS).all()
    assert frames["candidates"].pre_event_candidate_rank.ge(1).all()
    return frames


ERROR_COLS = ["horizontal_position_error_nm", "altitude_error_ft", "ground_speed_error_kt",
              "track_error_deg", "pair_min_horizontal_error_nm", "pair_min_vertical_error_ft",
              "pair_min_joint_error_nm", "pair_cpa_error_nm", "pair_tcpa_error_s", "pair_closing_error_kt"]


def calibrate_controls(raw: pd.DataFrame):
    assert set(raw.split) <= {"train", "validation", "test", "excluded_cut_or_embargo"}
    train = raw.loc[raw.split == "train"]
    thresholds, summary = [], []
    for horizon in HORIZONS:
        fit = train.loc[train.horizon_s == horizon]
        assert fit.window_id.nunique() >= 100, f"Too few training controls at {horizon}s"
        for variable in ERROR_COLS:
            values = pd.to_numeric(fit[variable], errors="coerce").dropna()
            assert len(values) >= 100 and values.ge(0).all(), (horizon, variable, len(values))
            for percentile in (.90, .95):
                thresholds.append({"horizon_s": horizon, "variable": variable,
                                   "percentile": percentile, "threshold": float(values.quantile(percentile)),
                                   "n_train_pair_rows": len(values), "n_train_control_windows": fit.window_id.nunique()})
        for split in ("train", "validation", "test", "excluded_cut_or_embargo"):
            sample = raw.loc[(raw.horizon_s == horizon) & (raw.split == split)]
            for variable in ERROR_COLS:
                x = pd.to_numeric(sample[variable], errors="coerce").dropna()
                q95_train = next(item["threshold"] for item in thresholds
                                 if item["horizon_s"] == horizon and item["variable"] == variable and item["percentile"] == .95)
                summary.append({"split": split, "horizon_s": horizon, "variable": variable,
                                "pair_rows": len(x), "control_windows": sample.window_id.nunique(),
                                "median_error": float(x.median()) if len(x) else np.nan,
                                "p90_error": float(x.quantile(.90)) if len(x) else np.nan,
                                "p95_error": float(x.quantile(.95)) if len(x) else np.nan,
                                "train_p95_threshold": q95_train,
                                "fraction_exceeding_train_p95": float((x > q95_train).mean()) if len(x) else np.nan})
    threshold_frame = pd.DataFrame(thresholds)
    summary_frame = pd.DataFrame(summary)
    raw.to_csv(OUT / "control_counterfactual_calibration.csv", index=False)
    threshold_frame.to_csv(OUT / "calibration_thresholds.csv", index=False)
    summary_frame.to_csv(OUT / "control_calibration_summary.csv", index=False)
    key = summary_frame.loc[(summary_frame.split == "validation") &
                            (summary_frame.horizon_s.isin(MAIN_HORIZONS)) &
                            (summary_frame.variable.isin(["pair_min_horizontal_error_nm",
                                                          "pair_min_vertical_error_ft",
                                                          "pair_min_joint_error_nm"]))]
    # These are simulator-stability checks, not physical/safety separation minima.
    stable = bool((key.fraction_exceeding_train_p95 <= .15).all() and
                  (key.p95_error <= 1.5 * key.train_p95_threshold).all())
    lines = ["# Control-only kinematic calibration", "",
             f"First-week V2 controls: **4,098**; TRAIN controls available for threshold fitting: "
             f"**{train.window_id.nunique()}**; VALIDATION and TEST controls are holdouts.",
             "All thresholds are 90th/95th percentiles of absolute prediction error among usable TRAIN-control pair/horizon rows. "
             "The error is model approximation and observation noise, not a conflict boundary. "
             "Validation does not retune thresholds; TEST is descriptive only.",
             "A quality-stability diagnostic requires validation exceedance ≤15% and validation p95 ≤1.5× the train p95 for horizontal, vertical and joint pair-minimum errors at 60/120/300 s. "
             f"Result: **{'PASS' if stable else 'WARNING'}**. The 600-s horizon is sensitivity-only.", "",
             "| Horizon | TRAIN pair rows | TRAIN horizontal-position p95 (NM) | TRAIN altitude p95 (ft) | "
             "TRAIN pair-min-horizontal p95 (NM) | TRAIN pair-min-vertical p95 (ft) | TRAIN pair-min-joint p95 (NM) | VALIDATION pair-h exceedance | VALIDATION pair-v exceedance | VALIDATION pair-joint exceedance |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon in HORIZONS:
        def row(split, variable):
            return summary_frame.loc[(summary_frame.split == split) &
                                     (summary_frame.horizon_s == horizon) &
                                     (summary_frame.variable == variable)].iloc[0]
        fh = row("train", "pair_min_horizontal_error_nm")
        fv = row("train", "pair_min_vertical_error_ft")
        fj = row("train", "pair_min_joint_error_nm")
        hp = row("train", "horizontal_position_error_nm")
        ap = row("train", "altitude_error_ft")
        vh = row("validation", "pair_min_horizontal_error_nm")
        vv = row("validation", "pair_min_vertical_error_ft")
        vj = row("validation", "pair_min_joint_error_nm")
        lines.append(f"| {horizon} | {fh.pair_rows} | {hp.p95_error:.3f} | {ap.p95_error:.1f} | "
                     f"{fh.p95_error:.3f} | {fv.p95_error:.1f} | {fj.p95_error:.3f} | "
                     f"{vh.fraction_exceeding_train_p95:.3f} | {vv.fraction_exceeding_train_p95:.3f} | "
                     f"{vj.fraction_exceeding_train_p95:.3f} |")
    lines += ["", "Focal position is extrapolated from the robust pre-event state. Pair minima are sampled every 5 s over a finite horizon. "
              "The same focal may contribute multiple neighbour rows; these error quantiles are descriptive, not independent confidence intervals. "
              "Observed pair coverage, subsequent-clearance censoring and fresh native states were required. "
              "The 1,255 cached first-week states outside the seven nominal dates were not needed by V2 anchors."]
    (OUT / "control_counterfactual_calibration_report.md").write_text("\n".join(lines) + "\n")
    return threshold_frame, summary_frame, stable


def threshold_lookup(thresholds: pd.DataFrame, horizon: int, variable: str, percentile: float):
    found = thresholds.loc[(thresholds.horizon_s == horizon) & (thresholds.variable == variable) &
                           np.isclose(thresholds.percentile, percentile)]
    assert len(found) == 1
    return float(found.threshold.iloc[0])


def level_target_ft(row: dict):
    target = row.get("assigned_level", np.nan)
    unit = row.get("assigned_level_unit")
    if not np.isfinite(target):
        return np.nan
    if unit == "F":
        return float(target) * 100.0
    if unit == "A":
        return float(target)
    return np.nan


def clearance_response(row: dict, horizon: int, thresholds: pd.DataFrame) -> tuple[str, str, float]:
    if row["ATC_INTERVENTION"] != 1:
        return "UNKNOWN", "CONTROL_NO_CLEARANCE", np.nan
    if row.get(f"focal_next_clearance_within_{horizon}s", False):
        return "UNKNOWN", "FOCAL_NEXT_CLEARANCE", np.nan
    kind = row.get("clearance_type", "")
    alt0 = row.get("anchor_altitude_ft", np.nan)
    alt1 = row.get(f"observed_focal_altitude_{horizon}s", np.nan)
    speed0 = row.get("anchor_ground_speed_kt", np.nan)
    speed1 = row.get(f"observed_focal_ground_speed_{horizon}s", np.nan)
    heading0 = row.get("anchor_heading_deg", np.nan)
    heading1 = row.get(f"observed_focal_heading_{horizon}s", np.nan)
    if kind == "LEVEL":
        target = level_target_ft(row)
        if not all(np.isfinite(v) for v in (target, alt0, alt1)):
            return "UNKNOWN", "MISSING_LEVEL_RESPONSE", np.nan
        error = threshold_lookup(thresholds, horizon, "altitude_error_ft", .95)
        initial_gap = abs(alt0 - target)
        if initial_gap <= error:
            return "UNKNOWN", "LEVEL_ALREADY_WITHIN_MODEL_ERROR", np.nan
        improvement = initial_gap - abs(alt1 - target)
        if improvement > error:
            return "YES", "LEVEL_MOVED_TOWARD_ASSIGNED", improvement
        if improvement > 0:
            return "PARTIAL", "LEVEL_SMALL_TOWARD_ASSIGNED", improvement
        if improvement < -error:
            return "NO", "LEVEL_MOVED_AWAY", improvement
        return "UNKNOWN", "LEVEL_CHANGE_WITHIN_MODEL_ERROR", improvement
    if kind == "HEADING":
        target = row.get("assigned_heading", np.nan)
        if not all(np.isfinite(v) for v in (target, heading0, heading1)):
            return "UNKNOWN", "MISSING_HEADING_RESPONSE", np.nan
        error = threshold_lookup(thresholds, horizon, "track_error_deg", .95)
        initial_gap = abs(float(angular_difference_deg(heading0, target)))
        improvement = initial_gap - abs(float(angular_difference_deg(heading1, target)))
        if initial_gap <= error:
            return "UNKNOWN", "HEADING_ALREADY_WITHIN_MODEL_ERROR", improvement
        if improvement > error:
            return "YES", "HEADING_MOVED_TOWARD_ASSIGNED", improvement
        if improvement > 0:
            return "PARTIAL", "HEADING_SMALL_TOWARD_ASSIGNED", improvement
        if improvement < -error:
            return "NO", "HEADING_MOVED_AWAY", improvement
        return "UNKNOWN", "HEADING_CHANGE_WITHIN_MODEL_ERROR", improvement
    if kind == "SPEED":
        before = json_field(row.get("previous_value", ""), "assigned_speed_val")
        after = json_field(row.get("new_value", ""), "assigned_speed_val")
        old_unit = json_field(row.get("previous_value", ""), "assigned_speed_unit")
        new_unit = json_field(row.get("new_value", ""), "assigned_speed_unit")
        if (before is None or after is None or old_unit != new_unit or
                not all(np.isfinite(v) for v in (speed0, speed1))):
            return "UNKNOWN", "SPEED_COMMAND_OR_GROUNDSPEED_NOT_COMPARABLE", np.nan
        commanded_change = float(after) - float(before)
        observed_change = float(speed1) - float(speed0)
        if abs(commanded_change) < 1e-9:
            return "UNKNOWN", "SPEED_NO_COMMANDED_DELTA", observed_change
        error = threshold_lookup(thresholds, horizon, "ground_speed_error_kt", .95)
        if commanded_change * observed_change > 0 and abs(observed_change) > error:
            # Assigned speed is airspeed/Mach while the observed field is ground speed.
            return "PARTIAL", "SPEED_GROUNDSPEED_DIRECTION_ONLY", observed_change
        if commanded_change * observed_change < 0 and abs(observed_change) > error:
            return "NO", "SPEED_GROUNDSPEED_OPPOSITE_DIRECTION", observed_change
        return "UNKNOWN", "SPEED_CHANGE_WITHIN_MODEL_ERROR", observed_change
    return "UNKNOWN", "MULTI_OR_OTHER_RESPONSE_UNVERIFIED", np.nan


def add_clearance_responses(base: pd.DataFrame, thresholds: pd.DataFrame):
    rows = []
    for item in base.loc[base.ATC_INTERVENTION == 1].to_dict("records"):
        for horizon in HORIZONS:
            status, reason, magnitude = clearance_response(item, horizon, thresholds)
            rows.append({"window_id": item["window_id"], "matched_pair_id": item["matched_pair_id"],
                         "clearance_type": item["clearance_type"], "horizon_s": horizon,
                         "clearance_response_confirmed": status, "response_reason": reason,
                         "response_improvement_native_units": magnitude})
    frame = pd.DataFrame(rows)
    assert len(frame) == 4098 * 4
    frame.to_csv(OUT / "clearance_response_summary.csv", index=False)
    return frame


def annotate_evidence(pair_horizon: pd.DataFrame, responses: pd.DataFrame,
                      thresholds: pd.DataFrame, percentile: float = .95,
                      baseline: str = "robust_symmetric") -> pd.DataFrame:
    """Apply control-derived materiality to physical rows; never uses Stage 07."""
    data = pair_horizon.loc[pair_horizon.ATC_INTERVENTION == 1].copy()
    response_cols = responses[["window_id", "horizon_s", "clearance_type", "clearance_response_confirmed"]]
    data = data.merge(response_cols, on=["window_id", "horizon_s"], validate="many_to_one")
    for variable, suffix in (("pair_min_horizontal_error_nm", "h"),
                             ("pair_min_vertical_error_ft", "v"),
                             ("pair_min_joint_error_nm", "j"),
                             ("pair_tcpa_error_s", "t")):
        for name, quantile in (("primary", percentile), ("p90", .90)):
            lookup = {h: threshold_lookup(thresholds, h, variable, quantile) for h in HORIZONS}
            data[f"threshold_{suffix}_{name}"] = data.horizon_s.map(lookup)
    if baseline == "robust_symmetric":
        cf_h = data.counterfactual_min_horizontal_sep
        cf_v = data.counterfactual_min_vertical_sep
        cf_joint = data.counterfactual_min_joint_sep_nm
        cf_tcpa = data.counterfactual_TCPA
        raw_tcpa = data.raw_counterfactual_TCPA
    elif baseline == "instantaneous_symmetric":
        cf_h = data.instantaneous_min_horizontal_sep
        cf_v = data.instantaneous_min_vertical_sep
        cf_joint = data.instantaneous_min_joint_sep_nm
        cf_tcpa = data.instantaneous_TCPA
        # The raw instant TCPA is not persisted; clipped positive timing plus
        # a decreasing separation is the conservative proxy in sensitivity.
        raw_tcpa = cf_tcpa
    elif baseline == "robust_observed_neighbour":
        cf_h = data.observed_neighbour_min_horizontal_sep
        cf_v = data.observed_neighbour_min_vertical_sep
        cf_joint = data.observed_neighbour_min_joint_sep_nm
        cf_tcpa = data.observed_neighbour_TCPA
        raw_tcpa = cf_tcpa
    else:
        raise ValueError(baseline)
    data["screen_cf_min_h"] = cf_h
    data["screen_cf_min_v"] = cf_v
    data["screen_cf_min_joint"] = cf_joint
    data["screen_cf_tcpa"] = cf_tcpa
    data["screen_delta_h"] = data.observed_min_horizontal_sep - cf_h
    data["screen_delta_v"] = data.observed_min_vertical_sep - cf_v
    data["screen_delta_joint"] = data.observed_min_joint_sep_nm - cf_joint
    data["screen_delta_tcpa"] = data.observed_TCPA - cf_tcpa
    data["cf_horizontal_convergence_nm"] = data.current_horizontal_sep_nm - cf_h
    data["cf_vertical_convergence_ft"] = data.current_vertical_sep_ft - cf_v
    data["pre_event_convergence"] = (
        (raw_tcpa > 0) & (raw_tcpa <= data.horizon_s) &
        ((data.cf_horizontal_convergence_nm > data.threshold_h_primary) |
         (data.cf_vertical_convergence_ft > data.threshold_v_primary)))
    data["material_horizontal"] = data.screen_delta_h > data.threshold_h_primary
    data["material_vertical"] = data.screen_delta_v > data.threshold_v_primary
    data["material_joint"] = data.screen_delta_joint > data.threshold_j_primary
    data["material_timing"] = ((data.screen_delta_tcpa > data.threshold_t_primary) &
                               (data.counterfactual_horizontal_closing_kt > 0))
    data["material_improvement"] = data[["material_horizontal", "material_vertical",
                                          "material_joint", "material_timing"]].any(axis=1)
    # For the primary robust/symmetric model, seek corroboration from BOTH
    # instantaneous-state and observed-neighbour counterfactual variants.
    agree_h = (data.material_horizontal &
               (data.observed_min_horizontal_sep - data.instantaneous_min_horizontal_sep > data.threshold_h_p90) &
               (data.observed_min_horizontal_sep - data.observed_neighbour_min_horizontal_sep > data.threshold_h_p90))
    agree_v = (data.material_vertical &
               (data.observed_min_vertical_sep - data.instantaneous_min_vertical_sep > data.threshold_v_p90) &
               (data.observed_min_vertical_sep - data.observed_neighbour_min_vertical_sep > data.threshold_v_p90))
    agree_t = (data.material_timing &
               (data.observed_TCPA - data.instantaneous_TCPA > data.threshold_t_p90) &
               (data.observed_TCPA - data.observed_neighbour_TCPA > data.threshold_t_p90))
    agree_joint = (data.material_joint &
                   (data.observed_min_joint_sep_nm - data.instantaneous_min_joint_sep_nm > data.threshold_j_p90) &
                   (data.observed_min_joint_sep_nm - data.observed_neighbour_min_joint_sep_nm > data.threshold_j_p90))
    data["component_model_agreement"] = agree_h | agree_v
    data["counterfactual_model_agreement"] = agree_joint | agree_t
    if baseline != "robust_symmetric":
        # Sensitivity labels still require agreement with the frozen primary.
        data["counterfactual_model_agreement"] &= data.material_improvement
    data["response_sufficient"] = data.clearance_response_confirmed.isin(["YES", "PARTIAL"])
    # Timing-only evidence is admissible for SPEED, where a clearance can
    # shift encounter timing without appreciably changing spatial geometry.
    # LEVEL and HEADING need a corroborated joint-geometry improvement.
    data["interaction_evidence"] = (data.usable & data.pre_event_convergence &
                                     interaction_agreement_for_clearance(data.clearance_type,
                                                                         agree_joint, agree_t) &
                                     data.counterfactual_model_agreement & data.response_sufficient)
    return data


def classify_events(base: pd.DataFrame, annotated: pd.DataFrame, responses: pd.DataFrame,
                    percentile: float = .95,
                    radius_nm: int = 30, baseline: str = "robust_symmetric",
                    main_horizons: tuple[int, ...] = MAIN_HORIZONS) -> pd.DataFrame:
    assert radius_nm in (30, 50)
    rows = []
    grouped = {key: group for key, group in annotated.groupby("window_id", sort=False)}
    response_grouped = {key: group for key, group in responses.groupby("window_id", sort=False)}
    for item in base.loc[base.ATC_INTERVENTION == 1].to_dict("records"):
        window_id = item["window_id"]
        all_pairs = grouped.get(window_id, annotated.iloc[0:0])
        part = all_pairs.loc[all_pairs.neighbourhood_nm <= radius_nm]
        candidate_total = int(item[f"candidate_partner_count_{radius_nm}nm"])
        evaluated = part.partner_id.nunique()
        complete_partner_set = candidate_total > 0 and candidate_total <= evaluated
        hlabels = {}
        supporting_partners = []
        usable_horizons = 0
        all_model_agree = True
        reasons = []
        for horizon in HORIZONS:
            h = part.loc[part.horizon_s == horizon]
            usable = h.loc[h.usable]
            if len(usable):
                usable_horizons += int(horizon in main_horizons)
            positive = usable.loc[usable.interaction_evidence]
            if len(positive):
                hlabels[horizon] = "LIKELY_INTERACTION_RELATED"
                supporting_partners += positive.partner_id.tolist() if horizon in main_horizons else []
            elif (complete_partner_set and len(h) == candidate_total and len(usable) == candidate_total and
                  (usable.clearance_response_confirmed == "YES").all() and
                  not usable.pre_event_convergence.any() and not usable.material_improvement.any()):
                hlabels[horizon] = "LIKELY_NON_INTERACTION_OPERATIONAL"
            else:
                hlabels[horizon] = "UNKNOWN"
            if len(usable.loc[usable.material_improvement & ~usable.counterfactual_model_agreement]):
                all_model_agree = False
        main_labels = [hlabels[h] for h in main_horizons]
        positive_h = sum(label == "LIKELY_INTERACTION_RELATED" for label in main_labels)
        negative_h = sum(label == "LIKELY_NON_INTERACTION_OPERATIONAL" for label in main_labels)
        same_partner = any(supporting_partners.count(key) >= 2 for key in set(supporting_partners))
        conflicting_horizons = positive_h > 0 and negative_h > 0
        response_values = response_grouped[window_id].loc[
            response_grouped[window_id].horizon_s.isin(main_horizons),
            "clearance_response_confirmed"].dropna().unique()
        response = ("YES" if "YES" in response_values and "NO" not in response_values else
                    "PARTIAL" if "PARTIAL" in response_values and "NO" not in response_values else
                    "NO" if "NO" in response_values else "UNKNOWN")
        focal_next = any(item.get(f"focal_next_clearance_within_{h}s", False) for h in main_horizons)
        # A later clearance censors only the horizons it precedes. Do not
        # invalidate shorter, still-usable evidence wholesale.
        if focal_next and usable_horizons < 2:
            reasons.append("FOCAL_NEXT_CLEARANCE")
        if item["quality_code"] != "OK":
            reasons.append(item["quality_code"])
        if candidate_total == 0:
            reasons.append("NO_SUITABLE_NEIGHBOUR")
        if not complete_partner_set:
            reasons.append("UNTESTED_CANDIDATES")
        if usable_horizons < 2:
            reasons.append("INSUFFICIENT_USABLE_HORIZONS")
        if response not in {"YES", "PARTIAL"}:
            reasons.append("CLEARANCE_RESPONSE_UNCONFIRMED")
        if not all_model_agree:
            reasons.append("COUNTERFACTUAL_SCENARIOS_DISAGREE")
        if positive_h and not same_partner:
            reasons.append("PARTNER_OR_HORIZON_AMBIGUITY")
        if conflicting_horizons:
            reasons.append("HORIZON_DISAGREEMENT")
        if not positive_h and not negative_h:
            reasons.append("MATERIALITY_OR_CONVERGENCE_INSUFFICIENT")
        label = conservative_label(
            has_coverage=candidate_total > 0 and usable_horizons >= 2,
            response=response, concordant=not conflicting_horizons and same_partner,
            converging=positive_h > 0, material_improvement=positive_h > 0,
            has_new_clearance=focal_next and usable_horizons < 2,
            evidence_horizons=positive_h, model_agreement=all_model_agree)
        if label == "LIKELY_INTERACTION_RELATED" and positive_h < 2:
            label = "UNKNOWN"
        if label == "UNKNOWN" and (negative_h >= 2 and complete_partner_set and not conflicting_horizons and
                                   response == "YES" and all_model_agree):
            label = "LIKELY_NON_INTERACTION_OPERATIONAL"
        if label != "UNKNOWN":
            reasons = []
        elif not reasons:
            reasons = ["EVIDENCE_NOT_CONCORDANT_ACROSS_HORIZONS"]
        # Candidate selected by frozen pre-event rank among supporting rows.
        preferred = part.loc[(part.horizon_s.isin(main_horizons)) &
                             (part.interaction_evidence if label == "LIKELY_INTERACTION_RELATED" else part.usable)]
        preferred = preferred.sort_values(["candidate_rank", "horizon_s"], kind="mergesort")
        if len(preferred):
            chosen = preferred.iloc[0]
            partner = chosen.partner_id
            primary_horizon = int(chosen.horizon_s)
        else:
            partner = ""
            primary_horizon = np.nan
        if label == "LIKELY_INTERACTION_RELATED" and same_partner:
            frequent = pd.Series(supporting_partners).value_counts().index[0]
            selected = preferred.loc[preferred.partner_id == frequent]
            if len(selected):
                chosen = selected.iloc[0]
                partner, primary_horizon = chosen.partner_id, int(chosen.horizon_s)
        supporting_horizons = positive_h if label == "LIKELY_INTERACTION_RELATED" else negative_h
        coverage_component = min(1.0, usable_horizons / len(main_horizons))
        response_component = {"YES": 1.0, "PARTIAL": .5}.get(response, 0.0)
        scenario_component = 1.0 if all_model_agree else 0.0
        horizon_component = min(1.0, supporting_horizons / len(main_horizons))
        if len(preferred):
            ratio_j = max(0.0, float(chosen.screen_delta_joint) / max(float(chosen.threshold_j_primary), 1e-9))
            ratio_t = max(0.0, float(chosen.screen_delta_tcpa) / max(float(chosen.threshold_t_primary), 1e-9))
            magnitude_component = min(1.0, max(ratio_j, ratio_t) / 2) if label == "LIKELY_INTERACTION_RELATED" else max(0.0, 1.0 - min(1.0, max(ratio_j, ratio_t)))
        else:
            magnitude_component = 0.0
        confidence = (.25 * coverage_component + .20 * response_component + .20 * scenario_component +
                      .20 * horizon_component + .15 * magnitude_component)
        if label == "UNKNOWN":
            confidence = min(confidence, .49)
        result = {"window_id": window_id, "matched_pair_id": int(item["matched_pair_id"]),
                  "clearance_id": item["clearance_id"], "flight_key": item["flight_key"],
                  "event_epoch": item["event_epoch"], "archive_week": PILOT_WEEK,
                  "analysis_altitude_ft": item["analysis_altitude_ft"],
                  "flight_phase": item["flight_phase"], "centre": item["centre"],
                  "clearance_type": item["clearance_type"],
                  "previous_value": item.get("previous_value", ""),
                  "new_value": item.get("new_value", ""),
                  "delta_value": item.get("delta_value", ""),
                  "assigned_level": item.get("assigned_level", np.nan),
                  "assigned_heading": item.get("assigned_heading", np.nan),
                  "assigned_speed": item.get("assigned_speed", np.nan),
                  "candidate_partner_id": partner,
                  "candidate_partner_count": candidate_total,
                  "candidate_partner_count_30nm": item["candidate_partner_count_30nm"],
                  "candidate_partner_count_50nm": item["candidate_partner_count_50nm"],
                  "primary_horizon": primary_horizon,
                  "clearance_response_confirmed": response,
                  "counterfactual_model_agreement": "YES" if all_model_agree else "NO",
                  "next_clearance_censored": bool(focal_next or part.loc[part.horizon_s.isin(main_horizons),
                                                                 "NEIGHBOUR_NEXT_CLEARANCE"].any()),
                  "stage09_label": label, "evidence_confidence_score": round(confidence, 6),
                  "unknown_reason": ";".join(sorted(set(reasons))) if label == "UNKNOWN" else "",
                  "supporting_main_horizons": positive_h if label == "LIKELY_INTERACTION_RELATED" else negative_h,
                  "candidate_partner_ambiguity": bool(positive_h and not same_partner),
                  "focal_state_age_s": item["focal_state_age_s"],
                  "stale_neighbours_within_50nm": item.get("stale_neighbours_within_50nm", 0),
                  "quality_code": item["quality_code"],
                  "materiality_percentile": percentile,
                  "neighbourhood_nm": radius_nm, "counterfactual_baseline": baseline}
        for horizon in HORIZONS:
            result[f"label_{horizon}s"] = hlabels[horizon]
            result[f"usable_{horizon}s"] = bool(part.loc[(part.horizon_s == horizon) & part.usable].shape[0])
        for source, target in (("counterfactual_min_horizontal_sep", "counterfactual_min_horizontal_sep"),
                               ("observed_min_horizontal_sep", "observed_min_horizontal_sep"),
                               ("delta_horizontal_sep", "delta_horizontal_sep"),
                               ("counterfactual_min_vertical_sep", "counterfactual_min_vertical_sep"),
                               ("observed_min_vertical_sep", "observed_min_vertical_sep"),
                               ("delta_vertical_sep", "delta_vertical_sep"),
                               ("counterfactual_min_joint_sep_nm", "counterfactual_min_joint_sep_nm"),
                               ("observed_min_joint_sep_nm", "observed_min_joint_sep_nm"),
                               ("delta_joint_sep_nm", "delta_joint_sep_nm"),
                               ("counterfactual_CPA", "counterfactual_CPA"),
                               ("observed_CPA", "observed_CPA"),
                               ("delta_CPA", "delta_CPA"),
                               ("counterfactual_TCPA", "counterfactual_TCPA"),
                               ("observed_TCPA", "observed_TCPA")):
            result[target] = float(chosen[source]) if len(preferred) else np.nan
        rows.append(result)
    result = pd.DataFrame(rows)
    assert len(result) == 4098 and result.window_id.is_unique and set(result.stage09_label) <= set(ALLOWED_LABELS)
    assert result.evidence_confidence_score.between(0, 1).all()
    return result


def sensitivity_analyses(base: pd.DataFrame, pair_horizon: pd.DataFrame,
                         responses: pd.DataFrame, thresholds: pd.DataFrame,
                         primary: pd.DataFrame) -> pd.DataFrame:
    configurations = [
        ("50nm_neighbourhood", 50, .95, "robust_symmetric", MAIN_HORIZONS),
        ("90th_percentile_error", 30, .90, "robust_symmetric", MAIN_HORIZONS),
        ("instantaneous_anchor_state", 30, .95, "instantaneous_symmetric", MAIN_HORIZONS),
        ("observed_neighbour", 30, .95, "robust_observed_neighbour", MAIN_HORIZONS),
        ("60_120_horizons", 30, .95, "robust_symmetric", (60, 120)),
        ("120_300_horizons", 30, .95, "robust_symmetric", (120, 300)),
    ]
    original = primary.set_index("window_id").stage09_label
    rows, label_rows = [], []
    cached_95 = annotate_evidence(pair_horizon, responses, thresholds)
    for name, radius, percentile, baseline, horizons in configurations:
        data = (cached_95 if percentile == .95 and baseline == "robust_symmetric" else
                annotate_evidence(pair_horizon, responses, thresholds, percentile, baseline))
        variant = classify_events(base, data, responses, percentile, radius, baseline, horizons)
        variant_labels = variant.set_index("window_id").stage09_label.loc[original.index]
        classified = original != "UNKNOWN"
        agreement_all = float((original == variant_labels).mean())
        agreement_classified = float((original.loc[classified] == variant_labels.loc[classified]).mean()) if classified.any() else np.nan
        counts = variant_labels.value_counts()
        primary_counts = original.value_counts()
        ratio_interaction = (counts.get("LIKELY_INTERACTION_RELATED", 0) /
                             max(primary_counts.get("LIKELY_INTERACTION_RELATED", 0), 1))
        ratio_non_interaction = (counts.get("LIKELY_NON_INTERACTION_OPERATIONAL", 0) /
                                 max(primary_counts.get("LIKELY_NON_INTERACTION_OPERATIONAL", 0), 1))
        count_stable = all(.8 <= ratio <= 1.25 for ratio in (ratio_interaction, ratio_non_interaction))
        status = ("STABLE" if agreement_classified >= .8 and count_stable else
                  "PARTIAL" if agreement_classified >= .6 else "UNSTABLE")
        rows.append({"configuration": name, "neighbourhood_nm": radius, "materiality_percentile": percentile,
                     "counterfactual_baseline": baseline, "main_horizons": ",".join(map(str, horizons)),
                     "LIKELY_INTERACTION_RELATED": int(counts.get("LIKELY_INTERACTION_RELATED", 0)),
                     "LIKELY_NON_INTERACTION_OPERATIONAL": int(counts.get("LIKELY_NON_INTERACTION_OPERATIONAL", 0)),
                     "UNKNOWN": int(counts.get("UNKNOWN", 0)),
                     "agreement_all_events": agreement_all,
                     "agreement_among_primary_classified": agreement_classified,
                     "stability_descriptor": status})
        label_rows.append(pd.DataFrame({"window_id": original.index, "configuration": name,
                                        "stage09_label": variant_labels.to_numpy()}))
        print(f"Stage 09 sensitivity {name}: "
              f"{dict(counts)}; classified agreement {agreement_classified:.3f}", flush=True)
    pd.concat(label_rows, ignore_index=True).to_parquet(OUT / "sensitivity_labels.parquet", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "sensitivity_analysis.csv", index=False)
    return result


def summarize_horizons(event_summary: pd.DataFrame, pair_horizon: pd.DataFrame) -> pd.DataFrame:
    interventions = pair_horizon.loc[(pair_horizon.ATC_INTERVENTION == 1) &
                                     (pair_horizon.neighbourhood_nm <= 30)]
    rows = []
    for horizon in HORIZONS:
        subset = interventions.loc[interventions.horizon_s == horizon]
        counts = event_summary[f"label_{horizon}s"].value_counts()
        rows.append({"horizon_s": horizon, "events_total": len(event_summary),
                     "events_with_usable_30nm_pair": int(event_summary[f"usable_{horizon}s"].sum()),
                     "usable_30nm_pair_rows": int(subset.usable.sum()),
                     "focal_next_clearance_pair_rows": int(subset.FOCAL_NEXT_CLEARANCE.sum()),
                     "neighbour_next_clearance_pair_rows": int(subset.NEIGHBOUR_NEXT_CLEARANCE.sum()),
                     **{name: int(counts.get(name, 0)) for name in ALLOWED_LABELS}})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "horizon_summary.csv", index=False)
    return result


def write_label_tables(events: pd.DataFrame, pair_horizon: pd.DataFrame):
    labels = pd.DataFrame({"stage09_label": ALLOWED_LABELS})
    counts = events.stage09_label.value_counts()
    labels["n"] = labels.stage09_label.map(counts).fillna(0).astype(int)
    labels["percent"] = 100 * labels.n / len(events)
    labels.to_csv(OUT / "counterfactual_label_counts.csv", index=False)
    summary = []
    for kind, part in events.groupby("clearance_type", sort=True):
        count = part.stage09_label.value_counts()
        row = {"clearance_type": kind, "n_events": len(part),
               "mean_evidence_confidence_score": part.evidence_confidence_score.mean(),
               "median_candidate_count_30nm": part.candidate_partner_count_30nm.median(),
               **{f"usable_{h}s_percent": 100 * part[f"usable_{h}s"].mean() for h in HORIZONS}}
        for label in ALLOWED_LABELS:
            row[label + "_n"] = int(count.get(label, 0))
            row[label + "_percent"] = 100 * row[label + "_n"] / len(part)
        summary.append(row)
    by_type = pd.DataFrame(summary)
    by_type.to_csv(OUT / "counterfactual_labels_by_clearance_type.csv", index=False)
    reasons = events.loc[events.stage09_label == "UNKNOWN", "unknown_reason"].str.split(";").explode().value_counts()
    pd.DataFrame({"unknown_reason": reasons.index, "n_events": reasons.to_numpy()}).to_csv(
        OUT / "unknown_reason_counts.csv", index=False)
    return labels, by_type


def posthoc_stage07(events: pd.DataFrame, windows: pd.DataFrame):
    """Read Stage 07 scores only after event labels have been frozen to disk."""
    scores = pd.read_parquet(ROOT / "07_explainability_cruise/test_predictions.parquet")
    scores = scores.loc[(scores.feature_family == "SCAT_FULL") & (scores.model == "xgboost") &
                        (scores.ATC_INTERVENTION == 1), ["window_id", "score"]]
    assert scores.window_id.is_unique and len(scores) == 592
    joined = events[["window_id", "stage09_label"]].merge(scores, on="window_id", how="inner", validate="one_to_one")
    assert len(joined) == 592
    grouped = joined.groupby("stage09_label").score.agg(["size", "mean", "median", "std"])
    grouped = grouped.rename(columns={"size": "n_test_interventions", "mean": "mean_stage07_score",
                                      "median": "median_stage07_score", "std": "sd_stage07_score"}).reset_index()
    grouped["stage07_score_scope"] = "SCAT_FULL_XGBOOST_TEST_ONLY_MATCHED_COHORT_NOT_POPULATION_PROBABILITY"
    grouped.to_csv(OUT / "stage07_posthoc_comparison.csv", index=False)
    feature_cols = ["min_horizontal_sep_within_30nm", "vertical_sep_at_event",
                    "min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
                    "max_vertical_closing_within_30nm", "relative_bearing_sin", "relative_bearing_cos",
                    "median_crossing_angle", "median_traffic_density", "min_topsky_predicted_sep_nm_proxy"]
    profile = events[["window_id", "stage09_label"]].merge(windows[["window_id", *feature_cols]],
                                                             on="window_id", validate="one_to_one")
    profile = profile.merge(scores, on="window_id", how="left", validate="one_to_one")
    records = []
    for label, group in profile.groupby("stage09_label"):
        for feature in [*feature_cols, "score"]:
            values = pd.to_numeric(group[feature], errors="coerce")
            records.append({"stage09_label": label, "feature": "Stage07_SCAT_FULL_XGBOOST_test_score" if feature == "score" else feature,
                            "n_events": len(group), "n_available": int(values.notna().sum()),
                            "median": float(values.median()) if values.notna().any() else np.nan,
                            "p25": float(values.quantile(.25)) if values.notna().any() else np.nan,
                            "p75": float(values.quantile(.75)) if values.notna().any() else np.nan,
                            "mean": float(values.mean()) if values.notna().any() else np.nan})
    pd.DataFrame(records).to_csv(OUT / "stage09_feature_comparison.csv", index=False)
    return grouped


def select_representative_cases(events: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for label in ALLOWED_LABELS:
        group = events.loc[(events.stage09_label == label) &
                           events.candidate_partner_id.notna() &
                           (events.candidate_partner_id != "")].copy()
        if group.empty:
            continue
        midpoint = group.evidence_confidence_score.median()
        group["median_distance"] = (group.evidence_confidence_score - midpoint).abs()
        group = group.sort_values(["median_distance", "event_epoch"], kind="mergesort")
        chosen = []
        for _, kind in group.groupby("clearance_type", sort=True):
            chosen.append(kind.iloc[0])
        already = {row.window_id for row in chosen}
        for row in group.itertuples(index=False):
            if len(chosen) >= 10:
                break
            if row.window_id not in already:
                chosen.append(row)
                already.add(row.window_id)
        picks.extend(chosen[:10])
    if not picks:
        result = pd.DataFrame(columns=["window_id", "stage09_label", "figure_path"])
    else:
        result = pd.DataFrame([row._asdict() if hasattr(row, "_asdict") else row.to_dict() for row in picks])
        result = result.drop(columns=["median_distance"], errors="ignore")
        result["figure_path"] = ""
    return result


def render_case_figure(item: dict, index: NativeStateIndex, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchor = float(item["event_epoch"])
    horizon = int(item["primary_horizon"]) if np.isfinite(item.get("primary_horizon", np.nan)) else 120
    horizon = max(60, min(horizon, 300))
    focal = index.last_before(item["flight_key"], anchor)
    other = index.last_before(item["candidate_partner_id"], anchor)
    if focal is None or other is None or not valid_kinematic_state(focal) or not valid_kinematic_state(other):
        return False
    f_robust, _, _ = index.robust_state(focal, anchor)
    o_robust, _, _ = index.robust_state(other, anchor)
    grid = np.arange(0, horizon + STEP, STEP, dtype=float)
    f_lat, f_lon, f_alt = propagate(f_robust, grid, anchor)
    o_lat, o_lon, o_alt = propagate(o_robust, grid, anchor)
    f_obs, _ = index.sampled_trajectory(focal["flight_key"], anchor, grid, focal)
    o_obs, _ = index.sampled_trajectory(other["flight_key"], anchor, grid, other)
    cf_h, cf_v = pair_series(f_lat, f_lon, f_alt, o_lat, o_lon, o_alt)
    obs_h, obs_v = pair_series(f_obs["latitude"], f_obs["longitude"], f_obs["altitude_ft"],
                              o_obs["latitude"], o_obs["longitude"], o_obs["altitude_ft"])
    mixed_h, mixed_v = pair_series(f_lat, f_lon, f_alt,
                                  o_obs["latitude"], o_obs["longitude"], o_obs["altitude_ft"])
    rolling_cpa = np.full(len(grid), np.nan)
    rolling_tcpa = np.full(len(grid), np.nan)
    for j, seconds in enumerate(grid[:-1]):
        if not all(np.isfinite(x[j]) for x in (f_obs["latitude"], f_obs["longitude"],
                                                 o_obs["latitude"], o_obs["longitude"],
                                                 f_obs["ground_speed_kt"], o_obs["ground_speed_kt"],
                                                 f_obs["track_deg"], o_obs["track_deg"])):
            continue
        f_state = {"time_epoch": anchor + seconds, "latitude": f_obs["latitude"][j],
                   "longitude": f_obs["longitude"][j], "altitude_ft": f_obs["altitude_ft"][j],
                   "ground_speed_kt": f_obs["ground_speed_kt"][j], "track_deg": f_obs["track_deg"][j],
                   "vertical_rate_fpm": 0.0}
        o_state = {"time_epoch": anchor + seconds, "latitude": o_obs["latitude"][j],
                   "longitude": o_obs["longitude"][j], "altitude_ft": o_obs["altitude_ft"][j],
                   "ground_speed_kt": o_obs["ground_speed_kt"][j], "track_deg": o_obs["track_deg"][j],
                   "vertical_rate_fpm": 0.0}
        cpa = finite_cpa(f_state, o_state, anchor + seconds, int(max(STEP, horizon - seconds)))
        rolling_cpa[j] = cpa["cpa_nm"]
        rolling_tcpa[j] = cpa["tcpa_s"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(f_obs["longitude"], f_obs["latitude"], color="tab:blue", label="focal observed")
    ax.plot(f_lon, f_lat, color="tab:blue", linestyle="--", label="focal no-intervention CV")
    ax.plot(o_obs["longitude"], o_obs["latitude"], color="tab:orange", label="neighbour observed")
    ax.scatter([focal["longitude"], other["longitude"]], [focal["latitude"], other["latitude"]],
               marker="x", color=["tab:blue", "tab:orange"], s=70, label="anchor positions")
    ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="Native positions and finite-horizon extrapolation")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(grid, obs_h, label="observed pair", color="black")
    ax.plot(grid, cf_h, label="symmetric no-intervention", color="tab:blue", linestyle="--")
    ax.plot(grid, mixed_h, label="focal CV + observed neighbour", color="tab:orange", linestyle=":")
    ax.set(xlabel="Seconds after anchor", ylabel="Horizontal separation (NM)", title="Pair horizontal geometry")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(grid, obs_v, label="observed pair", color="black")
    ax.plot(grid, cf_v, label="symmetric no-intervention", color="tab:blue", linestyle="--")
    ax.plot(grid, mixed_v, label="focal CV + observed neighbour", color="tab:orange", linestyle=":")
    ax.set(xlabel="Seconds after anchor", ylabel="Vertical separation (ft)", title="Pair vertical geometry")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(grid, rolling_cpa, color="tab:green", label="observed-state finite CPA")
    ax.set(xlabel="Seconds after anchor", ylabel="CPA within remaining horizon (NM)",
           title="Rolling observed-state CPA / TCPA")
    twin = ax.twinx()
    twin.plot(grid, rolling_tcpa, color="tab:red", alpha=.75, label="observed-state TCPA")
    twin.set_ylabel("TCPA within remaining horizon (s)")
    ax.legend(loc="upper left", fontsize=8)
    twin.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{item['stage09_label']} | {item['clearance_type']} | {item['window_id']} | {horizon}s", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return True


def representative_cases_and_figures(events: pd.DataFrame) -> pd.DataFrame:
    cases = select_representative_cases(events)
    if cases.empty:
        cases.to_csv(OUT / "representative_cases.csv", index=False)
        return cases
    by_day = cases.groupby(pd.to_datetime(cases.event_epoch, unit="s", utc=True).dt.floor("D"))
    for day, group in by_day:
        index = NativeStateIndex(load_day_states(day))
        for case in group.to_dict("records"):
            stem = case["window_id"].replace(":", "_")
            path = OUT / "figures" / "final_screen_v2" / f"{case['stage09_label']}_{stem}.png"
            if render_case_figure(case, index, path):
                cases.loc[cases.window_id == case["window_id"], "figure_path"] = str(path.relative_to(OUT))
    cases.to_csv(OUT / "representative_cases.csv", index=False)
    return cases


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for values in frame.loc[:, columns].head(limit).itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def quality_report(base: pd.DataFrame, candidate: pd.DataFrame, pair_horizon: pd.DataFrame,
                   annotated: pd.DataFrame, summary: pd.DataFrame, responses: pd.DataFrame,
                   horizon_summary: pd.DataFrame, unknown_counts: pd.DataFrame) -> dict:
    interventions = base.loc[base.ATC_INTERVENTION == 1]
    pairs_i = pair_horizon.loc[pair_horizon.ATC_INTERVENTION == 1]
    within30 = int(interventions.candidate_partner_count_30nm.gt(0).sum())
    only50 = int(((interventions.candidate_partner_count_30nm == 0) &
                  (interventions.candidate_partner_count_50nm > 0)).sum())
    none50 = int(interventions.candidate_partner_count_50nm.eq(0).sum())
    stale = int(interventions.quality_code.eq("STALE_STATE").sum())
    stale_neighbours = int(interventions.stale_neighbours_within_50nm.fillna(0).gt(0).sum())
    ambiguous = int(summary.candidate_partner_ambiguity.sum())
    censor_focal = int(summary.next_clearance_censored.sum())
    material = annotated.loc[annotated.usable & annotated.material_improvement &
                             (annotated.neighbourhood_nm <= 30)]
    scenario_agreement = float(material.counterfactual_model_agreement.mean()) if len(material) else np.nan
    response_counts = summary.clearance_response_confirmed.value_counts()
    facts = {"events": len(interventions), "within30": within30, "only_30_to_50": only50,
             "none_within50": none50, "stale_focal": stale,
             "events_with_stale_neighbour_candidate": stale_neighbours,
             "subsequent_clearance_censored_events": censor_focal,
             "candidate_partner_ambiguity_events": ambiguous,
             "scenario_agreement_among_usable_material_pair_horizons": scenario_agreement}
    lines = ["# Stage 09 cruise data quality and coverage", "",
             f"Exactly **{len(interventions):,}** first-week FL290–FL410 V2 intervention events and 4,098 V2 controls were reconciled. "
             "The control rows calibrate approximation error; they are not ATC counterfactual events.", "",
             f"- ≥1 candidate within 30 NM: **{within30:,}** events.",
             f"- Candidate only in 30–50 NM: **{only50:,}** events.",
             f"- No fresh candidate within 50 NM: **{none50:,}** events.",
             f"- Stale focal initial state: **{stale:,}** events; events with at least one stale neighbour candidate: **{stale_neighbours:,}**.",
             f"- Subsequent-clearance censoring in ≥1 main horizon: **{censor_focal:,}** events.",
             f"- Candidate/horizon partner ambiguity: **{ambiguous:,}** events.",
             f"- Agreement of instantaneous and observed-neighbour variants among usable, material 30-NM pair/horizon rows: **{scenario_agreement:.3f}**.",
             "- Pre-event native states are at most 6 s old; observed post-event samples must be within 3.5 s of each 5-s grid point. "
             "Positions are native plot positions, so state_age_s and position_age_s share the same timestamp. Large gaps are not silently interpolated.",
             "- 30/50 NM are *search radii*, never conflict labels or safety boundaries. All contemporaneous candidates are listed; detailed trajectories cover the union of top three 30-NM-priority and top three 50-NM CPA-ranked candidates.", "",
             "## Finite-horizon coverage and censoring", "",
             markdown_table(horizon_summary, list(horizon_summary.columns)), "",
             "Usability requires robust pre-event kinematics, fresh anchor states, ≥70% observed 5-s pair samples, an observed endpoint and no focal/partner actionable clearance before the horizon. "
             "The ≥70% criterion is a coverage-quality rule, not an aircraft-separation threshold.", "",
             "## Clearance response", "",
             markdown_table(pd.DataFrame({"status": ["YES", "PARTIAL", "NO", "UNKNOWN"],
                                          "n_events": [int(response_counts.get(s, 0)) for s in ["YES", "PARTIAL", "NO", "UNKNOWN"]]}),
                            ["status", "n_events"]), "",
             "A level command is assessed against its assigned flight level/altitude. Heading uses circular heading distance. "
             "SPEED response is at most PARTIAL because assigned airspeed/Mach and measured ground speed are not directly equivalent; "
             "missing or mixed-unit previous commands remain UNKNOWN. These choices avoid treating the existence of a clearance as evidence of response.", "",
             "## UNKNOWN reason codes", "",
             markdown_table(unknown_counts, ["unknown_reason", "n_events"], 30), "",
             "Reason counts are not mutually exclusive. UNKNOWN is preferred when coverage, censoring, model agreement, response or horizon consistency is inadequate."]
    (OUT / "data_quality_coverage_report.md").write_text("\n".join(lines) + "\n")
    return facts


def final_decision(calibration_stable: bool, labels: pd.DataFrame,
                   horizon_summary: pd.DataFrame, sensitivity: pd.DataFrame):
    counts = labels.stage09_label.value_counts()
    unknown = int(counts.get("UNKNOWN", 0))
    if unknown > len(labels) / 2 or horizon_summary.loc[horizon_summary.horizon_s == 300,
                                                        "events_with_usable_30nm_pair"].iloc[0] < len(labels) / 2:
        return "INSUFFICIENT_COUNTERFACTUAL_EVIDENCE"
    if not calibration_stable or (sensitivity.stability_descriptor == "UNSTABLE").any():
        return "REVISE_COUNTERFACTUAL"
    return "READY_FOR_STAGE10_COMPARISON"


def write_report(base: pd.DataFrame, candidates: pd.DataFrame, pairs: pd.DataFrame,
                 summary: pd.DataFrame, label_counts: pd.DataFrame, by_type: pd.DataFrame,
                 horizon_summary: pd.DataFrame, sensitivity: pd.DataFrame,
                 stage07: pd.DataFrame, cases: pd.DataFrame, quality: dict,
                 calibration_stable: bool, decision: str):
    calibration = pd.read_csv(OUT / "control_calibration_summary.csv")
    transfer = pd.read_csv(OUT / "stage09_feature_comparison.csv")
    response = pd.read_csv(OUT / "clearance_response_summary.csv")
    n_test_scores = int(stage07.n_test_interventions.sum())
    case_counts = cases.stage09_label.value_counts() if len(cases) else pd.Series(dtype=int)
    selected_features = transfer.loc[transfer.feature.isin([
        "min_horizontal_sep_within_30nm", "vertical_sep_at_event",
        "min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
        "min_topsky_predicted_sep_nm_proxy"])]
    lines = ["# SCAT Stage 09 — FL290–FL410 counterfactual kinematic/operational screen", "",
             "## 1. Scope", "",
             f"Only archive week `{PILOT_WEEK}`, `CRUISE_FL290_410`, `CRUISE_MATCHING_V2`; **4,098** intervention events. "
             "The 4,098 matched controls calibrate/validate simulator error, not ATC-event counterfactual labels. "
             "No Stage 01–08/10 rerun, other SCAT week or OpenSky processing occurred.", "",
             "## 2. Scientific question", "",
             "For a real cruise ATC intervention, does a finite-horizon estimate of *no focal intervention* show an evolving aircraft interaction "
             "that differs materially from the observed subsequent traffic geometry? This is distinct from Stage 07's intervention-versus-control prediction.", "",
             "## 3. Why this is not a causal counterfactual", "",
             "Constant kinematics does not know the pilot's unobserved decisions, controller intent, weather, coordination or a neighbour's unobserved alternative trajectory. "
             "The observed-neighbour scenario can itself have been influenced by the focal clearance. Therefore every result is a provisional "
             "`COUNTERFACTUAL_SCREEN_NOT_CAUSAL`, never a claim about what definitely would have happened.", "",
             "## 4. Input cohort", "",
             "[V2 pair cohort](04_windows/cruise_matching_v2.parquet), [V2 windows](05_features/window_features_cruise_matching_v2.parquet), "
             "[Stage 07 split assignments](07_explainability_cruise/split_assignments.parquet), existing Stage-02 native plots and Stage-03 clearances were read-only inputs. "
             "Exact event IDs, pair IDs, week, 29,000–41,000-ft anchors and clearance types were reconciled. "
             "Intervention types: 3,583 LEVEL, 458 SPEED, 57 HEADING.", "",
             "## 5. Control calibration", "",
             f"[Control calibration report](09_counterfactual_cruise/control_counterfactual_calibration_report.md): "
             f"**{'PASS' if calibration_stable else 'WARNING'}** on predeclared validation stability checks. "
             "TRAIN controls alone set 90th/95th-percentile absolute prediction-error magnitudes by horizon. "
             "VALIDATION checked stability; TEST was inspected only after freezing thresholds. "
             "These quantiles represent model error, not safety minima. Errors grow strongly with horizon, especially altitude.", "",
             markdown_table(calibration.loc[(calibration.split == "train") &
                                             (calibration.variable.isin(["pair_min_horizontal_error_nm",
                                                                          "pair_min_vertical_error_ft"]))],
                            ["horizon_s", "variable", "pair_rows", "p95_error"]), "",
             "## 6. Candidate-neighbour reconstruction", "",
             f"The Stage-02 first-week source contains 5,495,000 native rows; 5,493,745 within the seven nominal dates were cached in 22 resumable blocks. "
             f"All fresh contemporaneous candidates within 50 NM were enumerated: **{len(candidates):,}** candidate rows across 8,196 anchors. "
             f"Among interventions, {quality['within30']:,} had a candidate within 30 NM, {quality['only_30_to_50']:,} only in 30–50 NM, "
             f"and {quality['none_within50']:,} none within 50 NM. KD-tree prefiltering is followed by great-circle distance. "
             "The complete candidate table is retained, with detailed trajectories for the union of pre-event top-three candidates under the 30- and 50-NM rankings.", "",
             "## 7. Counterfactual models", "",
             "Primary: both focal and neighbour continue robust median/circular pre-event speed, track and vertical rate from up to 60 s of past native plots. "
             "Sensitivity A: both continue instantaneous anchor kinematics. Sensitivity B: focal continues robust pre-event kinematics while the neighbour follows native observed plots. "
             "Only pre-anchor states initialize the model. TopSky is SCAT_ONLY and appears only in post-hoc feature profiles; its publication timestamp is not fully verified.", "",
             "## 8. Horizon definitions", "",
             "Primary horizons are 60, 120 and 300 s; 600 s is sensitivity-only. Simulation and observed comparison use 5-s steps. "
             "Analytic constant-velocity CPA/TCPA is clipped to each finite horizon; raw signed TCPA is retained separately (negative means already diverging). "
             "Observed closest approach is the minimum of quality-valid native pair samples on that same finite grid. "
             "Positive observed-minus-estimated minimum separation means the observed pair was farther apart than the no-intervention estimate; it is not automatically risk reduction.", "",
             "## 9. Clearance censoring", "",
             "For each focal and candidate partner, the next actionable Stage-03 clearance is searched after the anchor. "
             "An outcome is censored at a horizon if either flight receives a new clearance before that horizon. "
             f"{quality['subsequent_clearance_censored_events']:,} interventions have such censoring in at least one primary horizon; shorter unaffected horizons remain usable. "
             "`FOCAL_NEXT_CLEARANCE` and `NEIGHBOUR_NEXT_CLEARANCE` are persisted per pair/horizon.", "",
             "## 10. Clearance-response validation", "",
             "Response must be compatible with the actual instruction; a recorded clearance is never sufficient by itself. "
             "LEVEL checks progress toward assigned altitude, HEADING uses circular heading, and SPEED uses only direction of ground-speed change when a comparable previous assigned unit exists; "
             "SPEED is at most PARTIAL because assigned airspeed/Mach is not ground speed. "
             "The detailed [response table](09_counterfactual_cruise/clearance_response_summary.csv) and "
             "[quality report](09_counterfactual_cruise/data_quality_coverage_report.md) show UNKNOWN reasons.", "",
             "## 11. Materiality calibration", "",
             "The primary materiality criterion is a positive observed-minus-estimated geometry/timing change exceeding the matching TRAIN-control 95th-percentile absolute prediction error at that horizon; "
             "the 90th percentile is sensitivity only. Evidence also requires pre-event finite-horizon convergence and agreement with instantaneous and observed-neighbour variants. "
             "A manual physical audit found that a large vertical-only delta can occur for a pair still very far apart. "
             "The revised primary gate requires material **joint** horizontal+vertical separation change "
             "(vertical feet converted to NM in a continuous Euclidean proxy). Only SPEED may alternatively use a corroborated material CPA timing change, also calibrated against control error; timing alone cannot label LEVEL or HEADING. "
             "It does not require both horizontal and vertical effects individually, and it is not a fixed physical conflict boundary. "
             "No 5-NM/1,000-ft, 7.5-NM/1,200-ft or colour-state rule enters a label.", "",
             "## 12. Label definitions", "",
             "`LIKELY_INTERACTION_RELATED` requires a fresh partner, at least two primary horizons with the *same partner* showing pre-event convergence and material observed improvement, "
             "confirmed/partial response, model-scenario agreement and no invalidating second clearance. "
             "`LIKELY_NON_INTERACTION_OPERATIONAL` requires a YES response, complete evaluated neighbour set within 30 NM, two usable horizons and no candidate showing material/convergent evidence. "
             "It means no pairwise interaction evidence in available data, not definitely unrelated to safety. "
             "Everything else is `UNKNOWN`, with reason codes. "
             "`evidence_confidence_score` is a transparent 0–1 evidence-quality index, **not a conflict probability**: "
             "0.25×main-horizon coverage + 0.20×response (YES=1, PARTIAL=.5) + 0.20×scenario agreement + "
             "0.20×supporting-horizon fraction + 0.15×effect/error magnitude (capped at 1); UNKNOWN is capped at .49. "
             "The score does not use Stage 07 or a learned model.", "",
             "## 13. Data coverage", "",
             markdown_table(horizon_summary, ["horizon_s", "events_with_usable_30nm_pair",
                                              "focal_next_clearance_pair_rows", "neighbour_next_clearance_pair_rows"]), "",
             f"Only {int(horizon_summary.loc[horizon_summary.horizon_s == 300, 'events_with_usable_30nm_pair'].iloc[0]):,} interventions have a usable 30-NM pair at 300 s. "
             "Later horizons are particularly limited by observation coverage and new clearances. See the dedicated coverage report.", "",
             "## 14. Overall label results", "",
             markdown_table(label_counts, ["stage09_label", "n", "percent"]), "",
             "The large UNKNOWN fraction is a substantive result of the conservative screen; it must not be redistributed among the other two labels.", "",
             "## 15. Results by clearance type", "",
             markdown_table(by_type, ["clearance_type", "n_events", "LIKELY_INTERACTION_RELATED_n",
                                      "LIKELY_NON_INTERACTION_OPERATIONAL_n", "UNKNOWN_n",
                                      "mean_evidence_confidence_score", "usable_120s_percent"]), "",
             "LEVEL commands dominate this cruise cohort, but a level instruction alone is not vertical-interaction evidence. "
             "SPEED response is often unconfirmable from ground speed versus assigned airspeed/Mach; HEADING has a small sample.", "",
             "## 16. Results by horizon", "",
             markdown_table(horizon_summary, ["horizon_s", "LIKELY_INTERACTION_RELATED",
                                              "LIKELY_NON_INTERACTION_OPERATIONAL", "UNKNOWN"]), "",
             "Horizon labels precede the final two-horizon/partner-consistency rule; 600-s labels are diagnostic only.", "",
             "## 17. Sensitivity analysis", "",
             markdown_table(sensitivity, ["configuration", "LIKELY_INTERACTION_RELATED",
                                          "LIKELY_NON_INTERACTION_OPERATIONAL", "UNKNOWN",
                                          "agreement_among_primary_classified", "stability_descriptor"]), "",
             "The 30/50-NM radii define candidate search, not conflict. The variants were predeclared and were not selected to maximize interaction-related cases. "
             "Disagreement, especially among the few positive labels, constrains interpretation. "
             "STABLE requires ≥80% agreement among primary classified events and each classified label count within 0.8–1.25× its primary count; otherwise ≥60% agreement is PARTIAL.", "",
             "## 18. Representative cases", "",
             f"[Representative case index](09_counterfactual_cruise/representative_cases.csv) selects "
             f"{int(case_counts.get('LIKELY_INTERACTION_RELATED', 0))} interaction-related, "
             f"{int(case_counts.get('LIKELY_NON_INTERACTION_OPERATIONAL', 0))} non-interaction and "
             f"{int(case_counts.get('UNKNOWN', 0))} unknown cases closest to each label's median confidence, with clearance-type diversity where available. "
             "Figures show observed focal/neighbour native paths, focal no-intervention extrapolation, finite-horizon horizontal/vertical separation and rolling CPA/TCPA. "
             "No safety-minimum reference lines are drawn.", "",
             "## 19. Stage 07 post-hoc comparison", "",
             f"Only **{n_test_scores}** Stage-07 TEST intervention scores exist; no Stage-07 model was rerun and train/validation predictions were not fabricated. "
             "The scores were joined **after** Stage-09 labels were frozen and were never used for labeling. "
             "They discriminate ATC intervention versus matched control, not interaction-relatedness or population intervention probability.", "",
             markdown_table(stage07, ["stage09_label", "n_test_interventions", "median_stage07_score",
                                      "mean_stage07_score"]), "",
             "## 20. Feature profiles by Stage 09 label", "",
             markdown_table(selected_features, ["stage09_label", "feature", "n_available", "median"]), "",
             "The [complete profile](09_counterfactual_cruise/stage09_feature_comparison.csv) includes separation, CPA, closing, circular bearing, crossing, density and SCAT-only TopSky. "
             "These post-hoc profiles did not feed back into label rules. Stage-07 scores are available for TEST intervention windows only.", "",
             "## 21. Implications for OpenSky", "",
             "OpenSky could later reproduce a quality-controlled observed state sequence, constant-kinematic extrapolation, neighbour search, pair geometry, separation, CPA/TCPA and closing. "
             "It cannot natively supply the ATC event, clearance type/command or TopSky trajectory, so Stage-09 labels cannot be generated directly there. "
             "This screen only suggests traffic patterns worth exploring once a separately authorized OpenSky pilot exists; no OpenSky rows were processed now.", "",
             "## 22. Limitations", "",
             "No causal identification; pilot/controller intent, wind and airspeed-vs-groundspeed differences are unmodelled. "
             "Neighbour observed paths may depend on the focal action. Search-radius and top-candidate choices create selection effects. "
             "An unobserved clearance or gap can invalidate apparent no-intervention geometry. The 600-s constant-kinematic approximation has large altitude error. "
             "The two provisional positive cases are supported at nested 60/120-s horizons by the same early encounter, not two independent encounters; neither belongs to the Stage-07 TEST score subset. "
             "CONTROL thresholds come from one matched week, not a population safety baseline. "
             "The small intervention-related count and UNKNOWN dominance prohibit strong prevalence or conflict claims.", "",
             "## 23. Recommendation for Stage 10 / scaling", "",
             f"**{decision}**. The finite-horizon engine and control calibration pass their checks, but the conservative screen's coverage and UNKNOWN dominance "
             "prevent a strong operational bridge at present. Review data availability/response measurement before Stage 10 or scaling. "
             "Do not run Stage 08 or Stage 10, process another SCAT week or process OpenSky under this authorization.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def terminal_summary(summary: pd.DataFrame, base: pd.DataFrame, label_counts: pd.DataFrame,
                     by_type: pd.DataFrame, horizon_summary: pd.DataFrame,
                     sensitivity: pd.DataFrame, stage07: pd.DataFrame,
                     calibration_stable: bool, decision: str):
    train = pd.read_csv(OUT / "control_calibration_summary.csv")
    print("=" * 60, "SCAT STAGE 09 — FL290–FL410 COUNTERFACTUAL SCREEN", "=" * 60,
          f"Week: {PILOT_WEEK}", f"Cohort: {COHORT}", "Interventions: 4098",
          "Controls evaluated: 4098; thresholds fit on usable TRAIN controls only",
          "-" * 60, "CONTROL CALIBRATION", sep="\n")
    for h in HORIZONS:
        a = train.loc[(train.split == "train") & (train.horizon_s == h) &
                      (train.variable == "horizontal_position_error_nm")].iloc[0]
        b = train.loc[(train.split == "train") & (train.horizon_s == h) &
                      (train.variable == "altitude_error_ft")].iloc[0]
        print(f"{h} s: horizontal error p95 = {a.p95_error:.3f} NM; vertical error p95 = {b.p95_error:.1f} ft")
    print(f"Validation calibration: {'PASS' if calibration_stable else 'WARNING'}")
    intervention = base.loc[base.ATC_INTERVENTION == 1]
    print("-" * 60, "DATA COVERAGE", sep="\n")
    print(f"Events with candidate within 30 NM: {intervention.candidate_partner_count_30nm.gt(0).sum()}")
    print(f"Events with candidate only within 30–50 NM: {((intervention.candidate_partner_count_30nm == 0) & (intervention.candidate_partner_count_50nm > 0)).sum()}")
    print(f"No suitable candidate: {intervention.candidate_partner_count_50nm.eq(0).sum()}")
    for h in HORIZONS:
        print(f"Usable at {h} s: {summary[f'usable_{h}s'].sum()}")
    print(f"Censored by subsequent clearance (≥1 main horizon): {summary.next_clearance_censored.sum()}")
    print("-" * 60, "CLEARANCE RESPONSE", sep="\n")
    for status in ("YES", "PARTIAL", "NO", "UNKNOWN"):
        print(f"{status}: {(summary.clearance_response_confirmed == status).sum()}")
    print("-" * 60, "COUNTERFACTUAL LABELS", sep="\n")
    for row in label_counts.itertuples(index=False):
        print(f"{row.stage09_label}: n = {row.n}; % = {row.percent:.1f}")
    print("-" * 60, "BY CLEARANCE TYPE", sep="\n")
    for row in by_type.itertuples(index=False):
        print(f"{row.clearance_type}: interaction-related = {getattr(row, 'LIKELY_INTERACTION_RELATED_n')}, "
              f"non-interaction = {getattr(row, 'LIKELY_NON_INTERACTION_OPERATIONAL_n')}, "
              f"unknown = {getattr(row, 'UNKNOWN_n')}")
    print("-" * 60, "SENSITIVITY", sep="\n")
    for row in sensitivity.itertuples(index=False):
        print(f"{row.configuration}: {row.stability_descriptor} (classified agreement {row.agreement_among_primary_classified:.3f})")
    print("-" * 60, "STAGE 07 POST-HOC — TEST ONLY", sep="\n")
    for row in stage07.itertuples(index=False):
        print(f"{row.stage09_label}: n = {row.n_test_interventions}; median score = {row.median_stage07_score:.3f}")
    print("No Stage 07 score was used to create Stage 09 labels.")
    print("-" * 60, "INTERPRETATION", sep="\n")
    print("Estimates provisional evidence of aircraft-interaction-related ATC intervention under finite-horizon kinematics.")
    print("NOT confirmed conflict, causality, loss of separation, or population conflict probability.")
    print("-" * 60, "FINAL DECISION", decision, "DO NOT RUN STAGE 10 OR STAGE 08.",
          "DO NOT PROCESS OTHER SCAT WEEKS OR OPENSKY.", "=" * 60, sep="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-cache", action="store_true", help="Only prepare checkpointed first-week minimal native-state cache")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only Stage 09 outputs/report")
    args = parser.parse_args()
    pairs, events, _, next_by_flight = read_inputs()
    if (OUT / "counterfactual_event_summary.parquet").exists() and not args.overwrite and not args.prepare_cache:
        raise SystemExit("Completed Stage 09 output already exists; inspect it before an explicit --overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = cache_first_week_states()
    if args.prepare_cache:
        print(f"Stage 09 cache ready: {manifest['cached_rows_nominal_days']:,} nominal-day first-week states")
        return
    config = {"analysis_scope": SCOPE, "week": PILOT_WEEK, "band": BAND,
              "cohort": COHORT, "method": METHOD, "target_events": 4098,
              "control_anchors": 4098, "horizons_s": HORIZONS, "main_horizons_s": MAIN_HORIZONS,
              "step_s": STEP, "anchor_max_age_s": ANCHOR_MAX_AGE_S,
              "observed_grid_max_age_s": OBS_MAX_AGE_S, "minimum_pair_grid_coverage": MIN_OBS_COVERAGE,
              "candidate_radii_nm": [30, 50], "top_candidates_per_radius": N_PARTNERS,
              "simulator_baselines": ["robust_pre_event_symmetric", "instantaneous_anchor_symmetric",
                                       "robust_pre_event_focal_with_observed_neighbour"],
              "calibration_training_controls_only": True, "materiality_percentiles": [.95, .90],
              "interaction_materiality_rule": "LEVEL/HEADING require corroborated joint geometry; SPEED permits corroborated joint geometry or timing; all materiality is TRAIN-control-error calibrated",
              "stage07_scores_for_labeling": False, "topsky_for_labeling": False,
              "safety_threshold_for_labeling": False,
              "confidence_formula": "0.25*horizon_coverage + 0.20*response + 0.20*scenario_agreement + 0.20*supporting_horizon_fraction + 0.15*joint_or_timing_effect_over_error; cap UNKNOWN at 0.49",
              "joint_separation_proxy": "sqrt(horizontal_sep_nm^2 + (vertical_sep_ft/6076.12)^2); control-error calibrated; not a safety threshold",
              "allowed_labels": ALLOWED_LABELS, "day_algorithm_version": DAY_ALGORITHM_VERSION,
              "source_states": str(ROOT / "02_traffic/scat_states.parquet"),
              "source_clearances": str(ROOT / "03_clearances/scat_clearances.parquet"),
              "source_pairs": str(ROOT / "04_windows/cruise_matching_v2.parquet"),
              "source_stage07_split": str(ROOT / "07_explainability_cruise/split_assignments.parquet"),
              "cache_manifest": manifest}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2))
    for day in DAYS:
        subset = events.loc[(events.event_epoch >= day.timestamp()) &
                            (events.event_epoch < (day + pd.Timedelta(days=1)).timestamp())]
        process_day(day, subset, next_by_flight)
    frames = collect_day_outputs()
    base = frames["events"]
    candidates = frames["candidates"]
    pair_horizon = frames["pair_horizon"]
    control_calibration = frames["control_calibration"]
    assert len(base) == 8196 and base.ATC_INTERVENTION.sum() == 4098
    thresholds, control_summary, stable = calibrate_controls(control_calibration)
    responses = add_clearance_responses(base, thresholds)
    annotated = annotate_evidence(pair_horizon, responses, thresholds)
    summary = classify_events(base, annotated, responses)
    assert len(summary) == 4098 and summary.matched_pair_id.is_unique
    # Freeze all primary labels BEFORE reading any Stage-07 prediction.
    summary.to_parquet(OUT / "counterfactual_event_summary.parquet", index=False, compression="zstd")
    candidates.to_parquet(OUT / "candidate_neighbours.parquet", index=False, compression="zstd")
    pair_horizon.to_parquet(OUT / "counterfactual_pair_horizon.parquet", index=False, compression="zstd")
    annotated.to_parquet(OUT / "intervention_pair_evidence.parquet", index=False, compression="zstd")
    print(f"Stage 09: frozen 4,098 provisional labels: {summary.stage09_label.value_counts().to_dict()}", flush=True)
    label_counts, by_type = write_label_tables(summary, pair_horizon)
    horizon_summary = summarize_horizons(summary, pair_horizon)
    sensitivity = sensitivity_analyses(base, pair_horizon, responses, thresholds, summary)
    unknown_counts = pd.read_csv(OUT / "unknown_reason_counts.csv")
    quality = quality_report(base, candidates, pair_horizon, annotated, summary,
                             responses, horizon_summary, unknown_counts)
    windows = pd.read_parquet(ROOT / "05_features/window_features_cruise_matching_v2.parquet")
    stage07 = posthoc_stage07(summary, windows)
    cases = representative_cases_and_figures(summary)
    decision = final_decision(stable, summary, horizon_summary, sensitivity)
    write_report(base, candidates, pair_horizon, summary, label_counts, by_type,
                 horizon_summary, sensitivity, stage07, cases, quality, stable, decision)
    terminal_summary(summary, base, label_counts, by_type, horizon_summary,
                     sensitivity, stage07, stable, decision)


if __name__ == "__main__":
    main()






