#!/usr/bin/env python3
"""Stage 09 V2: first-week clearance-aware, non-causal cruise screen.

V1 files are read-only KINEMATIC_COUNTERFACTUAL_BASELINE inputs. This script
never runs the legacy Stage 09 or modifies its outputs or source.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK
from cruise_counterfactual_kinematics import (
    angular_difference_deg, finite_cpa, joint_separation_nm, pair_series,
    propagate, xy_from_latlon,
)
from cruise_clearance_aware_kinematics import (
    MAX_PREVIOUS_INSTRUCTION_AGE_S, assigned_level_ft,
    classify_previous_level_intent, comparable_speed_delta,
    has_same_partner_across_horizons, previous_heading_track,
    previous_level_altitudes, scenarios_not_opposed, speed_trend_path,
)


V1 = importlib.import_module("09_counterfactual_clearance_analysis_cruise")
ROOT = DEFAULT_OUTPUT_ROOT
V1_OUT = ROOT / "09_counterfactual_cruise"
OUT = ROOT / "09_counterfactual_cruise_v2"
REPORT = ROOT / "PILOT_STAGE09_V2_CLEARANCE_AWARE_REPORT.md"
HORIZONS = (60, 120, 300, 600)
MAIN = (60, 120)
STEP = 5
METHOD = "COUNTERFACTUAL_SCREEN_NOT_CAUSAL"
COHORT = "CRUISE_MATCHING_V2"
BAND = "CRUISE_FL290_410"
ALLOWED = ("LIKELY_INTERACTION_RELATED", "LIKELY_NON_INTERACTION_OPERATIONAL", "UNKNOWN")
VERSION = 3
FIRST_DAY = pd.Timestamp("2016-10-15", tz="UTC")
DAYS = [FIRST_DAY + pd.Timedelta(days=i) for i in range(7)]


def qvalue(thresholds: pd.DataFrame, horizon: int, variable: str, percentile: float) -> float:
    row = thresholds.loc[(thresholds.horizon_s == horizon) & (thresholds.variable == variable) &
                         np.isclose(thresholds.percentile, percentile)]
    if len(row) != 1:
        raise ValueError((horizon, variable, percentile, len(row)))
    return float(row.threshold.iloc[0])


def optional_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def iso_epoch(text):
    try:
        return pd.Timestamp(text, tz="UTC").timestamp()
    except (TypeError, ValueError):
        return np.nan


def clearance_components(previous: dict, current: dict):
    old_level = assigned_level_ft(previous.get("cfl"), previous.get("cfl_unit"))
    new_level = assigned_level_ft(current.get("cfl"), current.get("cfl_unit"))
    old_heading = optional_number(previous.get("assigned_heading_val"))
    new_heading = optional_number(current.get("assigned_heading_val"))
    old_speed = optional_number(previous.get("assigned_speed_val"))
    new_speed = optional_number(current.get("assigned_speed_val"))
    old_unit = previous.get("assigned_speed_unit")
    new_unit = current.get("assigned_speed_unit")
    delta_level = new_level - old_level if np.isfinite(old_level) and np.isfinite(new_level) else np.nan
    delta_heading = (float(angular_difference_deg(new_heading, old_heading))
                     if np.isfinite(old_heading) and np.isfinite(new_heading) else np.nan)
    delta_speed = comparable_speed_delta(old_speed, old_unit, new_speed, new_unit)
    # A newly assigned or cancelled command is still a transition, but its
    # magnitude may be unidentifiable and cannot initialize a V2 scenario.
    level_change = np.isfinite(delta_level) and abs(delta_level) >= 1
    heading_change = ((np.isfinite(delta_heading) and abs(delta_heading) >= .5) or
                      np.isfinite(old_heading) != np.isfinite(new_heading))
    speed_change = ((np.isfinite(delta_speed) and abs(delta_speed) > 1e-9) or
                    np.isfinite(old_speed) != np.isfinite(new_speed) or old_unit != new_unit)
    kinds = [name for name, changed in (("LEVEL", level_change),
                                        ("HEADING", heading_change),
                                        ("SPEED", speed_change)) if changed]
    transition = ("MULTI_ACTION" if len(kinds) > 1 else
                  kinds[0] + "_CHANGE" if kinds else
                  "UNKNOWN_CHANGE" if any(not np.isfinite(v) for v in
                                          (delta_level, delta_heading, delta_speed)) else
                  "NO_EFFECTIVE_CHANGE")
    return {"previous_level_ft": old_level, "current_level_ft": new_level,
            "previous_heading": old_heading, "current_heading": new_heading,
            "previous_speed": old_speed, "current_speed": new_speed,
            "previous_speed_unit": old_unit, "current_speed_unit": new_unit,
            "delta_CFL_ft": delta_level, "circular_delta_heading": delta_heading,
            "delta_speed_comparable": delta_speed,
            "transition_class": transition, "changed_components": "+".join(kinds)}


def read_inputs():
    v1 = pd.read_parquet(V1_OUT / "counterfactual_event_summary.parquet")
    assert len(v1) == 4098 and v1.window_id.is_unique and v1.matched_pair_id.is_unique
    assert set(v1.archive_week) == {PILOT_WEEK} and v1.analysis_altitude_ft.between(29000, 41000).all()
    assert v1.clearance_type.value_counts().to_dict() == {"LEVEL": 3583, "SPEED": 458, "HEADING": 57}
    candidates = pd.read_parquet(V1_OUT / "candidate_neighbours.parquet")
    pairs = pd.read_parquet(V1_OUT / "counterfactual_pair_horizon.parquet")
    assert set(candidates.window_id.loc[candidates.ATC_INTERVENTION == 1]) <= set(v1.window_id)
    assert set(pairs.horizon_s) == set(HORIZONS)
    evidence = pd.read_parquet(V1_OUT / "intervention_pair_evidence.parquet")
    responses = pd.read_csv(V1_OUT / "clearance_response_summary.csv")
    thresholds = pd.read_csv(V1_OUT / "calibration_thresholds.csv")
    calibration = pd.read_csv(V1_OUT / "control_counterfactual_calibration.csv")
    day_events = pd.concat([pd.read_parquet(V1_OUT / "day_checkpoints" /
                                           f"day_{day.strftime('%Y%m%d')}_events.parquet")
                            for day in DAYS], ignore_index=True)
    assert len(day_events) == 8196 and day_events.window_id.is_unique
    assert set(v1.window_id) == set(day_events.loc[day_events.ATC_INTERVENTION == 1, "window_id"])
    return v1, candidates, pairs, evidence, responses, thresholds, calibration, day_events


def build_history(v1: pd.DataFrame):
    cols = ["clearance_id", "flight_key", "archive_week", "clearance_epoch",
            "clearance_type", "clearance_record_json", "previous_record_json",
            "previous_value", "new_value", "changed_fields"]
    all_clearances = pd.read_parquet(ROOT / "03_clearances/scat_clearances.parquet", columns=cols)
    week = all_clearances.loc[all_clearances.archive_week == PILOT_WEEK].copy()
    week = week.sort_values(["flight_key", "clearance_epoch", "clearance_id"], kind="mergesort")
    group = week.groupby("flight_key", sort=False)
    for column in ("clearance_id", "clearance_epoch", "clearance_type", "clearance_record_json"):
        week["previous_action_" + column] = group[column].shift(1)
    week["next_action_clearance_epoch"] = group.clearance_epoch.shift(-1)
    week["next_action_clearance_id"] = group.clearance_id.shift(-1)
    selected = week.loc[week.clearance_id.isin(v1.clearance_id)].copy()
    assert len(selected) == 4098 and selected.clearance_id.is_unique
    selected = v1[["window_id", "matched_pair_id", "clearance_id", "flight_key",
                   "event_epoch", "archive_week", "analysis_altitude_ft", "clearance_type"]].merge(
                       selected.drop(columns=["archive_week", "clearance_type"]),
                       on=["clearance_id", "flight_key"], validate="one_to_one")
    assert np.allclose(selected.event_epoch, selected.clearance_epoch)
    rows = []
    for item in selected.to_dict("records"):
        previous = json.loads(item["previous_record_json"])
        current = json.loads(item["clearance_record_json"])
        prev_action_time = optional_number(item["previous_action_clearance_epoch"])
        previous_record_time = iso_epoch(previous.get("time_stamp"))
        # The V1 source has one preceding actionable record per pilot event.
        assert np.isfinite(prev_action_time) and abs(prev_action_time - previous_record_time) < .01
        assert item["previous_action_clearance_record_json"] == item["previous_record_json"]
        fields = clearance_components(previous, current)
        rows.append({"window_id": item["window_id"], "matched_pair_id": int(item["matched_pair_id"]),
                     "clearance_id": item["clearance_id"], "flight_key": item["flight_key"],
                     "archive_week": PILOT_WEEK, "analysis_altitude_ft": item["analysis_altitude_ft"],
                     "event_epoch": item["event_epoch"], "clearance_type": item["clearance_type"],
                     "previous_clearance_id": item["previous_action_clearance_id"],
                     "previous_clearance_time": prev_action_time,
                     "previous_clearance_type": item["previous_action_clearance_type"],
                     "previous_instruction_age_s": item["event_epoch"] - prev_action_time,
                     "previous_instruction_fresh": 0 <= item["event_epoch"] - prev_action_time <= MAX_PREVIOUS_INSTRUCTION_AGE_S,
                     "previous_level": previous.get("cfl"), "previous_level_unit": previous.get("cfl_unit"),
                     "previous_heading": fields["previous_heading"],
                     "previous_speed": fields["previous_speed"],
                     "previous_speed_unit": fields["previous_speed_unit"],
                     "current_clearance_time": item["clearance_epoch"],
                     "current_level": current.get("cfl"), "current_level_unit": current.get("cfl_unit"),
                     "current_heading": fields["current_heading"],
                     "current_speed": fields["current_speed"],
                     "current_speed_unit": fields["current_speed_unit"],
                     "next_clearance_time": item["next_action_clearance_epoch"],
                     "next_clearance_id": item["next_action_clearance_id"],
                     "previous_raw_values": item["previous_record_json"],
                     "current_raw_values": item["clearance_record_json"],
                     "previous_value": item["previous_value"], "new_value": item["new_value"],
                     "changed_fields": item["changed_fields"],
                     **fields})
    result = pd.DataFrame(rows)
    assert len(result) == 4098 and result.window_id.is_unique
    assert result.clearance_type.value_counts().to_dict() == {"LEVEL": 3583, "SPEED": 458, "HEADING": 57}
    result.to_parquet(OUT / "clearance_history.parquet", index=False, compression="zstd")
    result.groupby(["clearance_type", "transition_class"], dropna=False).size().rename("n_events").reset_index().to_csv(
        OUT / "clearance_transition_summary.csv", index=False)
    return result


def pre_event_metrics(index, flight_key: str, epoch: float):
    positions = index.by_flight.get(flight_key)
    if positions is None:
        return {"pre_samples": 0, "altitude_trend_fpm": np.nan,
                "speed_trend_kt_per_s": np.nan, "pre_track_spread_deg": np.nan,
                "pre_speed_range_kt": np.nan}
    times = index.times[positions]
    lo = np.searchsorted(times, epoch - 60, side="left")
    hi = np.searchsorted(times, epoch, side="right")
    sample = index.states.iloc[positions[lo:hi]]
    sample = sample.loc[sample[["time_epoch", "altitude_ft", "ground_speed_kt", "track_deg"]].notna().all(axis=1)]
    if len(sample) < 3 or sample.time_epoch.max() - sample.time_epoch.min() < 10:
        return {"pre_samples": len(sample), "altitude_trend_fpm": np.nan,
                "speed_trend_kt_per_s": np.nan, "pre_track_spread_deg": np.nan,
                "pre_speed_range_kt": np.nan}
    first = sample.head(max(1, len(sample) // 3))
    last = sample.tail(max(1, len(sample) // 3))
    elapsed = float(last.time_epoch.median() - first.time_epoch.median())
    trend_alt = float((last.altitude_ft.median() - first.altitude_ft.median()) / elapsed * 60)
    trend_speed = float((last.ground_speed_kt.median() - first.ground_speed_kt.median()) / elapsed)
    centre = float(sample.track_deg.iloc[-1])
    spread = float(np.median(np.abs(angular_difference_deg(sample.track_deg.to_numpy(), centre))))
    return {"pre_samples": len(sample), "altitude_trend_fpm": trend_alt,
            "speed_trend_kt_per_s": trend_speed, "pre_track_spread_deg": spread,
            "pre_speed_range_kt": float(sample.ground_speed_kt.max() - sample.ground_speed_kt.min())}


def dynamic_regime(pre: dict, robust_rate_fpm: float):
    if pre["pre_samples"] < 3 or not np.isfinite(pre["altitude_trend_fpm"]):
        return "UNCLASSIFIED"
    rate, trend = robust_rate_fpm, pre["altitude_trend_fpm"]
    if abs(rate) <= 150 and abs(trend) <= 150:
        return "LEVEL_LIKE"
    if abs(rate) >= 300 and abs(trend) >= 300 and np.sign(rate) == np.sign(trend):
        return "CLIMB_DESCENT_LIKE"
    return "OTHER_VERTICAL"


def focal_clearance_aware_state(history: dict, focal: dict, robust: dict, pre: dict):
    kind = history["clearance_type"]
    anchor = float(history["event_epoch"])
    grid = np.arange(0, 601, STEP, dtype=float)
    lat_b, lon_b, alt_b = propagate(robust, grid, anchor)
    baseline = (lat_b, lon_b, alt_b)
    quality = "OK"
    intent = "NOT_APPLICABLE"
    available = False
    if kind == "LEVEL":
        intent, quality = classify_previous_level_intent(
            history["previous_level_ft"], history["previous_instruction_age_s"],
            float(alt_b[0]), float(robust["vertical_rate_fpm"]), pre["altitude_trend_fpm"])
        if not np.isfinite(history["delta_CFL_ft"]) or abs(history["delta_CFL_ft"]) < 1:
            quality = "NO_EFFECTIVE_LEVEL_CHANGE"
        available = quality == "OK"
        alt_a = previous_level_altitudes(alt_b[0], history["previous_level_ft"],
                                         robust["vertical_rate_fpm"], intent, grid) if available else np.full_like(grid, np.nan)
        aware = (lat_b, lon_b, alt_a)
    elif kind == "HEADING":
        track = previous_heading_track(history["previous_heading"], focal.get("heading_deg"),
                                       focal.get("track_deg"), history["previous_instruction_age_s"])
        if not np.isfinite(track):
            quality = "PREVIOUS_HEADING_UNAVAILABLE_OR_INACTIVE"
        elif not np.isfinite(history["current_heading"]):
            quality = "NEW_HEADING_UNAVAILABLE"
        elif not np.isfinite(history["circular_delta_heading"]) or abs(history["circular_delta_heading"]) < .5:
            quality = "NO_EFFECTIVE_HEADING_CHANGE"
        available = quality == "OK"
        if available:
            heading_state = {**robust, "track_deg": track}
            aware = propagate(heading_state, grid, anchor)
        else:
            aware = tuple(np.full_like(grid, np.nan) for _ in range(3))
        intent = "PREVIOUS_HEADING_ACTIVE" if available else "PREVIOUS_HEADING_INTENT_UNKNOWN"
    elif kind == "SPEED":
        delta = history["delta_speed_comparable"]
        if not history["previous_instruction_fresh"] or not np.isfinite(delta) or abs(delta) <= 1e-9:
            quality = "SPEED_COMMAND_UNITS_OR_HISTORY_UNAVAILABLE"
        elif not np.isfinite(pre["speed_trend_kt_per_s"]):
            quality = "INSUFFICIENT_PRE_SPEED_TREND"
        else:
            speed = speed_trend_path(robust["ground_speed_kt"], pre["speed_trend_kt_per_s"], grid)
            if not np.isfinite(speed).all():
                quality = "UNSTABLE_PRE_SPEED_TREND"
        available = quality == "OK"
        if available:
            anchor_lat, anchor_lon, anchor_alt = lat_b[0], lon_b[0], alt_b[0]
            distance = (float(robust["ground_speed_kt"]) * grid +
                        .5 * pre["speed_trend_kt_per_s"] * grid * grid) / 3600.0
            angle = np.deg2rad(float(robust["track_deg"]))
            aware = (anchor_lat + distance * np.cos(angle) / 60,
                     anchor_lon + distance * np.sin(angle) /
                     (60 * np.cos(np.deg2rad(anchor_lat))), alt_b)
        else:
            aware = tuple(np.full_like(grid, np.nan) for _ in range(3))
        intent = "PREVIOUS_SPEED_DIRECTION_WITH_OBSERVED_TREND" if available else "PREVIOUS_SPEED_INTENT_UNKNOWN"
    else:
        quality = "MULTI_ACTION_AMBIGUOUS" if "+" in kind else "UNSUPPORTED_CLEARANCE_TYPE"
        aware = tuple(np.full_like(grid, np.nan) for _ in range(3))
    return {"aware": aware, "baseline": baseline, "grid": grid,
            "available": available, "intent_state": intent, "quality": quality}


def pair_counterfactual_rows(history: dict, focal_model: dict, neighbour: dict,
                             neighbour_robust: dict, original: pd.DataFrame,
                             thresholds: pd.DataFrame, response_lookup: dict,
                             observed_focal_altitude: dict):
    anchor = float(history["event_epoch"])
    grid = focal_model["grid"]
    neighbour_lat, neighbour_lon, neighbour_alt = propagate(neighbour_robust, grid, anchor)
    focal_lat, focal_lon, focal_alt = focal_model["aware"]
    horizontal, vertical = pair_series(focal_lat, focal_lon, focal_alt,
                                       neighbour_lat, neighbour_lon, neighbour_alt)
    joint = joint_separation_nm(horizontal, vertical)
    initial_joint = float(joint[0]) if focal_model["available"] else np.nan
    rows = []
    for item in original.to_dict("records"):
        horizon = int(item["horizon_s"])
        n = horizon // STEP + 1
        if focal_model["available"]:
            a_min_h = float(np.min(horizontal[:n]))
            a_min_v = float(np.min(vertical[:n]))
            a_min_joint = float(np.min(joint[:n]))
            a_tcpa = float(grid[np.argmin(horizontal[:n])])
            a_cpa = a_min_h
            # Signed pre-event horizontal TCPA is estimated from the first
            # local velocity segment; negative means diverging.
            dx, dy = xy_from_latlon(neighbour_lat[0], neighbour_lon[0],
                                    focal_lat[0], focal_lon[0])
            fx, fy = xy_from_latlon(focal_lat[1], focal_lon[1], focal_lat[0], focal_lon[0])
            ox, oy = xy_from_latlon(neighbour_lat[1], neighbour_lon[1],
                                    neighbour_lat[0], neighbour_lon[0])
            rvx, rvy = (ox - fx) / STEP, (oy - fy) / STEP
            vv = rvx * rvx + rvy * rvy
            raw_tcp = -float(dx * rvx + dy * rvy) / vv if vv > 1e-14 else np.nan
            closing_kt = (-float(dx * rvx + dy * rvy) /
                          max(math.hypot(dx, dy), 1e-9) * 3600)
            vertical_closing_fpm = float((vertical[0] - vertical[n - 1]) * 60 / horizon)
            endpoint_x, endpoint_y = xy_from_latlon(neighbour_lat[n - 1], neighbour_lon[n - 1],
                                                     focal_lat[n - 1], focal_lon[n - 1])
            end_bearing = float(np.rad2deg(np.arctan2(endpoint_x, endpoint_y)) % 360)
        else:
            a_min_h = a_min_v = a_min_joint = a_tcpa = a_cpa = raw_tcp = np.nan
            closing_kt = vertical_closing_fpm = end_bearing = np.nan
        qj = qvalue(thresholds, horizon, "pair_min_joint_error_nm", .95)
        qt = qvalue(thresholds, horizon, "pair_tcpa_error_s", .95)
        qv = qvalue(thresholds, horizon, "altitude_error_ft", .95)
        q_pair_v = qvalue(thresholds, horizon, "pair_min_vertical_error_ft", .95)
        q_pair_h = qvalue(thresholds, horizon, "pair_min_horizontal_error_nm", .95)
        observed_joint = optional_number(item["observed_min_joint_sep_nm"])
        observed_tcp = optional_number(item["observed_TCPA"])
        delta_joint = observed_joint - a_min_joint if np.isfinite(observed_joint) else np.nan
        delta_tcp = observed_tcp - a_tcpa if np.isfinite(observed_tcp) else np.nan
        joint_material = bool(np.isfinite(delta_joint) and delta_joint > qj)
        delta_h = optional_number(item["observed_min_horizontal_sep"]) - a_min_h
        delta_v = optional_number(item["observed_min_vertical_sep"]) - a_min_v
        mechanism_material = (
            bool(np.isfinite(delta_v) and delta_v > q_pair_v) if history["clearance_type"] == "LEVEL" else
            bool(np.isfinite(delta_h) and delta_h > q_pair_h) if history["clearance_type"] == "HEADING" else
            True)
        timing_material = bool(history["clearance_type"] == "SPEED" and
                               np.isfinite(delta_tcp) and delta_tcp > qt and closing_kt > 0)
        convergence = bool(np.isfinite(raw_tcp) and 0 < raw_tcp <= horizon and
                           np.isfinite(initial_joint) and initial_joint - a_min_joint > qj)
        prior_v1_joint = optional_number(item["delta_joint_sep_nm"])
        strong_disagreement = bool(np.isfinite(prior_v1_joint) and
                                   not scenarios_not_opposed(prior_v1_joint, qj))
        b_material = bool(np.isfinite(prior_v1_joint) and prior_v1_joint > qj)
        response = response_lookup.get((history["window_id"], horizon), "UNKNOWN")
        response_v2 = response
        if history["clearance_type"] == "LEVEL" and response in {"YES", "PARTIAL"}:
            observed_alt = optional_number(observed_focal_altitude.get(horizon))
            new_target = history["current_level_ft"]
            if (not np.isfinite(observed_alt) or not np.isfinite(new_target) or
                    not np.isfinite(focal_alt[n - 1])):
                response_v2 = "UNKNOWN"
            elif (abs(focal_alt[n - 1] - new_target) - abs(observed_alt - new_target)) <= qv:
                response_v2 = "UNKNOWN"
        usable = bool(item["usable"] and focal_model["available"] and
                      np.isfinite(delta_joint) and np.isfinite(delta_tcp))
        aware_evidence = bool(usable and convergence and mechanism_material and
                              (joint_material or timing_material) and
                              response_v2 in {"YES", "PARTIAL"} and not strong_disagreement)
        rows.append({"window_id": history["window_id"], "matched_pair_id": int(history["matched_pair_id"]),
                     "clearance_type": history["clearance_type"], "partner_id": item["partner_id"],
                     "horizon_s": horizon, "candidate_rank": item["candidate_rank"],
                     "neighbourhood_nm": item["neighbourhood_nm"],
                     "v1_pair_usable": bool(item["usable"]), "usable_v2": usable,
                     "FOCAL_NEXT_CLEARANCE": bool(item["FOCAL_NEXT_CLEARANCE"]),
                     "NEIGHBOUR_NEXT_CLEARANCE": bool(item["NEIGHBOUR_NEXT_CLEARANCE"]),
                     "observed_pair_coverage": item["observed_pair_coverage"],
                     "previous_intent_available": bool(focal_model["available"]),
                     "counterfactual_min_horizontal_sep": a_min_h,
                     "counterfactual_min_vertical_sep": a_min_v,
                     "counterfactual_joint_sep": a_min_joint,
                     "counterfactual_CPA": a_cpa, "counterfactual_TCPA": a_tcpa,
                     "raw_counterfactual_TCPA": raw_tcp,
                     "horizontal_closing_kt": closing_kt,
                     "vertical_closing_fpm": vertical_closing_fpm,
                     "bearing_endpoint_deg": end_bearing,
                     "crossing_angle_deg": item["pre_event_crossing_angle_deg"],
                     "observed_min_horizontal_sep": item["observed_min_horizontal_sep"],
                     "observed_min_vertical_sep": item["observed_min_vertical_sep"],
                     "observed_min_joint_sep_nm": observed_joint,
                     "observed_CPA": item["observed_CPA"], "observed_TCPA": observed_tcp,
                     "delta_horizontal_sep": item["observed_min_horizontal_sep"] - a_min_h,
                     "delta_vertical_sep": item["observed_min_vertical_sep"] - a_min_v,
                     "delta_joint_sep_nm": delta_joint, "delta_CPA": item["observed_CPA"] - a_cpa,
                     "delta_TCPA": delta_tcp,
                     "v1_kinematic_min_joint_sep_nm": item["counterfactual_min_joint_sep_nm"],
                     "v1_kinematic_delta_joint_sep_nm": prior_v1_joint,
                     "threshold_joint_p95_nm": qj, "threshold_tcpa_p95_s": qt,
                     "clearance_response_confirmed_v1": response,
                     "clearance_response_confirmed_v2": response_v2,
                     "pre_event_convergence_v2": convergence,
                     "material_joint_v2": joint_material,
                     "material_mechanism_v2": mechanism_material,
                     "material_timing_v2": timing_material,
                     "v1_kinematic_material_joint": b_material,
                     "strong_scenario_disagreement": strong_disagreement,
                     "scenario_agreement": bool(not strong_disagreement),
                     "clearance_aware_evidence": aware_evidence})
    return rows


def process_days(history: pd.DataFrame, pair_v1: pd.DataFrame,
                 day_events: pd.DataFrame, thresholds: pd.DataFrame,
                 responses: pd.DataFrame):
    response_lookup = {(r.window_id, int(r.horizon_s)): r.clearance_response_confirmed
                       for r in responses.itertuples(index=False)}
    history_by_id = history.set_index("window_id").to_dict("index")
    pair_groups = {key: group for key, group in pair_v1.loc[pair_v1.ATC_INTERVENTION == 1].groupby("window_id", sort=False)}
    folder = OUT / "day_checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        tag = day.strftime("%Y%m%d")
        marker = folder / f"day_{tag}_complete.json"
        model_file = folder / f"day_{tag}_models.parquet"
        pair_file = folder / f"day_{tag}_pair_horizon.parquet"
        if marker.exists():
            check = json.loads(marker.read_text())
            if check.get("version") == VERSION and model_file.exists() and pair_file.exists():
                print(f"Stage 09 V2: reused day {tag}, {check['events']} anchors", flush=True)
                continue
        subset = day_events.loc[(day_events.event_epoch >= day.timestamp()) &
                                (day_events.event_epoch < (day + pd.Timedelta(days=1)).timestamp())]
        index = V1.NativeStateIndex(V1.load_day_states(day))
        model_rows, pair_rows = [], []
        for event in subset.to_dict("records"):
            anchor = float(event["event_epoch"])
            focal = index.last_before(event["flight_key"], anchor)
            pre = pre_event_metrics(index, event["flight_key"], anchor)
            if focal is not None and V1.valid_kinematic_state(focal):
                robust, robust_ok, _ = index.robust_state(focal, anchor)
                regime = dynamic_regime(pre, robust["vertical_rate_fpm"])
            else:
                robust, robust_ok, regime = None, False, "UNCLASSIFIED"
            model = {"window_id": event["window_id"], "matched_pair_id": int(event["matched_pair_id"]),
                     "ATC_INTERVENTION": int(event["ATC_INTERVENTION"]), "split": event["split"],
                     "event_epoch": anchor, "flight_key": event["flight_key"],
                     "anchor_altitude_ft": event.get("anchor_altitude_ft", np.nan),
                     "anchor_vertical_rate_fpm": event.get("anchor_vertical_rate_fpm", np.nan),
                     "robust_pre_state_available": bool(robust_ok),
                     "dynamic_regime": regime,
                     "heading_stable": bool(pre["pre_samples"] >= 3 and
                                            np.isfinite(pre["pre_track_spread_deg"]) and
                                            pre["pre_track_spread_deg"] <= 3),
                     "speed_stable": bool(pre["pre_samples"] >= 3 and
                                          np.isfinite(pre["pre_speed_range_kt"]) and
                                          pre["pre_speed_range_kt"] <= 10), **pre}
            if int(event["ATC_INTERVENTION"]) == 1:
                hist = {**history_by_id[event["window_id"]], "window_id": event["window_id"]}
                if robust_ok:
                    focal_model = focal_clearance_aware_state(hist, focal, robust, pre)
                    model["clearance_aware_available"] = focal_model["available"]
                    model["intent_state"] = focal_model["intent_state"]
                    model["intent_quality"] = focal_model["quality"]
                else:
                    focal_model = None
                    model["clearance_aware_available"] = False
                    model["intent_state"] = "PREVIOUS_INTENT_UNKNOWN"
                    model["intent_quality"] = "MISSING_FRESH_ROBUST_FOCAL_STATE"
                pair_event = pair_groups.get(event["window_id"])
                if pair_event is not None and focal_model is not None:
                    for partner, original in pair_event.groupby("partner_id", sort=False):
                        neighbour = index.last_before(partner, anchor)
                        if neighbour is None or not V1.valid_kinematic_state(neighbour):
                            continue
                        other_robust, other_ok, _ = index.robust_state(neighbour, anchor)
                        if not other_ok:
                            continue
                        obs_alt = {h: event.get(f"observed_focal_altitude_{h}s", np.nan) for h in HORIZONS}
                        pair_rows.extend(pair_counterfactual_rows(hist, focal_model, neighbour,
                                                                  other_robust, original, thresholds,
                                                                  response_lookup, obs_alt))
            model_rows.append(model)
        models = pd.DataFrame(model_rows)
        pairs = pd.DataFrame(pair_rows)
        models.to_parquet(model_file, index=False, compression="zstd")
        pairs.to_parquet(pair_file, index=False, compression="zstd")
        marker.write_text(json.dumps({"version": VERSION, "events": len(models),
                                      "interventions": int(models.ATC_INTERVENTION.sum()),
                                      "pair_horizon_rows": len(pairs)}, indent=2))
        print(f"Stage 09 V2: completed day {tag}: {len(models)} anchors, "
              f"{int(models.ATC_INTERVENTION.sum())} interventions, {len(pairs)} pair/horizon rows", flush=True)
        del index
        gc.collect()
    models = pd.concat([pd.read_parquet(folder / f"day_{d.strftime('%Y%m%d')}_models.parquet")
                        for d in DAYS], ignore_index=True)
    pairs = pd.concat([pd.read_parquet(folder / f"day_{d.strftime('%Y%m%d')}_pair_horizon.parquet")
                       for d in DAYS], ignore_index=True)
    assert len(models) == 8196 and models.window_id.is_unique
    assert models.ATC_INTERVENTION.sum() == 4098
    return models, pairs


def calibrate_dynamic_regimes(calibration: pd.DataFrame, models: pd.DataFrame,
                              thresholds: pd.DataFrame):
    controls = models.loc[models.ATC_INTERVENTION == 0,
                          ["window_id", "dynamic_regime", "heading_stable", "speed_stable"]]
    frame = calibration.merge(controls, on="window_id", validate="many_to_one")
    assert frame.split.isin(["train", "validation", "test", "excluded_cut_or_embargo"]).all()
    regime_masks = {"LEVEL_LIKE": frame.dynamic_regime.eq("LEVEL_LIKE"),
                    "CLIMB_DESCENT_LIKE": frame.dynamic_regime.eq("CLIMB_DESCENT_LIKE"),
                    "HEADING_STABLE": frame.heading_stable.eq(True),
                    "SPEED_STABLE": frame.speed_stable.eq(True)}
    variables = ["pair_min_horizontal_error_nm", "pair_min_vertical_error_ft",
                 "pair_min_joint_error_nm", "pair_tcpa_error_s",
                 "horizontal_position_error_nm", "altitude_error_ft"]
    rows = []
    for regime, mask in regime_masks.items():
        for horizon in HORIZONS:
            for variable in variables:
                train = frame.loc[mask & frame.split.eq("train") & frame.horizon_s.eq(horizon)]
                valid = frame.loc[mask & frame.split.eq("validation") & frame.horizon_s.eq(horizon)]
                a = train[variable].dropna()
                b = valid[variable].dropna()
                global_p95 = qvalue(thresholds, horizon, variable, .95)
                train_p95 = float(a.quantile(.95)) if len(a) else np.nan
                valid_p95 = float(b.quantile(.95)) if len(b) else np.nan
                sufficient = (train.window_id.nunique() >= 100 and
                              valid.window_id.nunique() >= 30)
                stable = bool(sufficient and train_p95 > 0 and
                              (b > train_p95).mean() <= .15 and
                              valid_p95 <= 1.5 * train_p95)
                rows.append({"dynamic_regime": regime, "horizon_s": horizon,
                             "variable": variable, "train_pair_rows": len(a),
                             "train_control_windows": train.window_id.nunique(),
                             "validation_pair_rows": len(b),
                             "validation_control_windows": valid.window_id.nunique(),
                             "train_p90_error": float(a.quantile(.90)) if len(a) else np.nan,
                             "train_p95_error": train_p95,
                             "validation_p95_error": valid_p95,
                             "validation_fraction_exceeding_train_p95": float((b > train_p95).mean()) if len(b) else np.nan,
                             "global_train_p95_error": global_p95,
                             "regime_to_global_p95_ratio": train_p95 / global_p95 if global_p95 > 0 else np.nan,
                             "calibration_stable": stable})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "control_calibration_by_dynamic_regime.csv", index=False)
    return result


def complete_v2_pair_table(pair_v1: pd.DataFrame, pair_v2: pd.DataFrame,
                           evidence_v1: pd.DataFrame):
    keys = ["window_id", "partner_id", "horizon_s"]
    base = pair_v1.loc[pair_v1.ATC_INTERVENTION == 1].copy()
    assert not base.duplicated(keys).any() and not pair_v2.duplicated(keys).any()
    selected = base[keys + ["matched_pair_id", "candidate_rank", "neighbourhood_nm",
                            "usable", "FOCAL_NEXT_CLEARANCE", "NEIGHBOUR_NEXT_CLEARANCE",
                            "observed_pair_coverage", "current_horizontal_sep_nm",
                            "current_vertical_sep_ft", "pre_event_bearing_deg",
                            "pre_event_crossing_angle_deg"]].rename(columns={
                                "usable": "v1_pair_usable", "pre_event_bearing_deg": "bearing_anchor_deg",
                                "pre_event_crossing_angle_deg": "crossing_angle_anchor_deg"})
    result = selected.merge(pair_v2.drop(columns=["matched_pair_id", "candidate_rank",
                                                  "neighbourhood_nm", "v1_pair_usable",
                                                  "FOCAL_NEXT_CLEARANCE", "NEIGHBOUR_NEXT_CLEARANCE",
                                                  "observed_pair_coverage"]),
                            on=keys, how="left", validate="one_to_one")
    evidence = evidence_v1[keys + ["interaction_evidence", "pre_event_convergence",
                                   "material_improvement", "counterfactual_model_agreement"]].rename(columns={
                                       "interaction_evidence": "kinematic_evidence_v1",
                                       "pre_event_convergence": "kinematic_pre_event_convergence_v1",
                                       "material_improvement": "kinematic_material_improvement_v1",
                                       "counterfactual_model_agreement": "kinematic_scenario_agreement_v1"})
    result = result.merge(evidence, on=keys, how="left", validate="one_to_one")
    for column in ("usable_v2", "previous_intent_available", "clearance_aware_evidence",
                   "pre_event_convergence_v2", "material_joint_v2", "material_mechanism_v2",
                   "material_timing_v2", "strong_scenario_disagreement", "kinematic_evidence_v1",
                   "kinematic_pre_event_convergence_v1", "kinematic_material_improvement_v1"):
        result[column] = result[column].fillna(False).astype(bool)
    assert len(result) == len(base) and not result.duplicated(keys).any()
    assert result.horizon_s.isin(HORIZONS).all()
    return result


def variant_flags(frame: pd.DataFrame, thresholds: pd.DataFrame,
                  percentile: float, baseline: str):
    if baseline == "kinematic_v1":
        return frame.kinematic_evidence_v1.to_numpy(dtype=bool), frame.kinematic_material_improvement_v1.to_numpy(dtype=bool), frame.kinematic_pre_event_convergence_v1.to_numpy(dtype=bool)
    qj = frame.horizon_s.map({h: qvalue(thresholds, h, "pair_min_joint_error_nm", percentile) for h in HORIZONS})
    qt = frame.horizon_s.map({h: qvalue(thresholds, h, "pair_tcpa_error_s", percentile) for h in HORIZONS})
    qv = frame.horizon_s.map({h: qvalue(thresholds, h, "pair_min_vertical_error_ft", percentile) for h in HORIZONS})
    qh = frame.horizon_s.map({h: qvalue(thresholds, h, "pair_min_horizontal_error_nm", percentile) for h in HORIZONS})
    a_joint = frame.counterfactual_joint_sep
    current_joint = joint_separation_nm(frame.current_horizontal_sep_nm,
                                        frame.current_vertical_sep_ft)
    converge = (frame.raw_counterfactual_TCPA.gt(0) &
                frame.raw_counterfactual_TCPA.le(frame.horizon_s) &
                pd.Series(current_joint, index=frame.index).sub(a_joint).gt(qj))
    joint = frame.delta_joint_sep_nm.gt(qj)
    timing = (frame.clearance_type.eq("SPEED") & frame.delta_TCPA.gt(qt) &
              frame.horizontal_closing_kt.gt(0))
    mechanism = ((frame.clearance_type.eq("LEVEL") & frame.delta_vertical_sep.gt(qv)) |
                 (frame.clearance_type.eq("HEADING") & frame.delta_horizontal_sep.gt(qh)) |
                 frame.clearance_type.eq("SPEED"))
    not_opposed = frame.v1_kinematic_delta_joint_sep_nm.ge(-qj)
    response = frame.clearance_response_confirmed_v2.isin(["YES", "PARTIAL"])
    material = joint | timing
    signal = (frame.usable_v2 & converge & material & mechanism & not_opposed & response)
    return signal.to_numpy(dtype=bool), material.to_numpy(dtype=bool), converge.to_numpy(dtype=bool)


def classify_variant(v1: pd.DataFrame, history: pd.DataFrame, models: pd.DataFrame,
                     candidates: pd.DataFrame, pairs: pd.DataFrame,
                     thresholds: pd.DataFrame, percentile: float = .95,
                     radius_nm: int = 30, horizons: tuple[int, ...] = MAIN,
                     baseline: str = "clearance_aware"):
    assert radius_nm in (30, 50) and set(horizons) <= set(HORIZONS)
    working = pairs.loc[pairs.neighbourhood_nm.le(radius_nm) & pairs.horizon_s.isin(horizons)].copy()
    signal, material, convergence = variant_flags(working, thresholds, percentile, baseline)
    working["variant_signal"] = signal
    working["variant_material"] = material
    working["variant_convergence"] = convergence
    grouped = {key: group for key, group in working.groupby("window_id", sort=False)}
    model_by_id = models.loc[models.ATC_INTERVENTION == 1].set_index("window_id")
    history_by_id = history.set_index("window_id")
    rows = []
    for event in v1.to_dict("records"):
        key = event["window_id"]
        model = model_by_id.loc[key]
        hist = history_by_id.loc[key]
        part = grouped.get(key, working.iloc[0:0])
        total = int(event[f"candidate_partner_count_{radius_nm}nm"])
        evaluated = part.partner_id.nunique()
        complete = bool(total > 0 and total == evaluated)
        positive = part.loc[part.variant_signal]
        support = list(zip(positive.horizon_s.astype(int), positive.partner_id))
        same_partner = has_same_partner_across_horizons(support)
        partner_ambiguous = len(set(positive.partner_id)) > 1
        negative_h = []
        positive_h = []
        usable_h = []
        for h in horizons:
            sample = part.loc[part.horizon_s == h]
            useful = sample.loc[sample.usable_v2 if baseline != "kinematic_v1" else sample.v1_pair_usable]
            usable_h.append(bool(len(useful)))
            positive_h.append(bool(sample.variant_signal.any()))
            negative_h.append(bool(complete and len(sample) == total and len(useful) == total and
                                   sample.clearance_response_confirmed_v2.eq("YES").all() and
                                   not sample.variant_convergence.any() and
                                   not sample.variant_material.any() and
                                   not sample.strong_scenario_disagreement.any()))
        reasons = []
        if not bool(model.get("clearance_aware_available", False)):
            reasons.append(str(model.get("intent_quality", "PREVIOUS_INTENT_UNKNOWN")))
        if total == 0:
            reasons.append("NO_QUALIFIED_NEIGHBOUR")
        elif not complete:
            reasons.append("UNTESTED_CANDIDATES")
        if not all(usable_h):
            reasons.append("INSUFFICIENT_60_120_COVERAGE" if tuple(horizons) == MAIN else "INSUFFICIENT_HORIZON_COVERAGE")
        if part.FOCAL_NEXT_CLEARANCE.any() or part.NEIGHBOUR_NEXT_CLEARANCE.any():
            reasons.append("SUBSEQUENT_CLEARANCE_CENSORED_HORIZON")
        if partner_ambiguous:
            reasons.append("PARTNER_AMBIGUITY")
        if len(horizons) >= 2 and any(positive_h) and not same_partner:
            reasons.append("PARTNER_SWITCH_OR_SINGLE_HORIZON")
        if part.strong_scenario_disagreement.any():
            reasons.append("COUNTERFACTUAL_SCENARIOS_DISAGREE")
        if not any(positive_h) and not all(negative_h):
            reasons.append("MATERIALITY_OR_CONVERGENCE_INSUFFICIENT")
        if baseline != "kinematic_v1" and not bool(model.get("clearance_aware_available", False)):
            label = "UNKNOWN"
        elif (all(positive_h) and (same_partner if len(horizons) >= 2 else not partner_ambiguous) and
              not partner_ambiguous and not part.loc[part.variant_signal, "strong_scenario_disagreement"].any()):
            label = "LIKELY_INTERACTION_RELATED"
        elif all(negative_h) and complete and not part.strong_scenario_disagreement.any():
            label = "LIKELY_NON_INTERACTION_OPERATIONAL"
        else:
            label = "UNKNOWN"
        preferred = (positive.sort_values(["candidate_rank", "horizon_s"]).iloc[0]
                     if len(positive) else part.sort_values(["candidate_rank", "horizon_s"]).iloc[0]
                     if len(part) else None)
        response = (preferred.clearance_response_confirmed_v2 if preferred is not None else "UNKNOWN")
        if response not in {"YES", "PARTIAL"} and label == "UNKNOWN":
            reasons.append("CLEARANCE_RESPONSE_UNCONFIRMED")
        available = bool(model.get("clearance_aware_available", False))
        scenario_quality = (float(not part.strong_scenario_disagreement.any()) if len(part) else 0.0)
        coverage_quality = sum(usable_h) / max(len(horizons), 1)
        response_quality = {"YES": 1.0, "PARTIAL": .5}.get(response, 0.0)
        horizon_quality = sum(positive_h if label == "LIKELY_INTERACTION_RELATED" else negative_h) / max(len(horizons), 1)
        ratio = (float(preferred.delta_joint_sep_nm) / max(float(preferred.threshold_joint_p95_nm), 1e-9)
                 if preferred is not None and np.isfinite(preferred.delta_joint_sep_nm) else 0.0)
        magnitude = min(1.0, max(0.0, ratio) / 2) if label == "LIKELY_INTERACTION_RELATED" else 0.0
        confidence = (.2 * coverage_quality + .2 * float(available) + .2 * response_quality +
                      .2 * scenario_quality + .1 * horizon_quality + .1 * magnitude)
        if label == "UNKNOWN":
            confidence = min(.49, confidence)
        rows.append({"window_id": key, "matched_pair_id": int(event["matched_pair_id"]),
                     "stage09_label": label,
                     "evidence_confidence_score_v2": round(confidence, 6),
                     "unknown_reason": ";".join(sorted(set(reasons))) if label == "UNKNOWN" else "",
                     "candidate_partner_id": preferred.partner_id if preferred is not None else "",
                     "candidate_partner_count": total,
                     "primary_horizon": int(preferred.horizon_s) if preferred is not None else np.nan,
                     "same_partner_60_120": same_partner,
                     "partner_switch": bool(any(positive_h) and not same_partner),
                     "scenario_agreement": bool(len(part) and not part.strong_scenario_disagreement.any()),
                     "clearance_response_confirmed": response,
                     "routine_level_change_compatible": bool(event["clearance_type"] == "LEVEL" and
                                                             label == "LIKELY_NON_INTERACTION_OPERATIONAL"),
                     **{f"usable_{h}s": bool(usable_h[i]) for i, h in enumerate(horizons)},
                     **{f"label_{h}s": ("LIKELY_INTERACTION_RELATED" if positive_h[i] else
                                        "LIKELY_NON_INTERACTION_OPERATIONAL" if negative_h[i] else "UNKNOWN")
                        for i, h in enumerate(horizons)}})
    result = pd.DataFrame(rows)
    assert len(result) == 4098 and result.window_id.is_unique and set(result.stage09_label) <= set(ALLOWED)
    assert result.evidence_confidence_score_v2.between(0, 1).all()
    return result


def write_result_tables(v1: pd.DataFrame, history: pd.DataFrame, models: pd.DataFrame,
                        candidates: pd.DataFrame, pairs: pd.DataFrame,
                        summary: pd.DataFrame):
    comparison = v1[["window_id", "clearance_type", "stage09_label",
                     "evidence_confidence_score"]].rename(columns={
                         "stage09_label": "V1_label", "evidence_confidence_score": "V1_confidence"})
    comparison = comparison.merge(summary[["window_id", "stage09_label", "evidence_confidence_score_v2",
                                           "unknown_reason", "scenario_agreement"]],
                                  on="window_id", validate="one_to_one").rename(columns={
                                      "stage09_label": "V2_label",
                                      "evidence_confidence_score_v2": "V2_confidence"})
    comparison = comparison.merge(history[["window_id", "previous_instruction_fresh"]],
                                  on="window_id", validate="one_to_one")
    comparison = comparison.merge(models.loc[models.ATC_INTERVENTION == 1,
                                             ["window_id", "clearance_aware_available", "intent_quality"]],
                                  on="window_id", validate="one_to_one")
    comparison["changed_label"] = comparison.V1_label != comparison.V2_label
    comparison["reason_for_change"] = np.where(
        ~comparison.changed_label, "UNCHANGED",
        np.where(comparison.V2_label.eq("UNKNOWN"), comparison.unknown_reason,
                 np.where(comparison.V2_label.eq("LIKELY_NON_INTERACTION_OPERATIONAL"),
                          "PREVIOUS_INTENT_EVALUABLE_NO_MATERIAL_PAIR_EVIDENCE",
                          "CLEARANCE_AWARE_MULTI_HORIZON_PAIR_EVIDENCE")))
    comparison = comparison.rename(columns={"previous_instruction_fresh": "previous_instruction_available",
                                            "clearance_aware_available": "clearance_aware_counterfactual_available"})
    comparison.to_csv(OUT / "stage09_v1_v2_comparison.csv", index=False)
    matrix = comparison.groupby(["V1_label", "V2_label"]).size().rename("n_events").reset_index()
    matrix.to_csv(OUT / "label_transition_matrix.csv", index=False)
    counts = summary.stage09_label.value_counts()
    label_counts = pd.DataFrame({"stage09_label": list(ALLOWED),
                                 "n_events": [int(counts.get(label, 0)) for label in ALLOWED]})
    label_counts["percent"] = 100 * label_counts.n_events / len(summary)
    label_counts.to_csv(OUT / "counterfactual_label_counts_v2.csv", index=False)
    reasons = summary.loc[summary.stage09_label.eq("UNKNOWN"), "unknown_reason"].str.split(";").explode().value_counts()
    unknown = pd.DataFrame({"unknown_reason": reasons.index, "n_events": reasons.to_numpy()})
    unknown.to_csv(OUT / "unknown_reason_counts_v2.csv", index=False)
    records = []
    if {"clearance_type", "previous_level_ft", "intent_state"} <= set(summary.columns):
        joined = (summary.copy() if "candidate_partner_count_30nm" in summary else
                  summary.merge(v1[["window_id", "candidate_partner_count_30nm"]],
                                on="window_id", validate="one_to_one"))
    else:
        joined = (summary.merge(v1[["window_id", "clearance_type", "candidate_partner_count_30nm"]],
                                on="window_id", validate="one_to_one")
                  .merge(history[["window_id", "previous_level_ft", "previous_heading",
                                  "previous_speed", "delta_speed_comparable", "previous_instruction_fresh"]],
                         on="window_id", validate="one_to_one")
                  .merge(models.loc[models.ATC_INTERVENTION == 1,
                                    ["window_id", "clearance_aware_available", "intent_state"]],
                         on="window_id", validate="one_to_one"))
    for kind, part in joined.groupby("clearance_type"):
        record = {"clearance_type": kind, "n_events": len(part),
                  "mean_evidence_confidence_score_v2": part.evidence_confidence_score_v2.mean(),
                  "median_candidates_30nm": part.candidate_partner_count_30nm.median(),
                  "usable_60s_percent": 100 * part.usable_60s.mean(),
                  "usable_120s_percent": 100 * part.usable_120s.mean(),
                  "clearance_aware_available_percent": 100 * part.clearance_aware_available.mean(),
                  "previous_instruction_fresh_percent": 100 * part.previous_instruction_fresh.mean(),
                  "previous_CFL_available_percent": 100 * part.previous_level_ft.notna().mean() if kind == "LEVEL" else np.nan,
                  "previous_intent_classified_percent": 100 * part.intent_state.isin([
                      "LEVEL_AT_PREVIOUS_CFL", "CLIMBING_TOWARD_PREVIOUS_CFL",
                      "DESCENDING_TOWARD_PREVIOUS_CFL"]).mean() if kind == "LEVEL" else np.nan,
                  "previous_heading_available_percent": 100 * part.previous_heading.notna().mean() if kind == "HEADING" else np.nan,
                  "previous_speed_available_percent": 100 * part.previous_speed.notna().mean() if kind == "SPEED" else np.nan,
                  "comparable_speed_units_percent": 100 * part.delta_speed_comparable.notna().mean() if kind == "SPEED" else np.nan}
        for label in ALLOWED:
            n = int(part.stage09_label.eq(label).sum())
            record[label + "_n"] = n
            record[label + "_percent"] = 100 * n / len(part)
        records.append(record)
    by_type = pd.DataFrame(records)
    by_type.to_csv(OUT / "labels_by_clearance_type_v2.csv", index=False)
    level = (history.loc[history.clearance_type == "LEVEL"].merge(
        models.loc[models.ATC_INTERVENTION == 1,
                   ["window_id", "anchor_altitude_ft", "altitude_trend_fpm", "dynamic_regime",
                    "intent_state", "intent_quality", "clearance_aware_available"]],
        on="window_id", validate="one_to_one")
        .merge(summary[["window_id", "stage09_label", "evidence_confidence_score_v2",
                        "unknown_reason", "usable_60s", "usable_120s",
                        "candidate_partner_count", "routine_level_change_compatible"]],
               on="window_id", validate="one_to_one"))
    level_pair = pairs.loc[pairs.clearance_type.eq("LEVEL") & pairs.neighbourhood_nm.le(30)]
    level_signal = level_pair.groupby("window_id").agg(
        any_material_interaction_evidence=("clearance_aware_evidence", "any"),
        any_material_joint=("material_joint_v2", "any"),
        any_mechanism_material=("material_mechanism_v2", "any")).reset_index()
    level = level.merge(level_signal, on="window_id", how="left", validate="one_to_one")
    level_50_signal = pairs.loc[pairs.clearance_type.eq("LEVEL")].groupby("window_id")[
        "clearance_aware_evidence"].any().rename("any_material_interaction_evidence_50nm")
    level = level.merge(level_50_signal, on="window_id", how="left", validate="one_to_one")
    for field in ("any_material_interaction_evidence", "any_material_joint", "any_mechanism_material"):
        level[field] = level[field].fillna(False).astype(bool)
    level["any_material_interaction_evidence_50nm"] = level[
        "any_material_interaction_evidence_50nm"].fillna(False).astype(bool)
    level.to_parquet(OUT / "level_counterfactual_details.parquet", index=False, compression="zstd")
    level_counts = {"n_level_events": len(level),
                    "previous_CFL_valid": int(level.previous_level_ft.notna().sum()),
                    "previous_instruction_fresh": int(level.previous_instruction_fresh.sum()),
                    "level_at_previous_CFL": int(level.intent_state.eq("LEVEL_AT_PREVIOUS_CFL").sum()),
                    "climbing_toward_previous_CFL": int(level.intent_state.eq("CLIMBING_TOWARD_PREVIOUS_CFL").sum()),
                    "descending_toward_previous_CFL": int(level.intent_state.eq("DESCENDING_TOWARD_PREVIOUS_CFL").sum()),
                    "previous_level_already_reached_but_unstable": int(level.intent_state.eq("PREVIOUS_LEVEL_ALREADY_REACHED").sum()),
                    "new_CFL_distinct": int(level.delta_CFL_ft.abs().ge(1).sum()),
                    "candidate_within_30nm": int(level.candidate_partner_count.gt(0).sum()),
                    "clearance_aware_evaluable": int(level.clearance_aware_available.sum()),
                    "usable_60s": int(level.usable_60s.sum()),
                    "usable_120s": int(level.usable_120s.sum()),
                    "any_material_interaction_evidence": int(level.any_material_interaction_evidence.sum()),
                    "LIKELY_INTERACTION_RELATED": int(level.stage09_label.eq("LIKELY_INTERACTION_RELATED").sum()),
                    "LIKELY_NON_INTERACTION_OPERATIONAL": int(level.stage09_label.eq("LIKELY_NON_INTERACTION_OPERATIONAL").sum()),
                    "UNKNOWN": int(level.stage09_label.eq("UNKNOWN").sum())}
    level_summary = pd.DataFrame({"metric": list(level_counts), "n_events": list(level_counts.values())})
    level_summary.to_csv(OUT / "level_counterfactual_summary.csv", index=False)
    delta = level.assign(delta_CFL_direction=np.select(
        [level.delta_CFL_ft.lt(-1), level.delta_CFL_ft.gt(1), level.delta_CFL_ft.abs().le(1)],
        ["DESCENT_CLEARANCE", "CLIMB_CLEARANCE", "SAME_LEVEL"], default="UNKNOWN"))
    delta.groupby(["delta_CFL_direction", "stage09_label"], dropna=False).agg(
        n_events=("window_id", "size"), median_delta_CFL_ft=("delta_CFL_ft", "median"),
        p25_delta_CFL_ft=("delta_CFL_ft", lambda x: x.quantile(.25)),
        p75_delta_CFL_ft=("delta_CFL_ft", lambda x: x.quantile(.75))).reset_index().to_csv(
            OUT / "level_delta_CFL_distribution.csv", index=False)
    for kind, filename in (("HEADING", "heading_counterfactual_summary.csv"),
                           ("SPEED", "speed_counterfactual_summary.csv")):
        part = joined.loc[joined.clearance_type == kind]
        fields = {"n_events": len(part),
                  "previous_instruction_fresh": int(part.previous_instruction_fresh.sum()),
                  "previous_heading_available": int(part.previous_heading.notna().sum()) if kind == "HEADING" else np.nan,
                  "previous_speed_available": int(part.previous_speed.notna().sum()) if kind == "SPEED" else np.nan,
                  "comparable_speed_units": int(part.delta_speed_comparable.notna().sum()) if kind == "SPEED" else np.nan,
                  "clearance_aware_evaluable": int(part.clearance_aware_available.sum()),
                  "usable_60s": int(part.usable_60s.sum()), "usable_120s": int(part.usable_120s.sum()),
                  **{label: int(part.stage09_label.eq(label).sum()) for label in ALLOWED}}
        pd.DataFrame({"metric": list(fields), "n_events": list(fields.values())}).to_csv(OUT / filename, index=False)
    return comparison, matrix, label_counts, by_type, unknown, level, level_summary


def sensitivity_analysis(v1: pd.DataFrame, history: pd.DataFrame, models: pd.DataFrame,
                         candidates: pd.DataFrame, pairs: pd.DataFrame,
                         thresholds: pd.DataFrame, primary: pd.DataFrame):
    configurations = [
        ("50nm_neighbourhood", 50, .95, MAIN, "clearance_aware"),
        ("90th_percentile_error", 30, .90, MAIN, "clearance_aware"),
        ("60s_only", 30, .95, (60,), "clearance_aware"),
        ("120s_only", 30, .95, (120,), "clearance_aware"),
        ("300s_quality_censored_sensitivity", 30, .95, (300,), "clearance_aware"),
        ("600s_quality_censored_audit", 30, .95, (600,), "clearance_aware"),
        ("60_120_consistency", 30, .95, MAIN, "clearance_aware"),
        ("V1_robust_kinematic_60_120", 30, .95, MAIN, "kinematic_v1"),
    ]
    baseline = primary.set_index("window_id").stage09_label
    rows, labels = [], []
    for name, radius, percentile, horizons, model in configurations:
        variant = (primary if name == "60_120_consistency" else
                   classify_variant(v1, history, models, candidates, pairs,
                                    thresholds, percentile, radius, horizons, model))
        assigned = variant.set_index("window_id").stage09_label.loc[baseline.index]
        counts = assigned.value_counts()
        rows.append({"configuration": name, "radius_nm": radius, "materiality_percentile": percentile,
                     "horizons": ",".join(map(str, horizons)), "baseline": model,
                     **{label: int(counts.get(label, 0)) for label in ALLOWED},
                     "agreement_all_events": float((assigned == baseline).mean()),
                     "agreement_among_primary_classified": float((assigned.loc[baseline != "UNKNOWN"] ==
                                                                   baseline.loc[baseline != "UNKNOWN"]).mean()),
                     "UNKNOWN_to_interaction": int(((baseline == "UNKNOWN") &
                                                     (assigned == "LIKELY_INTERACTION_RELATED")).sum())})
        labels.append(pd.DataFrame({"window_id": baseline.index, "configuration": name,
                                    "stage09_label": assigned.to_numpy()}))
        print(f"Stage 09 V2 sensitivity {name}: {dict(counts)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "sensitivity_analysis_v2.csv", index=False)
    pd.concat(labels, ignore_index=True).to_parquet(OUT / "sensitivity_labels_v2.parquet", index=False)
    return result


def pick_manual_cases(v1: pd.DataFrame, summary: pd.DataFrame,
                      level: pd.DataFrame, models: pd.DataFrame):
    eligible = (level.loc[(level.candidate_partner_count.gt(0) |
                           level.any_material_interaction_evidence_50nm) &
                          level.window_id.isin(models.loc[models.robust_pre_state_available,
                                                         "window_id"])].copy())
    chosen = []
    roles = {}

    def take(part: pd.DataFrame, n: int, role: str):
        remaining = part.loc[~part.window_id.isin(chosen)]
        for key in remaining.window_id.head(n):
            chosen.append(key)
            roles[key] = role

    take(eligible.loc[eligible.any_material_interaction_evidence_50nm].sort_values(
        ["evidence_confidence_score_v2", "window_id"], ascending=[False, True]),
        5, "LEVEL_SINGLE_HORIZON_INTERACTION_CANDIDATE")
    non = eligible.loc[eligible.stage09_label.eq("LIKELY_NON_INTERACTION_OPERATIONAL")].copy()
    non["median_distance"] = (non.evidence_confidence_score_v2 -
                              non.evidence_confidence_score_v2.median()).abs()
    take(non.sort_values(["median_distance", "window_id"]), 7,
         "LEVEL_NON_INTERACTION_REPRESENTATIVE")
    unknown_available = eligible.loc[eligible.stage09_label.eq("UNKNOWN") &
                                     eligible.clearance_aware_available].copy()
    unknown_available["median_distance"] = (unknown_available.evidence_confidence_score_v2 -
                                             unknown_available.evidence_confidence_score_v2.median()).abs()
    take(unknown_available.sort_values(["median_distance", "window_id"]), 5,
         "LEVEL_UNKNOWN_EVALUABLE")
    unknown_unavailable = eligible.loc[eligible.stage09_label.eq("UNKNOWN") &
                                       ~eligible.clearance_aware_available].copy()
    unknown_unavailable["median_distance"] = (unknown_unavailable.evidence_confidence_score_v2 -
                                               unknown_unavailable.evidence_confidence_score_v2.median()).abs()
    take(unknown_unavailable.sort_values(["median_distance", "window_id"]),
         max(0, 20 - len(chosen)), "LEVEL_UNKNOWN_INTENT_UNAVAILABLE")
    if len(chosen) < 20:
        take(eligible.sort_values(["evidence_confidence_score_v2", "window_id"]),
             20 - len(chosen), "LEVEL_ADDITIONAL_AMBIGUOUS")
    assert len(chosen) >= 20
    v1_positives = v1.loc[v1.stage09_label.eq("LIKELY_INTERACTION_RELATED") &
                          v1.clearance_type.isin(["HEADING", "SPEED"]), "window_id"].tolist()
    assert len(v1_positives) == 2
    for key in v1_positives:
        if key not in chosen:
            chosen.append(key)
            roles[key] = "V1_HEADING_OR_SPEED_POSITIVE"
    cases = summary.loc[summary.window_id.isin(chosen)].copy()
    cases["manual_audit_role"] = cases.window_id.map(roles)
    if not {"clearance_type", "V1_label"} <= set(cases.columns):
        cases = cases.merge(v1[["window_id", "clearance_type", "stage09_label"]].rename(columns={
            "stage09_label": "V1_label"}), on="window_id", validate="one_to_one")
    cases["audit_order"] = cases.window_id.map({key: i for i, key in enumerate(chosen)})
    return cases.sort_values("audit_order").reset_index(drop=True)


def render_case_figure(case: dict, history: dict, model_row: dict,
                       index, partner_id: str, path: Path):
    anchor = float(history["event_epoch"])
    focal = index.last_before(history["flight_key"], anchor)
    neighbour = index.last_before(partner_id, anchor)
    if focal is None or neighbour is None:
        return False
    focal_robust, focal_ok, _ = index.robust_state(focal, anchor)
    neighbour_robust, neighbour_ok, _ = index.robust_state(neighbour, anchor)
    if not (focal_ok and neighbour_ok):
        return False
    pre = pre_event_metrics(index, history["flight_key"], anchor)
    scenario = focal_clearance_aware_state(history, focal, focal_robust, pre)
    grid = np.arange(0, 121, STEP, dtype=float)
    observed_focal, _ = index.sampled_trajectory(history["flight_key"], anchor, grid, focal)
    observed_neighbour, _ = index.sampled_trajectory(partner_id, anchor, grid, neighbour)
    f_a = tuple(v[:len(grid)] for v in scenario["aware"])
    f_b = tuple(v[:len(grid)] for v in scenario["baseline"])
    n_b = propagate(neighbour_robust, grid, anchor)
    obs_h, obs_v = pair_series(observed_focal["latitude"], observed_focal["longitude"],
                               observed_focal["altitude_ft"], observed_neighbour["latitude"],
                               observed_neighbour["longitude"], observed_neighbour["altitude_ft"])
    a_h, a_v = pair_series(*f_a, *n_b)
    b_h, b_v = pair_series(*f_b, *n_b)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(observed_focal["longitude"], observed_focal["latitude"],
            color="tab:blue", label="focal observed")
    if scenario["available"]:
        ax.plot(f_a[1], f_a[0], color="tab:green", linestyle="--",
                label="focal previous-instruction CF-A")
    ax.plot(f_b[1], f_b[0], color="tab:purple", linestyle=":",
            label="focal kinematic V1 CF-B")
    ax.plot(observed_neighbour["longitude"], observed_neighbour["latitude"],
            color="tab:orange", label="neighbour observed")
    ax.scatter([focal["longitude"], neighbour["longitude"]],
               [focal["latitude"], neighbour["latitude"]], marker="x", s=70, color="black",
               label="anchors")
    for second, marker in ((60, "o"), (120, "s")):
        j = second // STEP
        if np.isfinite(observed_focal["longitude"][j]):
            ax.scatter(observed_focal["longitude"][j], observed_focal["latitude"][j],
                       marker=marker, s=50, color="tab:blue", label=f"{second}-s mark")
    ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="Plan view and 60/120-s marks")
    ax.legend(fontsize=7, loc="best")
    for panel, observed, aware, baseline, unit, title in (
        (axes[0, 1], obs_h, a_h, b_h, "NM", "Horizontal pair separation"),
        (axes[1, 0], obs_v, a_v, b_v, "ft", "Vertical pair separation")):
        panel.plot(grid, observed, color="black", label="observed pair")
        if scenario["available"]:
            panel.plot(grid, aware, color="tab:green", linestyle="--", label="CF-A")
        panel.plot(grid, baseline, color="tab:purple", linestyle=":", label="CF-B")
        panel.axvline(60, color="gray", linestyle="--", alpha=.4)
        panel.axvline(120, color="gray", linestyle="--", alpha=.4)
        panel.set(xlabel="Seconds after clearance", ylabel=f"Separation ({unit})", title=title)
        panel.legend(fontsize=8)
    ax = axes[1, 1]
    for series, colour, name in ((obs_h, "black", "observed"),
                                 (a_h, "tab:green", "CF-A"), (b_h, "tab:purple", "CF-B")):
        if name == "CF-A" and not scenario["available"]:
            continue
        rolling_cpa = []
        rolling_tcpa = []
        for j in range(len(grid)):
            values = series[j:]
            if not np.isfinite(values).any():
                rolling_cpa.append(np.nan)
                rolling_tcpa.append(np.nan)
            else:
                rel = int(np.nanargmin(values))
                rolling_cpa.append(float(values[rel]))
                rolling_tcpa.append(float(grid[j + rel] - grid[j]))
        ax.plot(grid, rolling_cpa, color=colour, label=f"{name} rolling finite CPA")
        if name == "observed":
            twin = ax.twinx()
            twin.plot(grid, rolling_tcpa, color="tab:red", alpha=.65,
                      label="observed rolling TCPA")
            twin.set_ylabel("Observed TCPA within remaining 120 s (s)")
            twin.legend(loc="upper right", fontsize=7)
    ax.set(xlabel="Seconds after clearance", ylabel="Horizontal CPA within remaining 120 s (NM)",
           title="Rolling finite-horizon CPA/TCPA")
    ax.legend(loc="upper left", fontsize=7)
    old_instruction = (f"CFL {history['previous_level']} {history['previous_level_unit']}" if
                       history["clearance_type"] == "LEVEL" else
                       f"HDG {history['previous_heading']}" if history["clearance_type"] == "HEADING" else
                       f"SPD {history['previous_speed']} {history['previous_speed_unit']}")
    new_instruction = (f"CFL {history['current_level']} {history['current_level_unit']}" if
                       history["clearance_type"] == "LEVEL" else
                       f"HDG {history['current_heading']}" if history["clearance_type"] == "HEADING" else
                       f"SPD {history['current_speed']} {history['current_speed_unit']}")
    fig.suptitle(f"{history['clearance_type']} | {case['stage09_label']} | {case['window_id']}\n"
                 f"Previous: {old_instruction}; new: {new_instruction}; age {history['previous_instruction_age_s']:.0f} s; "
                 f"CF-A: {scenario['intent_state']} ({scenario['quality']}); partner {partner_id}", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_manual_audit(cases: pd.DataFrame, history: pd.DataFrame,
                       models: pd.DataFrame, pairs: pd.DataFrame):
    history_by = history.set_index("window_id").to_dict("index")
    model_by = models.set_index("window_id").to_dict("index")
    pair_by = {key: group for key, group in pairs.groupby("window_id", sort=False)}
    output = []
    for day in DAYS:
        day_cases = cases.loc[(cases.window_id.map(lambda key: history_by[key]["event_epoch"]) >= day.timestamp()) &
                              (cases.window_id.map(lambda key: history_by[key]["event_epoch"]) <
                               (day + pd.Timedelta(days=1)).timestamp())]
        if day_cases.empty:
            continue
        index = V1.NativeStateIndex(V1.load_day_states(day))
        for case in day_cases.to_dict("records"):
            hist = {**history_by[case["window_id"]], "window_id": case["window_id"]}
            part = pair_by.get(case["window_id"])
            if part is None or part.empty:
                continue
            if case["manual_audit_role"] == "LEVEL_SINGLE_HORIZON_INTERACTION_CANDIDATE":
                chosen = part.loc[part.clearance_aware_evidence].sort_values(
                    ["candidate_rank", "horizon_s"])
            else:
                chosen = part.loc[part.partner_id == case["candidate_partner_id"]]
            if chosen.empty:
                chosen = part.sort_values(["candidate_rank", "horizon_s"]).head(1)
            partner = chosen.partner_id.iloc[0]
            stem = case["window_id"].replace(":", "_")
            path = OUT / "figures" / f"{case['clearance_type']}_{stem}.png"
            rendered = render_case_figure(case, hist, model_by[case["window_id"]],
                                          index, partner, path)
            summary = part.loc[part.partner_id == partner]
            entry = {"window_id": case["window_id"], "clearance_type": case["clearance_type"],
                     "manual_audit_role": case["manual_audit_role"], "V1_label": case["V1_label"],
                     "V2_label": case["stage09_label"], "unknown_reason": case["unknown_reason"],
                     "previous_instruction_age_s": hist["previous_instruction_age_s"],
                     "previous_level_ft": hist["previous_level_ft"],
                     "current_level_ft": hist["current_level_ft"],
                     "previous_heading": hist["previous_heading"],
                     "current_heading": hist["current_heading"],
                     "previous_speed": hist["previous_speed"],
                     "current_speed": hist["current_speed"],
                     "intent_state": model_by[case["window_id"]].get("intent_state"),
                     "intent_quality": model_by[case["window_id"]].get("intent_quality"),
                     "partner_id": partner,
                     "figure_path": str(path.relative_to(OUT)) if rendered else ""}
            for horizon in MAIN:
                row = summary.loc[summary.horizon_s == horizon]
                if len(row):
                    record = row.iloc[0]
                    entry[f"usable_{horizon}s"] = bool(record.usable_v2)
                    entry[f"joint_delta_{horizon}s_nm"] = record.delta_joint_sep_nm
                    entry[f"vertical_delta_{horizon}s_ft"] = record.delta_vertical_sep
                    entry[f"threshold_joint_{horizon}s_nm"] = record.threshold_joint_p95_nm
                    entry[f"clearance_aware_evidence_{horizon}s"] = bool(record.clearance_aware_evidence)
                    entry[f"strong_scenario_disagreement_{horizon}s"] = bool(record.strong_scenario_disagreement)
            output.append(entry)
        del index
        gc.collect()
    audit = pd.DataFrame(output).sort_values("window_id").reset_index(drop=True)
    assert audit.clearance_type.eq("LEVEL").sum() >= 20
    assert audit.loc[audit.V1_label.eq("LIKELY_INTERACTION_RELATED"), "clearance_type"].isin(["HEADING", "SPEED"]).sum() == 2
    audit.to_csv(OUT / "manual_audit_cases.csv", index=False)
    audit[["window_id", "clearance_type", "V1_label", "V2_label", "manual_audit_role",
           "partner_id", "figure_path"]].to_csv(OUT / "representative_cases_v2.csv", index=False)
    return audit


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 30):
    header = "| " + " | ".join(columns) + " |"
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
    return "\n".join([header, rule, *body])


def level_model_difference(pairs: pd.DataFrame):
    frame = pairs.loc[pairs.clearance_type.eq("LEVEL") & pairs.horizon_s.isin(MAIN) &
                      pairs.usable_v2].copy()
    frame["absolute_A_minus_B_joint_nm"] = (frame.counterfactual_joint_sep -
                                              frame.v1_kinematic_min_joint_sep_nm).abs()
    rows = []
    for h in MAIN:
        part = frame.loc[frame.horizon_s == h]
        rows.append({"horizon_s": h, "usable_pair_rows": len(part),
                     "events": part.window_id.nunique(),
                     "median_absolute_A_minus_B_joint_nm": float(part.absolute_A_minus_B_joint_nm.median()),
                     "p95_absolute_A_minus_B_joint_nm": float(part.absolute_A_minus_B_joint_nm.quantile(.95)),
                     "model_difference_above_control_error": int((part.absolute_A_minus_B_joint_nm >
                                                                   part.threshold_joint_p95_nm).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "level_aware_vs_kinematic_difference.csv", index=False)
    return result


def write_level_report(level: pd.DataFrame, level_summary: pd.DataFrame,
                       unknown: pd.DataFrame, pairs: pd.DataFrame):
    model_difference = level_model_difference(pairs)
    level_reasons = level.loc[level.stage09_label.eq("UNKNOWN"), "unknown_reason"].str.split(";").explode().value_counts()
    reason_table = pd.DataFrame({"unknown_reason": level_reasons.index,
                                 "n_level_events": level_reasons.to_numpy()})
    delta = pd.read_csv(OUT / "level_delta_CFL_distribution.csv")
    lines = ["# Stage 09 V2 — LEVEL counterfactual audit", "",
             "This is an operational, finite-horizon *screen*, not a causal reconstruction or conflict label. "
             "The previous CFL is taken from the immediately preceding actionable SCAT record; it is not invented when missing. "
             "A record older than 30 minutes, inconsistent pre-event motion or unstable capture cannot initialize CF-A.", "",
             "## Requested LEVEL counts", "",
             markdown_table(level_summary, ["metric", "n_events"]), "",
             "The `previous_level_already_reached_but_unstable` group is retained as uncertain rather than forced into level flight. "
             "A fresh previous instruction does not guarantee an evaluable model; a valid pre-event trend and coherent target are also required.", "",
             "## CF-A previous-level intent versus CF-B V1 kinematics", "",
             markdown_table(model_difference, list(model_difference.columns)), "",
             "Neither 60-s nor 120-s evaluated LEVEL pair exceeds its TRAIN-control joint-geometry error in the **difference between CF-A and CF-B**. "
             "Thus the previous-CFL model does not add materially distinguishable short-horizon geometry in this pilot. "
             "The one LEVEL signal at 50 NM occurs at 120 s only, lacks 60/120 same-partner support and is not a positive label. "
             "A separate physical audit rejected a distant pair whose apparent improvement was horizontal while vertical separation did not improve.", "",
             "## New CFL direction and labels (descriptive only)", "",
             markdown_table(delta, list(delta.columns)), "",
             "CFL direction or magnitude was never used alone to assign a label. Some cruise anchors carry descent clearances to much lower levels; this is not automatically a conflict-related action.", "",
             "## Why LEVEL remains UNKNOWN", "",
             markdown_table(reason_table, ["unknown_reason", "n_level_events"], 20), "",
             "Reason codes overlap. The primary 30-NM cohort contains zero LEVEL cases with material clearance-aware interaction evidence at both 60 and 120 s. "
             "The 191 non-interaction labels require two usable horizons, confirmed response, complete evaluated candidate set and no material/convergent pair evidence; "
             "they do not prove operational routine or safety irrelevance.", ""]
    (OUT / "level_counterfactual_report.md").write_text("\n".join(lines), encoding="utf-8")
    return model_difference


def stage07_posthoc(summary: pd.DataFrame):
    scores = pd.read_parquet(ROOT / "07_explainability_cruise/test_predictions.parquet")
    scores = scores.loc[(scores.feature_family == "SCAT_FULL") & (scores.model == "xgboost") &
                        (scores.ATC_INTERVENTION == 1), ["window_id", "score"]]
    assert len(scores) == 592 and scores.window_id.is_unique
    joined = summary[["window_id", "stage09_label"]].merge(
        scores, on="window_id", how="inner", validate="one_to_one")
    assert len(joined) == 592
    table = joined.groupby("stage09_label").score.agg(["size", "mean", "median", "std"]).rename(columns={
        "size": "n_test_interventions", "mean": "mean_stage07_score",
        "median": "median_stage07_score", "std": "sd_stage07_score"}).reset_index()
    table["scope"] = "SCAT_FULL_XGBOOST_TEST_ONLY_NOT_POPULATION_PROBABILITY"
    table.to_csv(OUT / "stage07_posthoc_comparison_v2.csv", index=False)
    return table


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_config(history: pd.DataFrame, models: pd.DataFrame):
    v1_files = [Path(__file__).with_name("09_counterfactual_clearance_analysis_cruise.py"),
                V1_OUT / "counterfactual_event_summary.parquet",
                V1_OUT / "counterfactual_pair_horizon.parquet",
                ROOT / "PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md"]
    hashes = {str(path): sha256_file(path) for path in v1_files}
    config = {"analysis_scope": "FIRST_WEEK_PILOT", "week": PILOT_WEEK,
              "band": BAND, "cohort": COHORT, "method": METHOD,
              "iteration": "STAGE09_V2_CLEARANCE_AWARE",
              "v1_role": "KINEMATIC_COUNTERFACTUAL_BASELINE_READ_ONLY",
              "v1_sha256": hashes, "events": len(history),
              "main_horizons_s": MAIN, "sensitivity_horizons_s": (300, 600),
              "step_s": STEP, "main_neighbourhood_nm": 30,
              "sensitivity_neighbourhood_nm": 50,
              "previous_instruction_max_age_s": MAX_PREVIOUS_INSTRUCTION_AGE_S,
              "materiality_primary_percentile": .95,
              "materiality_sensitivity_percentile": .90,
              "materiality_source": "V1 usable TRAIN-control prediction error; dynamic-regime audit descriptive only",
              "level_mechanism": "previous-CFL target cap, no overshoot; material vertical and joint improvement required",
              "heading_mechanism": "previous active heading and material horizontal plus joint improvement required",
              "speed_mechanism": "comparable command units, observed pre-ground-speed trend; joint or timing change; no IAS/Mach-to-ground-speed substitution",
              "same_partner_60_120_required": True,
              "nested_horizons_not_independent": True,
              "stage07_scores_for_labeling": False,
              "opensky_rules_for_labeling": False,
              "safety_threshold_for_labeling": False,
              "allowed_labels": ALLOWED,
              "day_checkpoint_version": VERSION,
              "native_states_read_from": str(V1_OUT / "state_cache"),
              "model_availability_by_type": (models.loc[models.ATC_INTERVENTION == 1]
                                              .groupby("intent_state").clearance_aware_available.sum().astype(int).to_dict())}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def final_decision(summary: pd.DataFrame, level_difference: pd.DataFrame,
                   regime: pd.DataFrame):
    # No change to this gate may be driven by a desired positive count.
    unknown_share = summary.stage09_label.eq("UNKNOWN").mean()
    level_material_model_gain = int(level_difference.model_difference_above_control_error.sum())
    if unknown_share > .5 or level_material_model_gain == 0:
        return "INSUFFICIENT_COUNTERFACTUAL_EVIDENCE"
    if not regime.loc[(regime.dynamic_regime == "LEVEL_LIKE") &
                      (regime.horizon_s.isin(MAIN)) &
                      (regime.variable == "pair_min_joint_error_nm"), "calibration_stable"].all():
        return "CLEARANCE_AWARE_IMPROVEMENT_BUT_MORE_VALIDATION_NEEDED"
    return "READY_FOR_STAGE10_COMPARISON"


def write_report(v1: pd.DataFrame, history: pd.DataFrame, models: pd.DataFrame,
                 candidates: pd.DataFrame, pairs: pd.DataFrame, summary: pd.DataFrame,
                 comparison: pd.DataFrame, matrix: pd.DataFrame, label_counts: pd.DataFrame,
                 by_type: pd.DataFrame, unknown: pd.DataFrame, level_summary: pd.DataFrame,
                 level_difference: pd.DataFrame, regime: pd.DataFrame,
                 sensitivity: pd.DataFrame, audit: pd.DataFrame,
                 stage07: pd.DataFrame, decision: str):
    transitions = pd.read_csv(OUT / "clearance_transition_summary.csv")
    level = by_type.loc[by_type.clearance_type == "LEVEL"].iloc[0]
    main_pair = pairs.loc[pairs.horizon_s.isin(MAIN) & pairs.neighbourhood_nm.le(30)]
    coverage = pd.DataFrame([{"horizon_s": h,
                              "events_with_usable_30nm_pair": int(pairs.loc[
                                  pairs.horizon_s.eq(h) & pairs.neighbourhood_nm.le(30) &
                                  pairs.usable_v2, "window_id"].nunique()),
                              "focal_next_clearance_pair_rows": int(pairs.loc[
                                  pairs.horizon_s.eq(h), "FOCAL_NEXT_CLEARANCE"].sum()),
                              "neighbour_next_clearance_pair_rows": int(pairs.loc[
                                  pairs.horizon_s.eq(h), "NEIGHBOUR_NEXT_CLEARANCE"].sum())}
                             for h in HORIZONS])
    coverage.to_csv(OUT / "horizon_coverage_v2.csv", index=False)
    dynamic = regime.loc[regime.horizon_s.isin(MAIN) &
                         regime.variable.isin(["pair_min_vertical_error_ft",
                                                   "pair_min_joint_error_nm"])].copy()
    responses = summary.clearance_response_confirmed.value_counts()
    v1_unknown = int(v1.stage09_label.eq("UNKNOWN").sum())
    v2_unknown = int(summary.stage09_label.eq("UNKNOWN").sum())
    affected = comparison.loc[comparison.changed_label]
    roles = audit.manual_audit_role.value_counts().rename_axis("case_role").reset_index(name="n_cases")
    stage07_text = (markdown_table(stage07, ["stage09_label", "n_test_interventions",
                                            "median_stage07_score", "mean_stage07_score"]) if len(stage07)
                    else "No eligible Stage-07 TEST intervention scores.")
    lines = ["# SCAT Stage 09 V2 — clearance-aware counterfactual screen", "",
             "## 1. Scope", "",
             f"Only `{PILOT_WEEK}`, `{BAND}`, `{COHORT}`: **4,098** interventions (3,583 LEVEL, 458 SPEED, 57 HEADING). "
             "Stage 09 V1 is preserved as `KINEMATIC_COUNTERFACTUAL_BASELINE`. This V2 is `COUNTERFACTUAL_SCREEN_NOT_CAUSAL`. "
             "Stage 08, Stage 10, other SCAT weeks and OpenSky were not run.", "",
             "## 2. Why Stage 09 V2 was needed", "",
             "V1's constant speed/track/vertical-rate extrapolation can miss the operational meaning of a LEVEL clearance. "
             "V2 tests a finite-horizon continuation of the *previous* controller instruction against the observed response to the *new* instruction. "
             "The objective is credibility, not more positive labels or a smaller UNKNOWN share.", "",
             "## 3. Stage 09 V1 baseline", "",
             "[V1 report](PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md): 2 interaction-related, 475 non-interaction and 3,621 UNKNOWN (88.36%); final gate `INSUFFICIENT_COUNTERFACTUAL_EVIDENCE`. "
             "V2 reads V1 candidate reconstruction, native-state cache, pair horizons, control calibration and response checks. "
             "The SHA-256 identities of key V1 files are frozen in [V2 run_config.json](09_counterfactual_cruise_v2/run_config.json).", "",
             "## 4. Clearance-history reconstruction", "",
             "[Exact history](09_counterfactual_cruise_v2/clearance_history.parquet) joins each event to the preceding/current/following actionable Stage-03 record. "
             "For all 4,098 events, the prior full-record timestamp agrees with the preceding actionable record. "
             f"A previous command is *fresh* only when it is ≤{int(MAX_PREVIOUS_INSTRUCTION_AGE_S/60)} minutes old; "
             f"{int(history.previous_instruction_fresh.sum()):,} of 4,098 pass. Missing, old or incompatible values are not imputed.", "",
             "## 5. Transition-type inventory", "",
             markdown_table(transitions, ["clearance_type", "transition_class", "n_events"]), "",
             "The pilot cohort contains no simultaneous multi-action intervention, but the parser and tests detect it. "
             "HEADING/SPEED cancellation or absent prior assignment is not treated as a comparable numeric command.", "",
             "## 6. Counterfactual methodology", "",
             "CF-A continues a defensible previous instruction; CF-B preserves V1 robust pre-event kinematics. "
             "Both start exclusively from pre-event native states and use V1's 5-s grid and quality gates. "
             "Neighbour trajectories use the V1 symmetric robust pre-event state; observed trajectories remain native samples. "
             "The previous-command freshness, model availability and CF-A/CF-B disagreement are retained explicitly.", "",
             "## 7. LEVEL counterfactual", "",
             "[Dedicated LEVEL report](09_counterfactual_cruise_v2/level_counterfactual_report.md) and "
             "[event details](09_counterfactual_cruise_v2/level_counterfactual_details.parquet) show the previous-CFL intent state. "
             "A stable aircraft near the previous CFL holds its measured altitude; a coherent climb/descent continues only until the previous CFL, without overshoot. "
             "Old, missing, inconsistent or unstable-capture commands are uncertain. "
             "A LEVEL interaction signal requires calibrated **vertical mechanism** *and* continuous joint-geometry materiality; it never follows from the LEVEL type alone.", "",
             markdown_table(level_summary, ["metric", "n_events"]), "",
             "## 8. HEADING counterfactual", "",
             "CF-A uses the previous assigned heading only when fresh and compatible with the measured pre-event heading, preserving the observed heading-to-track offset. "
             "Circular 359°/1° arithmetic is tested. A material horizontal and joint-geometry change is required. "
             "Only one HEADING event has an evaluable prior-heading CF-A in this cohort; the V1 HEADING positive therefore becomes UNKNOWN, not a negative.", "",
             "## 9. SPEED counterfactual", "",
             "Previous/new IAS or Mach values are compared **only within the same command unit**. The commanded number is never substituted for ground speed. "
             "CF-A continues a bounded *observed pre-event ground-speed trend*; CF-B holds robust ground speed. "
             "SPEED response based on ground speed remains at most PARTIAL. Joint geometry or calibrated timing may support a signal; no large lateral change is required.", "",
             "## 10. Dynamic-regime control calibration", "",
             "[Regime calibration](09_counterfactual_cruise_v2/control_calibration_by_dynamic_regime.csv) uses **TRAIN controls only** for 90th/95th prediction-error quantiles and VALIDATION for stability. "
             "LEVEL-like, climb/descent-like, heading-stable and speed-stable subsets are assigned from pre-anchor kinematics, not clearance type. "
             "The global V1 TRAIN-control 95th percentile remains the primary materiality rule; regime results are diagnostic because climb/descent validation is sparse. "
             "Controls have no previous-command intervention, so this calibrates the reused kinematic approximation and observation error, not a direct ground truth for CF-A. "
             "These are model-error scales, never safety minima.", "",
             markdown_table(dynamic, ["dynamic_regime", "horizon_s", "variable",
                                      "train_control_windows", "validation_control_windows",
                                      "train_p95_error", "global_train_p95_error",
                                      "calibration_stable"], 20), "",
             "## 11. Neighbour/partner reconstruction", "",
             f"All **{len(candidates):,}** V1 pre-event candidate rows are reused. 30 NM is the primary *search radius*, 50 NM sensitivity only. "
             "The frozen pre-event candidate ranking selects multiple detailed partners; no observed outcome selects a partner. "
             "Non-interaction requires the complete candidate set to have been evaluated, so untested neighbours cannot silently become negatives.", "",
             "## 12. 60/120-s evaluation", "",
             "The primary label requires same-partner consistency at **60 and 120 s**. These are nested views of one encounter, not independent confirmation. "
             "The 300-s and 600-s outputs are quality-censored sensitivity/audit labels only and never determine the primary label. "
             "V1 focal/partner subsequent-clearance censoring and native-sample coverage are inherited unchanged.", "",
             markdown_table(coverage, list(coverage.columns)), "",
             "## 13. Clearance-response validation", "",
             "V1 response is reused. For LEVEL, V2 additionally requires the observed altitude to move closer to the new CFL than CF-A by more than the 95th TRAIN-control altitude-position error at that horizon. "
             "HEADING requires a known active prior assignment; SPEED remains PARTIAL at best. "
             f"Selected partner/horizon response status in the final event summary: {int(responses.get('YES',0)):,} YES, "
             f"{int(responses.get('PARTIAL',0)):,} PARTIAL, {int(responses.get('UNKNOWN',0)):,} UNKNOWN and {int(responses.get('NO',0)):,} NO. "
             "A recorded clearance alone is never a response.", "",
             "## 14. Label logic", "",
             "Only `LIKELY_INTERACTION_RELATED`, `LIKELY_NON_INTERACTION_OPERATIONAL`, `UNKNOWN`. "
             "Positive evidence requires usable 60/120-s pair rows with the **same** partner, a defensible CF-A, compatible focal response, pre-event joint convergence, type-coherent material geometry/timing beyond the global V1 TRAIN-control p95, and no strong opposite CF-B result or invalidating new clearance. "
             "Non-interaction requires both horizons, complete candidate coverage, YES response and no material/convergent pair evidence. "
             "Everything ambiguous is UNKNOWN with reason codes. `evidence_confidence_score_v2` is a transparent evidence-quality index, not a conflict probability: "
             "0.20×coverage + 0.20×history/model availability + 0.20×response + 0.20×scenario agreement + 0.10×horizon support + 0.10×effect/error magnitude; UNKNOWN capped at 0.49.", "",
             "## 15. LEVEL results", "",
             f"{int(level['LIKELY_INTERACTION_RELATED_n'])} interaction-related, {int(level['LIKELY_NON_INTERACTION_OPERATIONAL_n'])} non-interaction, "
             f"{int(level['UNKNOWN_n'])} UNKNOWN. Previous CFL exists for all 3,583, but only 2,141 previous records are fresh and 1,806 previous-intent models are usable. "
             "At 60/120 s, CF-A and CF-B have median **zero** absolute joint-geometry difference; no evaluated LEVEL pair exceeds its control-error threshold in the *difference between models*. "
             "Thus V2 does not materially distinguish LEVEL no-intervention trajectories from V1 over the primary horizons.", "",
             markdown_table(level_difference, list(level_difference.columns)), "",
             "## 16. HEADING results", "",
             markdown_table(by_type.loc[by_type.clearance_type.eq("HEADING")],
                            ["n_events", "previous_heading_available_percent", "clearance_aware_available_percent",
                             "LIKELY_INTERACTION_RELATED_n", "LIKELY_NON_INTERACTION_OPERATIONAL_n", "UNKNOWN_n"]), "",
             "The previous heading is recorded for 29/57, but only one passes all age, active-heading, command-change and pre-state checks. "
             "The V1 positive has no usable previous-heading instruction and becomes UNKNOWN.", "",
             "## 17. SPEED results", "",
             markdown_table(by_type.loc[by_type.clearance_type.eq("SPEED")],
                            ["n_events", "previous_speed_available_percent", "comparable_speed_units_percent",
                             "clearance_aware_available_percent", "LIKELY_INTERACTION_RELATED_n",
                             "LIKELY_NON_INTERACTION_OPERATIONAL_n", "UNKNOWN_n"]), "",
             "The single V2 positive is also the V1 SPEED positive; CF-A and CF-B are nearly coincident there. "
             "This is retained provisional evidence, not new proof from controller intent or confirmed conflict.", "",
             "## 18. V1 vs V2 comparison", "",
             f"[Per-event comparison](09_counterfactual_cruise_v2/stage09_v1_v2_comparison.csv) and "
             "[transition matrix](09_counterfactual_cruise_v2/label_transition_matrix.csv): "
             f"{len(comparison)-len(affected):,} unchanged, {len(affected):,} changed. "
             f"UNKNOWN rises from {v1_unknown:,} ({100*v1_unknown/4098:.2f}%) to {v2_unknown:,} ({100*v2_unknown/4098:.2f}%). "
             "No V1 UNKNOWN becomes interaction-related merely from the new model.", "",
             markdown_table(matrix, ["V1_label", "V2_label", "n_events"]), "",
             "## 19. Sensitivity analysis", "",
             "[Full sensitivity table](09_counterfactual_cruise_v2/sensitivity_analysis_v2.csv) compares 30/50-NM searches, 95th/90th TRAIN-control error, individual 60/120-s horizons, 60+120 consistency and V1 robust-kinematic evidence under the V2 horizon rule. "
             "The one positive remains one under 50 NM and 90th-percentile variants, but 120 s alone admits more single-horizon signals. "
             "Quality-censored 300-s and 600-s results are reported separately and do not feed back into primary labels; the larger UNKNOWN share is not relaxed away.", "",
             markdown_table(sensitivity, ["configuration", "LIKELY_INTERACTION_RELATED",
                                          "LIKELY_NON_INTERACTION_OPERATIONAL", "UNKNOWN",
                                          "agreement_among_primary_classified"]), "",
             "## 20. UNKNOWN analysis", "",
             "UNKNOWN is preferred when previous intent is old/inconsistent, response unconfirmed, candidate coverage incomplete, another clearance censors an outcome, or CF-A/CF-B/horizons disagree. "
             "Reasons overlap and are preserved rather than redistributed. The large 95.3% share is a substantive limit on inference.", "",
             markdown_table(unknown, ["unknown_reason", "n_events"], 20), "",
             "## 21. Representative/manual cases", "",
             f"[Manual case index](09_counterfactual_cruise_v2/manual_audit_cases.csv) and "
             f"[review notes](09_counterfactual_cruise_v2/manual_audit_review.md) cover **{int(audit.clearance_type.eq('LEVEL').sum())} LEVEL** cases plus both V1 HEADING/SPEED positives. "
             f"All {int(audit.figure_path.ne('').sum())} cases have [indexed figures](09_counterfactual_cruise_v2/representative_cases_v2.csv) showing previous/new commands, CF-A, CF-B, native focal/neighbour paths, 60/120 markers and continuous separation/CPA/TCPA. "
             "The audit includes a 50-NM LEVEL single-horizon candidate, non-interaction and unresolved previous-intent cases. "
             "No case-specific rule was tuned to make a label look attractive.", "",
             markdown_table(roles, ["case_role", "n_cases"]), "",
             "## 22. Stage 07 post-hoc comparison", "",
             "Only the 592 SCAT_FULL XGBoost **TEST** intervention scores were read, after V2 labels were frozen to Parquet. "
             "They measure intervention-versus-matched-control likelihood, not conflict or interaction-relatedness. "
             "No V2 positive has a Stage-07 TEST score, so its score distribution cannot be compared. "
             "No Stage-07 model was rerun or used to label V2.", "",
             stage07_text, "",
             "## 23. Implications for OpenSky", "",
             "OpenSky may later reproduce state tracks, finite-horizon kinematics, candidate search, relative geometry, CPA/TCPA and coverage checks. "
             "It does **not** natively contain the clearance event, previous/new CFL, assigned heading/speed, instruction units or TopSky/controller intent. "
             "V2 labels therefore cannot be directly generated in OpenSky. No OpenSky data was processed here.", "",
             "## 24. Limitations", "",
             "The previous record may be old or no longer operationally binding, wind and airspeed-vs-ground-speed differences remain unmodelled, and observed neighbour motion may depend on the focal action. "
             "Even CF-A is a scenario, not the factual no-clearance path. 60/120 s are nested and often too short for a level target-cap model to diverge from constant vertical rate. "
             "30/50 NM are candidate-search radii only; neither is a safety/conflict threshold. Top-three detailed trajectories can leave candidates untested, which blocks non-interaction labels. "
             "The 30-minute previous-command age and intent-coherence tolerances are conservative modelling choices, not independently validated controller procedures. "
             "Dynamic-regime errors are descriptive and climb/descent validation is sparse. A rare positive and 95.3% UNKNOWN preclude prevalence, causality or confirmed-conflict claims.", "",
             "## 25. Final decision", "",
             f"**{decision}**. The clearance history and bounded model are reproducible, but the LEVEL CF-A adds no material short-horizon geometry over V1, "
             "the prior heading/speed instruction is often unavailable or incompatible, and UNKNOWN dominates. "
             "This is not a reason to lower thresholds or recode UNKNOWN. "
             "Do **not** run Stage 08, Stage 10, another SCAT week or OpenSky under this authorization.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return coverage


def terminal_summary(history: pd.DataFrame, models: pd.DataFrame, summary: pd.DataFrame,
                     comparison: pd.DataFrame, by_type: pd.DataFrame,
                     unknown: pd.DataFrame, level_difference: pd.DataFrame,
                     decision: str):
    counts = summary.stage09_label.value_counts()
    level_model = models.loc[models.ATC_INTERVENTION == 1, ["window_id", "clearance_aware_available"]].merge(
        history[["window_id", "clearance_type"]], on="window_id", validate="one_to_one")
    print("=" * 60, "SCAT STAGE 09 V2 — CLEARANCE-AWARE COUNTERFACTUAL", "=" * 60,
          f"Week: {PILOT_WEEK}", "Band: FL290–FL410", "Events: 4098",
          "-" * 60, "CLEARANCE TYPES", "LEVEL: 3583", "HEADING: 57", "SPEED: 458",
          "-" * 60, "CLEARANCE HISTORY",
          f"Previous instruction fresh (≤30 min): {int(history.previous_instruction_fresh.sum())}",
          f"LEVEL with previous CFL: {int(history.loc[history.clearance_type.eq('LEVEL'), 'previous_level_ft'].notna().sum())}",
          f"HEADING with previous heading: {int(history.loc[history.clearance_type.eq('HEADING'), 'previous_heading'].notna().sum())}",
          f"SPEED with comparable previous instruction: {int(history.loc[history.clearance_type.eq('SPEED'), 'delta_speed_comparable'].notna().sum())}",
          "-" * 60, "LEVEL COUNTERFACTUAL",
          f"Evaluable intent: {int(level_model.loc[level_model.clearance_type.eq('LEVEL'), 'clearance_aware_available'].sum())}",
          f"Material CF-A-vs-CF-B difference above control error, 60/120 s: {int(level_difference.model_difference_above_control_error.sum())}",
          f"LEVEL positives / non-interaction / unknown: {int(by_type.loc[by_type.clearance_type.eq('LEVEL'), 'LIKELY_INTERACTION_RELATED_n'].iloc[0])} / {int(by_type.loc[by_type.clearance_type.eq('LEVEL'), 'LIKELY_NON_INTERACTION_OPERATIONAL_n'].iloc[0])} / {int(by_type.loc[by_type.clearance_type.eq('LEVEL'), 'UNKNOWN_n'].iloc[0])}",
          "-" * 60, "OVERALL V2 LABELS", sep="\n")
    for label in ALLOWED:
        n = int(counts.get(label, 0))
        print(f"{label}: {n} ({100*n/4098:.2f}%)")
    prior = comparison.set_index("window_id")
    print("-" * 60, "V1 -> V2", f"Unchanged: {int((~comparison.changed_label).sum())}",
          f"UNKNOWN -> INTERACTION: {int(((prior.V1_label=='UNKNOWN') & (prior.V2_label=='LIKELY_INTERACTION_RELATED')).sum())}",
          f"UNKNOWN -> NON_INTERACTION: {int(((prior.V1_label=='UNKNOWN') & (prior.V2_label=='LIKELY_NON_INTERACTION_OPERATIONAL')).sum())}",
          f"INTERACTION -> UNKNOWN: {int(((prior.V1_label=='LIKELY_INTERACTION_RELATED') & (prior.V2_label=='UNKNOWN')).sum())}",
          f"NON_INTERACTION -> UNKNOWN: {int(((prior.V1_label=='LIKELY_NON_INTERACTION_OPERATIONAL') & (prior.V2_label=='UNKNOWN')).sum())}",
          "-" * 60, "TOP UNKNOWN REASONS", sep="\n")
    for item in unknown.head(7).itertuples(index=False):
        print(f"{item.unknown_reason}: {item.n_events}")
    print("-" * 60, "INTERPRETATION",
          "Finite-horizon clearance-aware evidence, not confirmed conflict, causal necessity, loss of separation or population probability.",
          "-" * 60, "FINAL DECISION", decision,
          "DO NOT RUN STAGE 10 OR STAGE 08.",
          "DO NOT PROCESS OTHER SCAT WEEKS OR OPENSKY.", "=" * 60, sep="\n")







def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true",
                        help="Build only V2 read-only-input day checkpoints and inspect them")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate V2 summaries only; V1 is always read-only")
    args = parser.parse_args()
    if (OUT / "counterfactual_event_summary_v2.parquet").exists() and not args.overwrite and not args.extract_only:
        raise SystemExit("Completed V2 exists; inspect it before an explicit V2-only --overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    v1, candidates, pair_v1, evidence, responses, thresholds, calibration, day_events = read_inputs()
    history = build_history(v1)
    models, pair_v2 = process_days(history, pair_v1, day_events, thresholds, responses)
    if args.extract_only:
        print(f"V2 extraction ready: {len(models)} anchors, {len(pair_v2)} clearance-aware pair/horizon rows")
        return
    regime = calibrate_dynamic_regimes(calibration, models, thresholds)
    pairs = complete_v2_pair_table(pair_v1, pair_v2, evidence)
    pairs.to_parquet(OUT / "counterfactual_pair_horizon_v2.parquet", index=False, compression="zstd")
    models.to_parquet(OUT / "pre_event_model_diagnostics.parquet", index=False, compression="zstd")
    primary = classify_variant(v1, history, models, candidates, pairs, thresholds)
    diagnostic_labels = {
        h: classify_variant(v1, history, models, candidates, pairs, thresholds,
                            horizons=(h,)).set_index("window_id").stage09_label
        for h in (300, 600)}
    detail_fields = ["counterfactual_min_horizontal_sep", "counterfactual_min_vertical_sep",
                     "counterfactual_joint_sep", "counterfactual_CPA", "counterfactual_TCPA",
                     "observed_min_horizontal_sep", "observed_min_vertical_sep",
                     "observed_min_joint_sep_nm", "observed_CPA", "observed_TCPA",
                     "delta_horizontal_sep", "delta_vertical_sep", "delta_joint_sep_nm",
                     "delta_CPA", "delta_TCPA"]
    detail = pairs[["window_id", "partner_id", "horizon_s", *detail_fields]].rename(columns={
        "partner_id": "candidate_partner_id", "horizon_s": "primary_horizon"})
    summary = (primary.merge(history.drop(columns=["matched_pair_id"]),
                             on="window_id", validate="one_to_one")
               .merge(models.loc[models.ATC_INTERVENTION == 1,
                                 ["window_id", "dynamic_regime", "intent_state", "intent_quality",
                                  "clearance_aware_available", "altitude_trend_fpm",
                                  "speed_trend_kt_per_s", "robust_pre_state_available"]],
                      on="window_id", validate="one_to_one")
               .merge(v1[["window_id", "stage09_label", "evidence_confidence_score",
                          "candidate_partner_count_30nm", "candidate_partner_count_50nm",
                          "next_clearance_censored", "focal_state_age_s"]].rename(columns={
                   "stage09_label": "V1_label", "evidence_confidence_score": "V1_confidence"}),
                   on="window_id", validate="one_to_one")
               .merge(detail, on=["window_id", "candidate_partner_id", "primary_horizon"],
                      how="left", validate="one_to_one"))
    for h in (300, 600):
        usable = set(pairs.loc[pairs.horizon_s.eq(h) & pairs.neighbourhood_nm.le(30) &
                               pairs.usable_v2, "window_id"])
        summary[f"usable_{h}s"] = summary.window_id.isin(usable)
        summary[f"label_{h}s"] = summary.window_id.map(diagnostic_labels[h])
    assert len(summary) == 4098 and summary.window_id.is_unique
    assert set(summary.archive_week) == {PILOT_WEEK} and summary.analysis_altitude_ft.between(29000, 41000).all()
    # Freeze all 4,098 labels before opening any Stage-07 prediction file.
    summary.to_parquet(OUT / "counterfactual_event_summary_v2.parquet", index=False, compression="zstd")
    print(f"Stage 09 V2: frozen labels {summary.stage09_label.value_counts().to_dict()}", flush=True)
    comparison, matrix, label_counts, by_type, unknown, level, level_summary = write_result_tables(
        v1, history, models, candidates, pairs, summary)
    sensitivity = sensitivity_analysis(v1, history, models, candidates, pairs, thresholds, primary)
    manual_cases = pick_manual_cases(v1, summary, level, models)
    audit = write_manual_audit(manual_cases, history, models, pairs)
    level_difference = write_level_report(level, level_summary, unknown, pairs)
    config = write_config(history, models)
    stage07 = stage07_posthoc(summary)
    decision = final_decision(summary, level_difference, regime)
    write_report(v1, history, models, candidates, pairs, summary, comparison, matrix,
                 label_counts, by_type, unknown, level_summary, level_difference,
                 regime, sensitivity, audit, stage07, decision)
    for path, expected in config["v1_sha256"].items():
        assert sha256_file(Path(path)) == expected, f"V1 input unexpectedly changed: {path}"
    terminal_summary(history, models, summary, comparison, by_type, unknown,
                     level_difference, decision)


if __name__ == "__main__":
    main()


