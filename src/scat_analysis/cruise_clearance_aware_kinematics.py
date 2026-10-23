"""Pre-event controller-intent alternatives for the SCAT Stage 09 V2 screen.

These are transparent operational approximations, not causal trajectories or
aircraft-safety/conflict rules. Distances used below are quality tolerances for
checking an assigned instruction against the aircraft state, never pairwise
separation minima.
"""

from __future__ import annotations

import numpy as np

from cruise_counterfactual_kinematics import angular_difference_deg


MAX_PREVIOUS_INSTRUCTION_AGE_S = 1800.0
LEVEL_CAPTURE_TOLERANCE_FT = 300.0
LEVEL_STABLE_RATE_FPM = 150.0
LEVEL_MOVING_RATE_FPM = 150.0
HEADING_COMMAND_TOLERANCE_DEG = 15.0
MAX_SPEED_TREND_KT_PER_S = 0.25


def assigned_level_ft(value, unit):
    """SCAT F is hundreds of feet (FL); A is altitude in feet."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    if unit == "F":
        number *= 100.0
    elif unit != "A":
        return np.nan
    return number if 0 <= number <= 60000 else np.nan


def previous_is_fresh(age_s):
    return bool(np.isfinite(age_s) and 0 <= age_s <= MAX_PREVIOUS_INSTRUCTION_AGE_S)


def classify_previous_level_intent(previous_ft, age_s, anchor_altitude_ft,
                                   robust_rate_fpm, observed_pre_trend_fpm):
    """Classify only with pre-event values; uncertain states cannot be imposed."""
    if not np.isfinite(previous_ft):
        return "PREVIOUS_LEVEL_INTENT_UNKNOWN", "MISSING_PREVIOUS_CFL"
    if not previous_is_fresh(age_s):
        return "PREVIOUS_LEVEL_INTENT_UNKNOWN", "PREVIOUS_INSTRUCTION_TOO_OLD"
    if not all(np.isfinite(v) for v in (anchor_altitude_ft, robust_rate_fpm,
                                       observed_pre_trend_fpm)):
        return "PREVIOUS_LEVEL_INTENT_UNKNOWN", "INSUFFICIENT_PRE_EVENT_TREND"
    remaining = float(previous_ft - anchor_altitude_ft)
    rate = float(robust_rate_fpm)
    trend = float(observed_pre_trend_fpm)
    if abs(remaining) <= LEVEL_CAPTURE_TOLERANCE_FT:
        if abs(rate) <= LEVEL_STABLE_RATE_FPM and abs(trend) <= LEVEL_STABLE_RATE_FPM:
            return "LEVEL_AT_PREVIOUS_CFL", "OK"
        return "PREVIOUS_LEVEL_ALREADY_REACHED", "LEVEL_CAPTURE_NOT_STABLE"
    if abs(rate) >= LEVEL_MOVING_RATE_FPM and abs(trend) >= LEVEL_MOVING_RATE_FPM:
        if np.sign(rate) == np.sign(trend) == np.sign(remaining):
            return ("CLIMBING_TOWARD_PREVIOUS_CFL" if remaining > 0 else
                    "DESCENDING_TOWARD_PREVIOUS_CFL"), "OK"
        if np.sign(rate) == np.sign(trend) == -np.sign(remaining):
            return "MOVING_AWAY_FROM_PREVIOUS_CFL", "MOVING_AWAY_FROM_PREVIOUS_CFL"
    return "PREVIOUS_LEVEL_INTENT_UNKNOWN", "INCONSISTENT_PREVIOUS_LEVEL_INTENT"


def previous_level_altitudes(anchor_altitude_ft, target_ft, rate_fpm,
                             intent_state, seconds):
    """Continue a previous target without teleportation or target overshoot."""
    times = np.asarray(seconds, dtype=float)
    if np.any(times < 0):
        raise ValueError("Only nonnegative horizons are allowed")
    if intent_state == "LEVEL_AT_PREVIOUS_CFL":
        return np.full_like(times, float(anchor_altitude_ft), dtype=float)
    if intent_state in {"CLIMBING_TOWARD_PREVIOUS_CFL",
                        "DESCENDING_TOWARD_PREVIOUS_CFL"}:
        forecast = float(anchor_altitude_ft) + float(rate_fpm) * times / 60.0
        if intent_state == "CLIMBING_TOWARD_PREVIOUS_CFL":
            return np.minimum(forecast, float(target_ft))
        return np.maximum(forecast, float(target_ft))
    return np.full_like(times, np.nan, dtype=float)


def previous_heading_track(previous_heading_deg, measured_heading_deg,
                           measured_track_deg, age_s):
    """Retain measured wind/drift offset; reject an apparently inactive heading."""
    try:
        values = tuple(float(v) for v in (previous_heading_deg, measured_heading_deg,
                                          measured_track_deg))
    except (TypeError, ValueError):
        return np.nan
    if not previous_is_fresh(age_s) or not all(np.isfinite(v) for v in values):
        return np.nan
    if abs(float(angular_difference_deg(measured_heading_deg, previous_heading_deg))) > HEADING_COMMAND_TOLERANCE_DEG:
        return np.nan
    return float((previous_heading_deg + angular_difference_deg(measured_track_deg,
                                                                measured_heading_deg)) % 360.0)


def comparable_speed_delta(previous_speed, previous_unit, new_speed, new_unit):
    """Return a command delta only when the operational units really match."""
    if previous_unit not in {"KNOT", "MACH"} or previous_unit != new_unit:
        return np.nan
    try:
        before, after = float(previous_speed), float(new_speed)
    except (TypeError, ValueError):
        return np.nan
    if not all(np.isfinite(v) and v > 0 for v in (before, after)):
        return np.nan
    return after - before


def speed_trend_path(anchor_speed_kt, trend_kt_per_s, seconds):
    """Pre-event ground-speed trend, bounded as a model-quality guard."""
    times = np.asarray(seconds, dtype=float)
    if np.any(times < 0) or not all(np.isfinite(v) for v in (anchor_speed_kt, trend_kt_per_s)):
        raise ValueError("Invalid speed or time grid")
    if abs(trend_kt_per_s) > MAX_SPEED_TREND_KT_PER_S:
        return np.full_like(times, np.nan, dtype=float)
    speeds = float(anchor_speed_kt) + float(trend_kt_per_s) * times
    if np.any(speeds < 0):
        return np.full_like(times, np.nan, dtype=float)
    return speeds


def has_same_partner_across_horizons(supporting_rows):
    """Rows are (horizon, partner), and nested horizons are not independent."""
    at_60 = {partner for horizon, partner in supporting_rows if horizon == 60}
    at_120 = {partner for horizon, partner in supporting_rows if horizon == 120}
    return bool(at_60 & at_120)


def scenarios_not_opposed(kinematic_delta_joint_nm, control_error_nm):
    """A clearance-aware positive cannot override strong opposite V1 evidence."""
    return bool(np.isfinite(kinematic_delta_joint_nm) and
                np.isfinite(control_error_nm) and
                kinematic_delta_joint_nm >= -control_error_nm)

