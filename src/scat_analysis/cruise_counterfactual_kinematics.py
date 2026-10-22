"""Unit-explicit, finite-horizon geometry for the SCAT cruise Stage 09 screen.

These functions are descriptive kinematics, not conflict or causal models.
Track is clockwise from true north; x is east in nautical miles, y is north.
TCPA is seconds from the anchor and is clipped to [0, horizon] only in
``finite_cpa``; a negative raw TCPA means the pair is already diverging.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


NM_PER_DEG_LAT = 60.0
FT_PER_MIN_TO_FT_PER_SEC = 1.0 / 60.0
KNOT_TO_NM_PER_SEC = 1.0 / 3600.0
EARTH_RADIUS_NM = 3440.065
FT_PER_NM = 6076.12


def angular_difference_deg(a, b):
    """Signed shortest a-b difference, in [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def circular_median_deg(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    radians = np.deg2rad(values)
    direction = np.arctan2(np.median(np.sin(radians)), np.median(np.cos(radians)))
    return float(np.rad2deg(direction) % 360.0)


def xy_from_latlon(lat, lon, origin_lat, origin_lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - origin_lon) * NM_PER_DEG_LAT * np.cos(np.deg2rad(origin_lat))
    y = (lat - origin_lat) * NM_PER_DEG_LAT
    return x, y


def haversine_nm(lat1, lon1, lat2, lon2):
    a1, a2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = a2 - a1
    dlon = np.deg2rad(lon2 - lon1)
    hav = np.sin(dlat / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))


def velocity_nm_s(speed_kt, track_deg):
    heading = np.deg2rad(track_deg)
    speed = np.asarray(speed_kt) * KNOT_TO_NM_PER_SEC
    return speed * np.sin(heading), speed * np.cos(heading)


def propagate(state: dict, seconds, anchor_epoch: float):
    """Propagate a measured state to the anchor, then along a finite time grid."""
    t = np.asarray(seconds, dtype=float)
    if np.any(t < 0):
        raise ValueError("Only nonnegative seconds are allowed")
    age = anchor_epoch - float(state["time_epoch"])
    if age < -1e-6:
        raise ValueError("Counterfactual initial state must not be post-anchor")
    east, north = velocity_nm_s(float(state["ground_speed_kt"]), float(state["track_deg"]))
    dt = age + t
    lat = float(state["latitude"]) + north * dt / NM_PER_DEG_LAT
    cos_lat = np.cos(np.deg2rad(float(state["latitude"])))
    if cos_lat < 0.1:
        raise ValueError("Projection unstable near the pole")
    lon = float(state["longitude"]) + east * dt / (NM_PER_DEG_LAT * cos_lat)
    altitude = float(state["altitude_ft"]) + float(state["vertical_rate_fpm"]) * dt * FT_PER_MIN_TO_FT_PER_SEC
    return lat, lon, altitude


def finite_cpa(focal: dict, other: dict, anchor_epoch: float, horizon_s: int):
    """Continuous constant-velocity horizontal CPA, clipped to a finite horizon."""
    if horizon_s <= 0:
        raise ValueError("horizon must be positive")
    fa, fo, _ = propagate(focal, np.array([0.0]), anchor_epoch)
    oa, oo, _ = propagate(other, np.array([0.0]), anchor_epoch)
    ox, oy = xy_from_latlon(oa[0], oo[0], fa[0], fo[0])
    vfx, vfy = velocity_nm_s(focal["ground_speed_kt"], focal["track_deg"])
    vox, voy = velocity_nm_s(other["ground_speed_kt"], other["track_deg"])
    rvx, rvy = float(vox - vfx), float(voy - vfy)
    vv = rvx * rvx + rvy * rvy
    raw_tcpa = -float(ox * rvx + oy * rvy) / vv if vv > 1e-14 else np.nan
    clipped = float(np.clip(raw_tcpa, 0.0, horizon_s)) if np.isfinite(raw_tcpa) else 0.0
    cpa = float(np.hypot(ox + rvx * clipped, oy + rvy * clipped))
    current = float(np.hypot(ox, oy))
    horizontal_closing = -float(ox * rvx + oy * rvy) / max(current, 1e-9) * 3600.0
    return {"cpa_nm": cpa, "tcpa_s": clipped, "raw_tcpa_s": raw_tcpa,
            "current_horizontal_sep_nm": current, "horizontal_closing_kt": horizontal_closing}


def nearest_within_radius(focal: dict, snapshot: list[dict], radius_nm: float) -> list[dict]:
    """KD-tree prefilter, then great-circle distance; returns all candidates."""
    if not snapshot:
        return []
    lat = np.asarray([s["latitude"] for s in snapshot], dtype=float)
    lon = np.asarray([s["longitude"] for s in snapshot], dtype=float)
    x, y = xy_from_latlon(lat, lon, focal["latitude"], focal["longitude"])
    tree = cKDTree(np.column_stack([x, y]))
    # Small projection guard, exact great-circle distance makes the final cut.
    keep = tree.query_ball_point([0.0, 0.0], radius_nm * 1.02)
    rows = []
    for i in keep:
        state = snapshot[i]
        if state["flight_key"] == focal["flight_key"]:
            continue
        distance = float(haversine_nm(focal["latitude"], focal["longitude"], state["latitude"], state["longitude"]))
        if distance <= radius_nm:
            rows.append({**state, "current_horizontal_sep_nm": distance})
    return sorted(rows, key=lambda row: (row["current_horizontal_sep_nm"], row["flight_key"]))


def pair_series(focal_lat, focal_lon, focal_alt, other_lat, other_lon, other_alt):
    horizontal = haversine_nm(np.asarray(focal_lat), np.asarray(focal_lon),
                              np.asarray(other_lat), np.asarray(other_lon))
    vertical = np.abs(np.asarray(focal_alt) - np.asarray(other_alt))
    return horizontal, vertical


def joint_separation_nm(horizontal_nm, vertical_ft):
    """Three-dimensional Euclidean proximity proxy in nautical miles.

    Continuous geometry only: neither a safety minimum nor a conflict label.
    """
    return np.hypot(np.asarray(horizontal_nm, dtype=float),
                    np.asarray(vertical_ft, dtype=float) / FT_PER_NM)


def next_clearance_censors(next_epoch: float, event_epoch: float, horizon_s: int) -> bool:
    return bool(np.isfinite(next_epoch) and event_epoch < next_epoch <= event_epoch + horizon_s)


def interaction_agreement_for_clearance(clearance_type, joint_agreement, timing_agreement):
    """Permit timing-only corroboration for SPEED, not LEVEL or HEADING."""
    return joint_agreement | ((np.asarray(clearance_type) == "SPEED") & timing_agreement)


def conservative_label(has_coverage: bool, response: str, concordant: bool,
                       converging: bool, material_improvement: bool,
                       has_new_clearance: bool, evidence_horizons: int,
                       model_agreement: bool) -> str:
    """A deliberately strict, testable final-label gate; no safety minima."""
    if (not has_coverage or response not in {"YES", "PARTIAL"} or
            has_new_clearance or not model_agreement):
        return "UNKNOWN"
    if concordant and converging and material_improvement and evidence_horizons >= 2:
        return "LIKELY_INTERACTION_RELATED"
    if not converging and not material_improvement and evidence_horizons >= 2 and response == "YES":
        return "LIKELY_NON_INTERACTION_OPERATIONAL"
    return "UNKNOWN"
