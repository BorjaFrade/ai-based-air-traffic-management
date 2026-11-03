#!/usr/bin/env python3
"""Stage 15: predicted-separation severity vector at ATC orders and no-order controls.

Question
--------
Stages 13 and 14 describe the *current* geometry to the nearest neighbour when
an ATC level or heading order is recorded. This stage asks the next question:

    If every aircraft kept its velocity, how severe would the predicted
    separation be, in each look-ahead window (t=0, 0-1, 1-2, 2-5, 5-10 min),
    for the *worst* neighbour, and how does that compare with comparable
    moments in which the same aircraft received no order?

Unit of analysis
----------------
One *query* = one aircraft at one instant. Queries are

* ``event``   : the 50,774 level orders (Stage 13) and the 1,176 heading orders
                (Stage 14), exactly as those stages select them;
* ``control`` : up to K instants of the SAME flight, observed in FL290-410, at
                least ``--clearance-guard-s`` seconds away from any clearance of
                that flight, with the same vertical-rate class (climb / level /
                descent) as the event it is matched to.

For each query every other aircraft observed in FL290-410 within
``--radius-nm`` (latest native plot, at most 10 s old, as in Stages 13/14) is
propagated at constant velocity on a 5 s grid up to 10 min. Pair states use the
ENROUTE dynamic rule of ``src/acquisition/separation_profiles.json`` (RED /
ORANGE / YELLOW / NORMAL with the active-separator condition), vectorised here
and checked against ``opensky_safety.classify_pair_dynamic`` by the tests.

Two vertical models are produced for the focal aircraft:

* ``A`` (primary, all queries): observed vertical rate held constant with the
  profile dead-band. Same assumption as the detector; describes what the
  controller *sees* before the order takes effect.
* ``C`` (level orders only, columns prefixed ``post_``): the focal aircraft
  follows the cleared level (reaction delay, then the observed rate if it
  already points to the new level, otherwise a nominal rate). It is a
  what-if for "does the order remove a predicted loss?" and is a modelling
  assumption, not an observation. Horizontal motion is identical in A and C.

Nothing here uses a future observation: only plots at or before the query time
enter the prediction. No clearance is treated as a conflict label.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage13_level_change_geometry as s13  # noqa: E402
import stage14_heading_change_geometry as s14  # noqa: E402

BASE = s13.BASE
OUT = BASE / "15_predicted_separation"
STATES = s13.STATES
CLEARANCES = s13.CLEARANCES
PROFILE_PATH = HERE.parent / "acquisition" / "separation_profiles.json"
MAX_AGE_S = s13.MAX_AGE_S
EARTH_RADIUS_NM = s13.EARTH_RADIUS_NM
HORIZON_S = 600
STATE_NAMES = {0: "NORMAL", 1: "YELLOW", 2: "ORANGE", 3: "RED"}
EVENT_COLS = [
    "event_id", "order_type", "clearance_id", "archive_week", "flight_key",
    "source_member", "clearance_epoch", "matched_state_epoch", "matched_latitude",
    "matched_longitude", "matched_altitude_ft", "matched_flight_level",
    "matched_ground_speed_kt", "matched_track_deg", "matched_vertical_rate_fpm",
    "previous_cfl", "new_cfl", "direction", "absolute_heading_instruction_from_observed_deg",
    "turn_direction", "heading_event_kind",
]


# --------------------------------------------------------------------------- rules
def load_rules(path: Path = PROFILE_PATH) -> dict:
    """Read the ENROUTE thresholds from the separation profile (single source of truth)."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))["ENROUTE"]
    st = profile["states"]
    red, orange, yellow = (st["RED_LOSS_OF_SEPARATION"], st["ORANGE_HIGH_RISK"],
                           st["YELLOW_PROXIMITY_WARNING"])
    return {
        "logic_version": profile["prediction"].get("logic_version", profile.get("logic_version")),
        "red_h": float(red["horizontal_nm"]), "red_v": float(red["vertical_ft"]),
        "tol": float(red.get("vertical_tolerance_ft", 0.0)),
        "deadband": float(profile["prediction"]["vertical_rate_deadband_fpm"]),
        "step": int(profile["prediction"]["prediction_step_seconds"]),
        "o_h": float(orange["horizontal_nm"]), "o_v": float(orange["vertical_ft"]),
        "y_h": float(yellow["horizontal_nm"]), "y_v": float(yellow["vertical_ft"]),
        "eps_h": float(yellow.get("horizontal_deterioration_epsilon_nm", 0.01)),
        "eps_v": float(yellow.get("vertical_deterioration_epsilon_ft", 1.0)),
    }


def make_windows(step: int, horizon: int = HORIZON_S) -> dict[str, tuple[int, int]]:
    """Inclusive grid-index windows: t=0, (0,1], (1,2], (2,5], (5,10] minutes."""
    k = lambda s: s // step  # noqa: E731
    return {"0": (0, 0), "1": (1, k(60)), "2": (k(60) + 1, k(120)),
            "5": (k(120) + 1, k(300)), "10": (k(300) + 1, k(horizon))}


def classify_states(H: np.ndarray, V: np.ndarray, rules: dict) -> np.ndarray:
    """Vectorised ENROUTE dynamic classification.

    ``H`` (NM) and ``V`` (ft, absolute) have shape (N, K+1). The state at sample k
    needs sample k+1 for the active-separator trend, so the result has shape
    (N, K) with 0=NORMAL, 1=YELLOW, 2=ORANGE, 3=RED.
    """
    Vn = np.where(np.abs(V - rules["red_v"]) <= rules["tol"], rules["red_v"], V)
    h0, v0, h1, v1 = H[:, :-1], Vn[:, :-1], H[:, 1:], Vn[:, 1:]
    h_red = h0 < rules["red_h"]
    v_red = v0 < rules["red_v"]
    h_det = h1 < h0 - rules["eps_h"]
    v_det = v1 < v0 - rules["eps_v"]
    active = np.where(h_red & ~v_red, v_det, np.where(~h_red & v_red, h_det, h_det & v_det))
    orange = active & (h0 < rules["o_h"]) & (v0 < rules["o_v"])
    yellow = active & (h0 < rules["y_h"]) & (v0 < rules["y_v"])
    return np.where(h_red & v_red, 3, np.where(orange, 2, np.where(yellow, 1, 0))).astype(np.int8)


# --------------------------------------------------------------------------- geometry
def deadbanded(vr: np.ndarray | float, deadband: float):
    vr = np.nan_to_num(np.asarray(vr, dtype=float), nan=0.0)
    return np.where(np.abs(vr) < deadband, 0.0, vr)


def focal_post_order_altitude(z0: float, vr0: float, target: float, grid_s: np.ndarray,
                              delay_s: float, nominal_fpm: float, deadband: float) -> np.ndarray:
    """Altitude of the focal aircraft if it complies with the cleared level.

    Before ``delay_s`` the observed vertical rate (dead-banded) continues. Afterwards
    the aircraft moves toward ``target`` at the observed rate when it already points
    that way at >=500 fpm, otherwise at ``nominal_fpm``; it stops at ``target``.
    """
    vr0 = float(deadbanded(vr0, deadband))
    z_delay = z0 + vr0 * delay_s / 60.0
    sign = np.sign(target - z_delay)
    pre = z0 + vr0 * np.minimum(grid_s, delay_s) / 60.0
    if sign == 0:
        return np.full_like(grid_s, target, dtype=float) if abs(z0 - target) < 1 else pre
    rate = abs(vr0) if (np.sign(vr0) == sign and abs(vr0) >= 500.0) else nominal_fpm
    moved = z_delay + sign * rate * np.maximum(grid_s - delay_s, 0.0) / 60.0
    moved = np.minimum(moved, target) if sign > 0 else np.maximum(moved, target)
    return np.where(grid_s <= delay_s, pre, moved)


def summarise(S: np.ndarray, M: np.ndarray, windows: dict, step: int, prefix: str = "") -> dict:
    """Window / cumulative severity and timing features for one query."""
    n = S.shape[0]
    out: dict = {}
    run_state, run_min = 0, np.inf
    for name, (a, b) in windows.items():
        w_state = int(S[:, a:b + 1].max()) if n else 0
        w_min = float(M[:, a:b + 1].min()) if n else np.nan
        run_state = max(run_state, w_state)
        run_min = min(run_min, w_min) if n else np.nan
        out[f"{prefix}state_w{name}"] = w_state
        out[f"{prefix}minnorm_w{name}"] = w_min
        out[f"{prefix}state_c{name}"] = run_state
        out[f"{prefix}minnorm_c{name}"] = run_min
    for label, level in (("red", 3), ("orange", 2), ("nonnormal", 1)):
        hit = S >= level if n else np.zeros((0, 1), dtype=bool)
        any_hit = hit.any(axis=1) if n else np.zeros(0, dtype=bool)
        if n and any_hit.any():
            first = np.where(any_hit, hit.argmax(axis=1), S.shape[1]).min()
            out[f"{prefix}t_first_{label}_s"] = float(first * step)
        else:
            out[f"{prefix}t_first_{label}_s"] = np.nan
        out[f"{prefix}n_{label}_nb"] = int(any_hit.sum())
    return out


def evaluate_pairs(dx, dy, vxn, vyn, zn, vrn, focal: dict, rules: dict, grid_s: np.ndarray,
                   windows: dict, post_target: float | None, delay_s: float,
                   nominal_fpm: float) -> dict:
    """Predicted pair states for one focal aircraft against N neighbours.

    ``dx, dy`` : neighbour minus focal position at the query time (NM)
    ``vxn, vyn`` : neighbour velocity (NM/s); ``zn, vrn``: altitude (ft) and dead-banded rate
    """
    step = rules["step"]
    K = len(grid_s) - 1
    rvx, rvy = vxn - focal["vx"], vyn - focal["vy"]
    H = np.hypot(dx[:, None] + rvx[:, None] * grid_s[None, :],
                 dy[:, None] + rvy[:, None] * grid_s[None, :])
    zf = focal["z"] + focal["vr"] * grid_s / 60.0
    zn_t = zn[:, None] + vrn[:, None] * grid_s[None, :] / 60.0
    V = np.abs(zn_t - zf[None, :])
    S = classify_states(H, V, rules)
    M = np.maximum(H[:, :K] / rules["red_h"], V[:, :K] / rules["red_v"])
    out = summarise(S, M, windows, step)
    best = int(np.argmin(M.min(axis=1)))
    k_best = int(np.argmin(M[best]))
    out.update(rel_idx=best, rel_minnorm=float(M[best, k_best]), rel_t_min_s=float(k_best * step),
               rel_h_now_nm=float(H[best, 0]), rel_v_now_ft=float(V[best, 0]),
               rel_h_at_min_nm=float(H[best, k_best]), rel_v_at_min_ft=float(V[best, k_best]),
               rel_closing_kt=float(3600.0 * math.hypot(rvx[best], rvy[best])))
    out["nearest_h_nm"] = float(H[:, 0].min())
    out["nearest_v_ft"] = float(V[int(np.argmin(H[:, 0])), 0])
    if post_target is not None:
        zf_post = focal_post_order_altitude(focal["z"], focal["vr_raw"], post_target, grid_s,
                                            delay_s, nominal_fpm, rules["deadband"])
        Vp = np.abs(zn_t - zf_post[None, :])
        Sp = classify_states(H, Vp, rules)
        Mp = np.maximum(H[:, :K] / rules["red_h"], Vp[:, :K] / rules["red_v"])
        out.update(summarise(Sp, Mp, windows, step, prefix="post_"))
    return out


# --------------------------------------------------------------------------- data access
def load_week_states_lean(parquet: pq.ParquetFile, week: str) -> tuple[pd.DataFrame, dict]:
    """Same selection and ordering as ``stage13.load_week_states`` with bounded peak memory.

    Row groups are filtered one at a time so a week never needs the full row groups in RAM.
    """
    groups = s13.week_row_groups(parquet, week)
    if not groups:
        raise ValueError(f"No surveillance row groups for {week}")
    numeric = ["time_epoch", "latitude", "longitude", "altitude_ft", "flight_level",
               "ground_speed_kt", "track_deg", "vertical_rate_fpm"]
    parts, native_rows = [], 0
    for g in groups:
        frame = parquet.read_row_group(g, columns=s13.STATE_COLS, use_threads=True).to_pandas()
        frame = frame.loc[frame.archive_week.eq(week)]
        native_rows += len(frame)
        for field in numeric:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        keep = frame.flight_key.notna() & frame.source_member.notna() & \
            frame.time_epoch.notna() & frame.latitude.between(-90, 90) & \
            frame.longitude.between(-180, 180) & frame.altitude_ft.between(0, 60000) & \
            frame.flight_level.between(s13.FL_LOW, s13.FL_HIGH) & \
            ~frame.on_ground.fillna(False).astype(bool)
        parts.append(frame.loc[keep].copy())
        del frame
    # String columns as categoricals keep a week within a few hundred MB.
    from pandas.api.types import union_categoricals
    for column in ("flight_key", "source_member"):
        for part in parts:
            part[column] = part[column].astype("category")
        categories = union_categoricals([part[column] for part in parts]).categories
        for part in parts:
            part[column] = part[column].cat.set_categories(categories)
    states = pd.concat(parts, ignore_index=True)
    del parts
    members = states.groupby("flight_key", sort=False, observed=True).source_member.nunique()
    collision_keys = set(members.index[members.gt(1)].astype(str))
    if collision_keys:
        entity = states.flight_key.astype(str)
        mask = states.flight_key.isin(collision_keys)
        entity[mask] = entity[mask] + "::" + states.source_member[mask].astype(str)
        states["entity_key"] = entity.astype("category")
        del entity
    else:
        states["entity_key"] = states.flight_key
    states.sort_values(["time_epoch", "plot_index"], kind="mergesort", inplace=True)
    states.reset_index(drop=True, inplace=True)
    audit = {"week": week, "native_rows": native_rows, "in_band_valid_airborne_rows": len(states),
             "colliding_flight_keys": len(collision_keys), "row_groups": len(groups)}
    return states, audit


def load_events() -> tuple[pd.DataFrame, dict]:
    """Union of the Stage 13 (level) and Stage 14 (heading) eligible orders."""
    level, level_audit = s13.load_events()
    heading, heading_audit = s14.load_events()
    level = level.assign(order_type="LEVEL")
    heading = heading.assign(order_type="HEADING")
    frames = []
    for frame in (level, heading):
        frame = frame.copy()
        frame["event_id"] = frame.order_type + ":" + frame.clearance_id.astype(str)
        for column in EVENT_COLS:
            if column not in frame.columns:
                frame[column] = np.nan
        frames.append(frame[EVENT_COLS])
    events = pd.concat(frames, ignore_index=True)
    if events.event_id.duplicated().any():
        raise ValueError("Duplicated event ids")
    events = events.sort_values(["archive_week", "clearance_epoch", "event_id"],
                                kind="mergesort").reset_index(drop=True)
    return events, {"level": level_audit, "heading": heading_audit,
                    "events": int(len(events)),
                    "level_events": int(len(level)), "heading_events": int(len(heading))}


_CLEARANCE_CACHE: dict = {}


def load_clearance_times(week: str, collision_keys: set[str]) -> dict[str, np.ndarray]:
    """All clearance epochs of each flight (any type), used to keep controls order-free."""
    if "table" not in _CLEARANCE_CACHE:
        _CLEARANCE_CACHE["table"] = pd.read_parquet(
            CLEARANCES, columns=["archive_week", "flight_key", "source_member", "clearance_epoch"])
    table = _CLEARANCE_CACHE["table"]
    table = table.loc[table.archive_week.eq(week)].copy()
    table["clearance_epoch"] = pd.to_numeric(table.clearance_epoch, errors="coerce")
    table = table.dropna(subset=["flight_key", "clearance_epoch"])
    key = table.flight_key.astype(str)
    table["entity"] = np.where(key.isin(collision_keys),
                               key + "::" + table.source_member.astype(str), key)
    return {entity: np.sort(group.clearance_epoch.to_numpy(dtype=float))
            for entity, group in table.groupby("entity", sort=False)}


def build_arrays(states: pd.DataFrame, deadband: float, vertical_model: str = "rate") -> dict:
    arrays = {name: states[name].to_numpy() for name in [
        "flight_key", "source_member", "entity_key", "time_epoch", "altitude_ft", "flight_level",
        "ground_speed_kt", "track_deg", "vertical_rate_fpm"]}
    arrays["lat_rad"] = np.deg2rad(states.latitude.to_numpy(dtype=float))
    arrays["lon_rad"] = np.deg2rad(states.longitude.to_numpy(dtype=float))
    codes, names = pd.factorize(states.entity_key, sort=False)
    arrays["entity_code"] = codes.astype(np.int64)
    arrays["code_of"] = {name: i for i, name in enumerate(names)}
    arrays["vr_eff"] = deadbanded(states.vertical_rate_fpm.to_numpy(dtype=float), deadband)
    if vertical_model == "level":  # sensitivity: nobody changes altitude during the prediction
        arrays["vr_eff"] = np.zeros_like(arrays["vr_eff"])
    speed = states.ground_speed_kt.to_numpy(dtype=float)
    track = np.deg2rad(states.track_deg.to_numpy(dtype=float))
    arrays["vx"] = speed * np.sin(track) / 3600.0
    arrays["vy"] = speed * np.cos(track) / 3600.0
    arrays["kin_ok"] = np.isfinite(arrays["vx"]) & np.isfinite(arrays["vy"])
    order = np.argsort(arrays["entity_code"], kind="stable")
    cut = np.flatnonzero(np.diff(arrays["entity_code"][order])) + 1
    arrays["entity_rows"] = {int(arrays["entity_code"][g[0]]): np.sort(g)
                             for g in np.split(order, cut) if len(g)}
    return arrays


def vertical_class(vr: np.ndarray | float, threshold: float):
    vr = np.asarray(vr, dtype=float)
    return np.where(vr > threshold, 1, np.where(vr < -threshold, -1, 0))


def pick_controls(event: dict, arrays: dict, clearance_times: dict[str, np.ndarray], k: int,
                  seed: int, guard_s: float, class_threshold: float) -> np.ndarray:
    """Row indices of up to ``k`` order-free states of the same flight with the same vertical class."""
    entity = str(event["entity"])
    code = arrays["code_of"].get(entity)
    vr = event["matched_vertical_rate_fpm"]
    if code is None or not np.isfinite(vr):
        return np.zeros(0, dtype=np.int64)
    rows = arrays["entity_rows"][code]
    times = arrays["time_epoch"][rows]
    ok = arrays["kin_ok"][rows] & np.isfinite(arrays["vertical_rate_fpm"][rows]) & \
        (vertical_class(arrays["vertical_rate_fpm"][rows], class_threshold) ==
         vertical_class(vr, class_threshold)) & (np.abs(times - event["clearance_epoch"]) > guard_s)
    ce = clearance_times.get(entity)
    if ce is not None and len(ce):
        idx = np.searchsorted(ce, times)
        left = ce[np.clip(idx - 1, 0, len(ce) - 1)]
        right = ce[np.clip(idx, 0, len(ce) - 1)]
        ok &= np.minimum(np.abs(times - left), np.abs(times - right)) > guard_s
    candidates = rows[ok]
    if not len(candidates):
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng([seed, zlib.crc32(str(event["event_id"]).encode())])
    return np.sort(rng.choice(candidates, size=min(k, len(candidates)), replace=False))


def gather_neighbours(arrays: dict, query: dict, start: int, end: int, radius_nm: float):
    """Latest plot (<=10 s old) of every other entity, aligned to the query time."""
    if end <= start:
        return None
    pos = np.arange(start, end)
    _, first_reversed = np.unique(arrays["entity_code"][pos][::-1], return_index=True)
    p = pos[-1 - first_reversed]
    p = p[(arrays["entity_code"][p] != query["code"]) & arrays["kin_ok"][p]]
    if not len(p):
        return None
    lat0, lon0 = math.radians(query["lat"]), math.radians(query["lon"])
    dlat = arrays["lat_rad"][p] - lat0
    dlon = (arrays["lon_rad"][p] - lon0 + np.pi) % (2 * np.pi) - np.pi
    dx0 = EARTH_RADIUS_NM * dlon * np.cos((lat0 + arrays["lat_rad"][p]) / 2)
    dy0 = EARTH_RADIUS_NM * dlat
    age_n = query["epoch"] - arrays["time_epoch"][p]
    dx = dx0 + arrays["vx"][p] * age_n - query["vx"] * query["age"]
    dy = dy0 + arrays["vy"][p] * age_n - query["vy"] * query["age"]
    inside = np.hypot(dx, dy) <= radius_nm
    if not inside.any():
        return None
    p, dx, dy, age_n = p[inside], dx[inside], dy[inside], age_n[inside]
    return {"rows": p, "dx": dx, "dy": dy, "vx": arrays["vx"][p], "vy": arrays["vy"][p],
            "z": arrays["altitude_ft"][p].astype(float) + arrays["vr_eff"][p] * age_n / 60.0,
            "vr": arrays["vr_eff"][p]}


def focal_from_row(arrays: dict, i: int, rules: dict) -> dict:
    return {"lat": float(np.rad2deg(arrays["lat_rad"][i])), "lon": float(np.rad2deg(arrays["lon_rad"][i])),
            "epoch": float(arrays["time_epoch"][i]), "age": 0.0, "vx": float(arrays["vx"][i]),
            "vy": float(arrays["vy"][i]), "z_raw": float(arrays["altitude_ft"][i]),
            "vr_raw": float(arrays["vertical_rate_fpm"][i]), "fl": float(arrays["flight_level"][i]),
            "code": int(arrays["entity_code"][i])}


def focal_from_event(ev: dict, arrays: dict) -> dict:
    speed = float(ev["matched_ground_speed_kt"])
    track = math.radians(float(ev["matched_track_deg"]))
    return {"lat": float(ev["matched_latitude"]), "lon": float(ev["matched_longitude"]),
            "epoch": float(ev["clearance_epoch"]),
            "age": float(ev["clearance_epoch"] - ev["matched_state_epoch"]),
            "vx": speed * math.sin(track) / 3600.0, "vy": speed * math.cos(track) / 3600.0,
            "z_raw": float(ev["matched_altitude_ft"]), "vr_raw": float(ev["matched_vertical_rate_fpm"]),
            "fl": float(ev["matched_flight_level"]), "code": arrays["code_of"].get(str(ev["entity"]), -1)}


def run_query(query: dict, arrays: dict, times: np.ndarray, rules: dict, grid_s: np.ndarray,
              windows: dict, cfg: dict, post_target: float | None) -> dict:
    """Features of one query (focal aircraft at one instant)."""
    out: dict = {"valid_query": False, "n_nb_50": 0, "n_nb_100": 0, "n_nb_used": 0}
    needed = [query["vx"], query["vy"], query["z_raw"], query["lat"], query["lon"]]
    if not all(map(math.isfinite, needed)):
        return out
    out["valid_query"] = True
    start = int(np.searchsorted(times, query["epoch"] - MAX_AGE_S, side="left"))
    end = int(np.searchsorted(times, query["epoch"], side="right"))
    nb = gather_neighbours(arrays, query, start, end, cfg["radius_nm"])
    if nb is None:
        base = summarise(np.zeros((0, len(grid_s) - 1), dtype=np.int8),
                         np.zeros((0, len(grid_s) - 1)), windows, rules["step"])
        out.update(base)
        if post_target is not None:
            out.update(summarise(np.zeros((0, len(grid_s) - 1), dtype=np.int8),
                                 np.zeros((0, len(grid_s) - 1)), windows, rules["step"], "post_"))
        return out
    distance = np.hypot(nb["dx"], nb["dy"])
    out["n_nb_50"] = int(np.count_nonzero(distance <= 50))
    out["n_nb_100"] = int(np.count_nonzero(distance <= 100))
    out["n_nb_used"] = int(len(distance))
    focal = dict(query)
    focal["z"] = query["z_raw"] + float(deadbanded(query["vr_raw"], rules["deadband"])) * query["age"] / 60.0
    focal["vr"] = 0.0 if cfg.get("vertical_model") == "level" else \
        float(deadbanded(query["vr_raw"], rules["deadband"]))
    res = evaluate_pairs(nb["dx"], nb["dy"], nb["vx"], nb["vy"], nb["z"], nb["vr"], focal, rules,
                         grid_s, windows, post_target, cfg["delay_s"], cfg["nominal_fpm"])
    best = res.pop("rel_idx")
    res["rel_flight_key"] = str(arrays["flight_key"][int(nb["rows"][best])])
    res["rel_initial_distance_nm"] = float(distance[best])
    out.update(res)
    return out


# --------------------------------------------------------------------------- weekly driver
def analyse_week(week: str, week_events: pd.DataFrame, states: pd.DataFrame, rules: dict,
                 cfg: dict) -> pd.DataFrame:
    arrays = build_arrays(states, rules["deadband"], cfg.get("vertical_model", "rate"))
    times = arrays["time_epoch"].astype(float)
    if np.any(np.diff(times) < 0):
        raise ValueError("Surveillance states are not sorted by time")
    per_key = states.groupby("flight_key", sort=False, observed=True).source_member.nunique()
    colliding_keys = set(per_key.index[per_key.gt(1)].astype(str))
    clearance_times = load_clearance_times(week, colliding_keys)
    grid_s = np.arange(HORIZON_S // rules["step"] + 2, dtype=float) * rules["step"]
    windows = make_windows(rules["step"])
    rows = []
    for ev in week_events.to_dict("records"):
        entity = str(ev["flight_key"])
        if entity in colliding_keys:
            entity += "::" + str(ev["source_member"])
        ev["entity"] = entity
        focal = focal_from_event(ev, arrays)
        post_target = None
        if ev["order_type"] == "LEVEL" and np.isfinite(ev["new_cfl"]):
            post_target = focal["z_raw"] + (float(ev["new_cfl"]) - focal["fl"]) * 100.0
        record = {c: ev.get(c) for c in EVENT_COLS}
        record.update(query_kind="event", parent_event_id=ev["event_id"], control_offset_s=0.0,
                      vertical_class=int(vertical_class(focal["vr_raw"], cfg["class_threshold"])))
        record.update(run_query(focal, arrays, times, rules, grid_s, windows, cfg, post_target))
        rows.append(record)
        for i in pick_controls(ev, arrays, clearance_times, cfg["controls"], cfg["seed"],
                               cfg["guard_s"], cfg["class_threshold"]):
            q = focal_from_row(arrays, int(i), rules)
            ctrl = {c: np.nan for c in EVENT_COLS}
            ctrl.update(event_id=f"CONTROL:{ev['event_id']}:{int(i)}", order_type=ev["order_type"],
                        archive_week=week, flight_key=str(arrays["flight_key"][int(i)]),
                        source_member=str(arrays["source_member"][int(i)]),
                        clearance_epoch=float(arrays["time_epoch"][int(i)]),
                        matched_state_epoch=float(arrays["time_epoch"][int(i)]),
                        matched_flight_level=q["fl"], matched_vertical_rate_fpm=q["vr_raw"],
                        matched_altitude_ft=q["z_raw"], direction=ev["direction"])
            ctrl.update(query_kind="control", parent_event_id=ev["event_id"],
                        control_offset_s=float(arrays["time_epoch"][int(i)] - ev["clearance_epoch"]),
                        vertical_class=int(vertical_class(q["vr_raw"], cfg["class_threshold"])))
            ctrl.update(run_query(q, arrays, times, rules, grid_s, windows, cfg, None))
            rows.append(ctrl)
    return pd.DataFrame(rows)


def peak_rss_mib() -> float:
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", default="all", help="Comma-separated archive_week values or 'all'")
    parser.add_argument("--controls-per-event", type=int, default=3)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--radius-nm", type=float, default=150.0,
                        help="Candidate radius. 10 min x 16 NM/min head-on closure ~ 160 NM.")
    parser.add_argument("--clearance-guard-s", type=float, default=120.0)
    parser.add_argument("--class-threshold-fpm", type=float, default=300.0)
    parser.add_argument("--response-delay-s", type=float, default=30.0)
    parser.add_argument("--nominal-rate-fpm", type=float, default=1500.0)
    parser.add_argument("--vertical-model", choices=["rate", "level"], default="rate",
                        help="rate: observed vertical rates held (default, as the detector); "
                             "level: every aircraft keeps its altitude (sensitivity run; needs --out-dir)")
    parser.add_argument("--max-events-per-week", type=int, default=0, help="Smoke test; needs --out-dir")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)
    if (args.max_events_per_week or args.vertical_model != "rate") and Path(args.out_dir) == OUT:
        parser.error("--max-events-per-week / --vertical-model level require a different --out-dir")
    out = Path(args.out_dir)
    for sub in ("tables", "diagnostics"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    cfg = {"radius_nm": args.radius_nm, "controls": args.controls_per_event, "seed": args.seed,
           "guard_s": args.clearance_guard_s, "class_threshold": args.class_threshold_fpm,
           "delay_s": args.response_delay_s, "nominal_fpm": args.nominal_rate_fpm,
           "vertical_model": args.vertical_model}
    start = time.monotonic()
    events, event_audit = load_events()
    all_weeks = sorted(events.archive_week.unique())
    weeks = all_weeks if args.weeks == "all" else [w for w in args.weeks.split(",") if w]
    unknown = set(weeks) - set(all_weeks)
    if unknown:
        raise SystemExit(f"Unknown weeks: {sorted(unknown)}")
    source = pq.ParquetFile(STATES)
    shards, weekly = [], []
    for week in weeks:
        shard, audit_path = out / "tables" / f"queries_{week}.parquet", out / "diagnostics" / f"week_{week}.json"
        if shard.exists() and audit_path.exists() and not args.force:
            print("CACHED", week, flush=True)
            shards.append(shard)
            weekly.append(json.loads(audit_path.read_text()))
            continue
        t0 = time.monotonic()
        states, audit = load_week_states_lean(source, week)
        week_events = events.loc[events.archive_week.eq(week)]
        if args.max_events_per_week:
            week_events = week_events.sample(min(args.max_events_per_week, len(week_events)),
                                             random_state=args.seed).sort_values("clearance_epoch")
        result = analyse_week(week, week_events, states, rules, cfg)
        result.to_parquet(shard, index=False, compression="zstd")
        kinds = result.query_kind.value_counts().to_dict()
        audit.update(events=int(kinds.get("event", 0)), controls=int(kinds.get("control", 0)),
                     invalid_queries=int((~result.valid_query).sum()),
                     elapsed_seconds=round(time.monotonic() - t0, 1), peak_rss_mib=peak_rss_mib())
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        shards.append(shard)
        weekly.append(audit)
        print(week, json.dumps(audit), flush=True)
        del states, result
    queries = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    if queries.event_id.duplicated().any():
        raise ValueError("Duplicated query ids")
    queries.to_parquet(out / "tables" / "predicted_separation_queries.parquet", index=False, compression="zstd")
    run = {"created_utc": datetime.now(timezone.utc).isoformat(), "rules": rules, "config": cfg,
           "weeks": weeks, "event_filter": event_audit, "weekly": weekly,
           "n_queries": int(len(queries)), "n_events": int(queries.query_kind.eq("event").sum()),
           "n_controls": int(queries.query_kind.eq("control").sum()),
           "elapsed_seconds": round(time.monotonic() - start, 1), "peak_rss_mib": peak_rss_mib(),
           "vertical_models": {"A": "observed vertical rate held (dead-band) - primary",
                               "C": "focal follows cleared level - level orders only, prefix post_"}}
    (out / "diagnostics" / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("n_queries", "n_events", "n_controls", "elapsed_seconds", "peak_rss_mib")}))
    if not args.no_report:
        import stage15_report
        stage15_report.main(["--out-dir", str(out)])


if __name__ == "__main__":
    main()
