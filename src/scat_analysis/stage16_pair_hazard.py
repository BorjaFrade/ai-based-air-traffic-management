#!/usr/bin/env python3
"""Stage 16: does the predicted state of a pair anticipate controller action?

Stage 15 compared the moment of an order with control moments. This stage turns
the question around, which is the direction needed to calibrate colour tiers:

    Given the predicted separation of a pair of aircraft at time T, how likely is
    it that the controller issues a level or heading order to either aircraft in
    the next W minutes, compared with pairs that never get close?

Design
------
* Snapshots every ``--snapshot-step-s`` seconds. At each one, every aircraft
  observed in FL290-410 (latest native plot, at most 10 s old) is a member of the
  population; every pair closer than ``--pair-radius-nm`` is evaluated.
* Two predictors of the next 20 minutes, both using only information available
  at T:
    ``R``  the SCAT predicted trajectory (flight-plan route, latest update at or
           before T, at most ``--max-pred-age-s`` old): the aircraft follows the
           polyline of the route fixes ahead of it at its observed ground speed
           and follows the predicted altitude profile (``afl_value``);
    ``C``  constant velocity and vertical rate (as in Stage 15), as comparator.
  Aircraft without a usable route use ``C`` inside ``R``.
* Per pair and predictor: minimum normalised separation m = max(H/5 NM, V/1000 ft)
  over the horizon, the time tau of that minimum, and the worst ENROUTE colour
  state (Stage 15 rule) within each horizon.
* Label: a LEVEL or HEADING order to either aircraft in (T, T+W], W = 2, 5, 10 min.
  The label looks into the future by construction; no feature does.
* Counts are accumulated per cell (m bin x tau bin) and per colour state, per UTC
  day (the cluster for bootstrap intervals), plus a prediction-error validation
  against the positions actually observed later.

No clearance is treated as a conflict label: the target is controller action.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage15_predicted_separation as s15  # noqa: E402

s13 = s15.s13
BASE = s15.BASE
OUT = BASE / "16_pair_hazard"
STATES, CLEARANCES = s15.STATES, s15.CLEARANCES
PREDICTIONS = BASE / "02_traffic/scat_predicted_trajectories.parquet"
EARTH_RADIUS_NM = s15.EARTH_RADIUS_NM
CENTER_LAT, CENTER_LON = 62.0, 16.0
M_EDGES = np.array([0, 1, 1.2, 1.5, 2, 3, 4, 6, 8, 12, np.inf])
TAU_EDGES = np.array([0, 120, 300, 600, 1200.001])
DENS_EDGES = np.array([0, 15, 25, 35, np.inf])  # aircraft in FL290-410 at the snapshot
LABEL_W = (120, 300, 600)
STATE_H = (60, 120, 300, 600, 1200)
VAL_H = (300, 600, 900, 1200)
PREDICTORS = ("R", "C")
MAX_PATH_FIXES = 10
KEY_SCALE = 1e7


# --------------------------------------------------------------------------- geometry
def to_plane(lat_deg, lon_deg):
    """Azimuthal equidistant projection (NM) about the centre of Swedish airspace."""
    phi, lam = np.deg2rad(np.asarray(lat_deg, float)), np.deg2rad(np.asarray(lon_deg, float))
    phi0, lam0 = math.radians(CENTER_LAT), math.radians(CENTER_LON)
    dl = lam - lam0
    cosc = np.clip(math.sin(phi0) * np.sin(phi) + math.cos(phi0) * np.cos(phi) * np.cos(dl), -1, 1)
    c = np.arccos(cosc)
    sinc = np.sin(c)
    k = np.where(c > 1e-9, c / np.where(sinc > 1e-12, sinc, 1.0), 1.0)
    x = EARTH_RADIUS_NM * k * np.cos(phi) * np.sin(dl)
    y = EARTH_RADIUS_NM * k * (math.cos(phi0) * np.sin(phi) - math.sin(phi0) * np.cos(phi) * np.cos(dl))
    return x, y


def plane_direction(lat_deg, lon_deg, track_deg):
    """Unit vector of the track direction in the projection plane."""
    d = 1.0 / 60.0
    lat2 = np.asarray(lat_deg, float) + np.cos(np.deg2rad(track_deg)) * d
    lon2 = np.asarray(lon_deg, float) + np.sin(np.deg2rad(track_deg)) * d / np.cos(np.deg2rad(lat_deg))
    x1, y1 = to_plane(lat_deg, lon_deg)
    x2, y2 = to_plane(lat2, lon2)
    ux, uy = x2 - x1, y2 - y1
    norm = np.hypot(ux, uy)
    norm = np.where(norm > 0, norm, 1.0)
    return ux / norm, uy / norm


def route_positions(x0, y0, z0, ux, uy, speed, px, py, pz, grid, far: float = 2000.0):
    """Positions/altitudes of aircraft following a route polyline at constant ground speed.

    ``px, py, pz`` (n, K) are the route points ahead (NaN padded at the end). The first
    leg starts at the observed position. After the last point the aircraft continues
    straight along the last leg (or its track when the route has no point).
    """
    n, K = px.shape
    ar = np.arange(n)
    X = np.concatenate([x0[:, None], px], 1)
    Y = np.concatenate([y0[:, None], py], 1)
    Z = np.concatenate([z0[:, None], pz], 1)
    real = np.isfinite(X) & np.isfinite(Y)
    last = real.sum(1) - 1
    prev = np.maximum(last - 1, 0)
    dxl, dyl = X[ar, last] - X[ar, prev], Y[ar, last] - Y[ar, prev]
    norm = np.hypot(dxl, dyl)
    ok = norm > 1e-9
    dirx = np.where(ok, dxl / np.where(ok, norm, 1.0), ux)
    diry = np.where(ok, dyl / np.where(ok, norm, 1.0), uy)
    k = np.arange(K + 1)[None, :]
    beyond = k > last[:, None]
    steps = np.clip(k - last[:, None], 0, None)
    X = np.where(beyond, X[ar, last][:, None] + dirx[:, None] * far * steps, X)
    Y = np.where(beyond, Y[ar, last][:, None] + diry[:, None] * far * steps, Y)
    for j in range(1, K + 1):  # forward fill altitude
        Z[:, j] = np.where(np.isfinite(Z[:, j]), Z[:, j], Z[:, j - 1])
    dX, dY, dZ = np.diff(X, axis=1), np.diff(Y, axis=1), np.diff(Z, axis=1)
    length = np.maximum(np.hypot(dX, dY), 1e-9)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(length, axis=1)], 1)
    d = speed[:, None] * grid[None, :]
    seg = np.clip((d[:, :, None] >= cum[:, None, 1:]).sum(-1), 0, K - 1)
    frac = (d - np.take_along_axis(cum, seg, 1)) / np.take_along_axis(length, seg, 1)
    xs = np.take_along_axis(X, seg, 1) + frac * np.take_along_axis(dX, seg, 1)
    ys = np.take_along_axis(Y, seg, 1) + frac * np.take_along_axis(dY, seg, 1)
    zs = np.take_along_axis(Z, seg, 1) + np.clip(frac, 0, 1) * np.take_along_axis(dZ, seg, 1)
    return xs, ys, zs


def constant_velocity(x0, y0, z0, vx, vy, vr, grid):
    return (x0[:, None] + vx[:, None] * grid[None, :], y0[:, None] + vy[:, None] * grid[None, :],
            z0[:, None] + vr[:, None] * grid[None, :] / 60.0)


def pair_features(X, Y, Z, i0, i1, grid, rules, coarse_stride: int, keep_below: float):
    """Minimum normalised separation, its time and the colour states of selected pairs.

    Returns (keep_mask, features) where features cover only the kept pairs. A pair is kept
    when its coarse (every ``coarse_stride`` samples) minimum is below ``keep_below``; all
    pairs left out are certainly NORMAL and far (their m exceeds ``keep_below`` - closure margin).
    """
    cols = np.arange(0, X.shape[1], coarse_stride)
    Hc = np.hypot(X[i0][:, cols] - X[i1][:, cols], Y[i0][:, cols] - Y[i1][:, cols])
    Vc = np.abs(Z[i0][:, cols] - Z[i1][:, cols])
    keep = np.maximum(Hc / rules["red_h"], Vc / rules["red_v"]).min(1) < keep_below
    a, b = i0[keep], i1[keep]
    keys = ["m_min", "tau", "h_now", "v_now", "t_first_red"] + [f"state_{h}" for h in STATE_H]
    if not len(a):
        return keep, {k: np.zeros(0, dtype=np.int64 if k.startswith("state_") else float) for k in keys}
    H = np.hypot(X[a] - X[b], Y[a] - Y[b])
    V = np.abs(Z[a] - Z[b])
    M = np.maximum(H / rules["red_h"], V / rules["red_v"])
    S = s15.classify_states(H, V, rules)
    out = {"m_min": M.min(1), "tau": grid[M.argmin(1)], "h_now": H[:, 0], "v_now": V[:, 0]}
    is_red = S == 3
    out["t_first_red"] = np.where(is_red.any(1), grid[:S.shape[1]][is_red.argmax(1)], np.nan)
    for h in STATE_H:
        out[f"state_{h}"] = S[:, grid[:S.shape[1]] <= h].max(1)
    return keep, out


# --------------------------------------------------------------------------- data
def entity_name(flight_key, source_member, collisions: set[str]):
    key = flight_key.astype(str)
    if not collisions:
        return key.to_numpy()
    member = source_member.astype(str)
    return np.where(key.isin(collisions), key + "::" + member, key).astype(object)


def load_orders(events: pd.DataFrame, week: str, collisions: set[str]) -> dict[str, np.ndarray]:
    """Epochs of the eligible LEVEL and HEADING orders (the Stage 13/14 selection) by aircraft.

    Records that change nothing (same level, metadata updates, aircraft outside FL290-410)
    are not orders for this purpose; counting them would inflate the baseline rate.
    """
    table = events.loc[events.archive_week.eq(week), ["flight_key", "source_member", "clearance_epoch"]].copy()
    table["entity"] = entity_name(table.flight_key, table.source_member, collisions)
    return {e: np.sort(g.clearance_epoch.to_numpy(float)) for e, g in table.groupby("entity", sort=False)}


def order_labels(entity: str, T: float, orders: dict) -> list[bool]:
    epochs = orders.get(entity)
    if epochs is None:
        return [False] * len(LABEL_W)
    lo = np.searchsorted(epochs, T, side="right")
    return [bool(lo < len(epochs) and epochs[lo] <= T + w) for w in LABEL_W]


def load_prediction_index(parquet: pq.ParquetFile, week: str, code_of: dict, collisions: set[str],
                          t0: float) -> pd.DataFrame:
    """Light index (no route) of the predicted-trajectory updates of a week."""
    rows = []
    for g in s13.week_row_groups(parquet, week):
        frame = parquet.read_row_group(g, columns=["archive_week", "flight_key", "source_member",
                                                   "prediction_epoch"]).to_pandas()
        frame["rg"], frame["off"] = g, np.arange(len(frame))
        frame = frame.loc[frame.archive_week.eq(week)]
        frame["entity"] = entity_name(frame.flight_key, frame.source_member, collisions)
        frame["code"] = frame.entity.map(code_of)
        rows.append(frame.dropna(subset=["code", "prediction_epoch"])[["code", "prediction_epoch", "rg", "off"]])
    index = pd.concat(rows, ignore_index=True)
    index["code"] = index.code.astype(np.int64)
    index["key"] = index.code * KEY_SCALE + (index.prediction_epoch - t0)
    return index.sort_values("key", kind="mergesort").reset_index(drop=True)


def latest_prediction(index: pd.DataFrame, codes: np.ndarray, T: np.ndarray, t0: float, max_age: float) -> np.ndarray:
    """Index of the latest update at or before T for each aircraft (-1 when none or stale)."""
    q = codes * KEY_SCALE + (T - t0)
    pos = np.searchsorted(index.key.to_numpy(), q, side="right") - 1
    ok = pos >= 0
    pos_c = np.where(ok, pos, 0)
    ok &= index.code.to_numpy()[pos_c] == codes
    ok &= (T - index.prediction_epoch.to_numpy()[pos_c]) <= max_age
    return np.where(ok, pos, -1)


def parse_route(text: str):
    """Plane coordinates, epoch and altitude (ft) of the fixes of one predicted trajectory."""
    try:
        route = json.loads(text)
    except (TypeError, ValueError):
        return None
    lat = np.array([f.get("lat") if f.get("lat") is not None else np.nan for f in route], float)
    lon = np.array([f.get("lon") if f.get("lon") is not None else np.nan for f in route], float)
    afl = np.array([f.get("afl_value", np.nan) if f.get("afl_unit") == "F" and f.get("afl_value") is not None
                    else np.nan for f in route], float) * 100.0
    try:
        eto = np.array([f["eto"] for f in route], dtype="datetime64[s]").astype(np.int64).astype(float)
    except (KeyError, ValueError):
        return None
    good = np.isfinite(lat) & np.isfinite(lon)
    if not good.any():
        return None
    x, y = to_plane(lat[good], lon[good])
    return x, y, eto[good], afl[good]


def build_path(rec, T: float, x0: float, y0: float, ux: float, uy: float):
    """Fixes still ahead of the aircraft: not yet passed per ETO and not behind its track."""
    x, y, eto, afl = rec
    idx = np.flatnonzero(eto > T - 60.0)
    s = 0
    while s < len(idx):
        dx, dy = x[idx[s]] - x0, y[idx[s]] - y0
        if math.hypot(dx, dy) < 40.0 and dx * ux + dy * uy < 0:
            s += 1
        else:
            break
    sel = idx[s:s + MAX_PATH_FIXES]
    return x[sel], y[sel], afl[sel]


# --------------------------------------------------------------------------- weekly driver
def week_clock(states: pd.DataFrame) -> float:
    return float(states.time_epoch.iloc[0]) - 1.0


def analyse_week(week: str, states: pd.DataFrame, pred_parquet: pq.ParquetFile, rules: dict, args,
                 events: pd.DataFrame) -> dict:
    arrays = s15.build_arrays(states, rules["deadband"])
    times = arrays["time_epoch"].astype(float)
    t0 = week_clock(states)
    lat_deg, lon_deg = states.latitude.to_numpy(float), states.longitude.to_numpy(float)
    track = states.track_deg.to_numpy(float)
    px, py = to_plane(lat_deg, lon_deg)
    ux, uy = plane_direction(lat_deg, lon_deg, track)
    speed = states.ground_speed_kt.to_numpy(float) / 3600.0
    vx, vy = ux * speed, uy * speed
    per_key = states.groupby("flight_key", sort=False, observed=True).source_member.nunique()
    collisions = set(per_key.index[per_key.gt(1)].astype(str))
    orders = load_orders(events, week, collisions)
    index = load_prediction_index(pred_parquet, week, arrays["code_of"], collisions, t0)
    names = {v: k for k, v in arrays["code_of"].items()}

    # ---- snapshots
    snap_times = np.arange(math.ceil(times[0] / args.snapshot_step_s) * args.snapshot_step_s,
                           times[-1] - args.horizon_s, args.snapshot_step_s)
    if args.max_snapshots and len(snap_times) > args.max_snapshots:
        snap_times = snap_times[np.linspace(0, len(snap_times) - 1, args.max_snapshots).astype(int)]
    snap_rows = []
    for T in snap_times:
        a, b = np.searchsorted(times, T - s15.MAX_AGE_S, "left"), np.searchsorted(times, T, "right")
        pos = np.arange(a, b)
        if len(pos) < 2:
            snap_rows.append(np.zeros(0, dtype=np.int64))
            continue
        _, first_rev = np.unique(arrays["entity_code"][pos][::-1], return_index=True)
        rows = pos[-1 - first_rev]
        snap_rows.append(rows[arrays["kin_ok"][rows]])
    counts = np.array([len(r) for r in snap_rows])
    offsets = np.r_[0, np.cumsum(counts)]
    flat_rows = np.concatenate(snap_rows) if offsets[-1] else np.zeros(0, dtype=np.int64)
    flat_T = np.repeat(snap_times, counts)
    flat_code = arrays["entity_code"][flat_rows]

    # ---- predicted trajectories (only the records that are needed)
    pred_pos = latest_prediction(index, flat_code, flat_T, t0, args.max_pred_age_s)
    A = len(flat_rows)
    path_x = np.full((A, MAX_PATH_FIXES), np.nan)
    path_y, path_z = path_x.copy(), path_x.copy()
    has_route = np.zeros(A, dtype=bool)
    need = np.flatnonzero(pred_pos >= 0)
    rec_rg = index.rg.to_numpy()[pred_pos[need]]
    rec_off = index.off.to_numpy()[pred_pos[need]]
    for rg in np.unique(rec_rg):
        mine = need[rec_rg == rg]
        offs = rec_off[rec_rg == rg]
        column = pred_parquet.read_row_group(int(rg), columns=["route_json"]).column(0)
        cache: dict[int, object] = {}
        for a_idx, off in zip(mine, offs):
            rec = cache.get(int(off), 0)
            if isinstance(rec, int) and rec == 0:
                rec = parse_route(column[int(off)].as_py())
                cache[int(off)] = rec
            if rec is None:
                continue
            r = flat_rows[a_idx]
            fx, fy, fz = build_path(rec, flat_T[a_idx], px[r], py[r], ux[r], uy[r])
            if len(fx):
                path_x[a_idx, :len(fx)], path_y[a_idx, :len(fx)], path_z[a_idx, :len(fx)] = fx, fy, fz
                has_route[a_idx] = True
        del column, cache

    # ---- labels
    labels = np.array([order_labels(names[c], T, orders) for c, T in zip(flat_code, flat_T)]) \
        if A else np.zeros((0, len(LABEL_W)), dtype=bool)

    # ---- pair evaluation
    grid = np.arange(0, args.horizon_s + args.grid_s, args.grid_s, dtype=float)
    stride = max(1, int(round(60 / args.grid_s)))
    nM, nT, nD = len(M_EDGES) - 1, len(TAU_EDGES) - 1, len(DENS_EDGES) - 1
    rng = np.random.default_rng(args.seed + int(t0) % 1000)
    cells: dict = {}
    states_tab: dict = {}
    sample_rows, validation = [], []
    for s, T in enumerate(snap_times):
        lo, hi = offsets[s], offsets[s + 1]
        n = hi - lo
        if n < 2:
            continue
        rows = flat_rows[lo:hi]
        day = int(T // 86400)
        x0, y0, z0 = px[rows], py[rows], arrays["altitude_ft"][rows].astype(float)
        XC, YC, ZC = constant_velocity(x0, y0, z0, vx[rows], vy[rows], arrays["vr_eff"][rows], grid)
        XR, YR, ZR = route_positions(x0, y0, z0, ux[rows], uy[rows], speed[rows], path_x[lo:hi], path_y[lo:hi],
                                     path_z[lo:hi], grid)
        ok = has_route[lo:hi]
        XR, YR, ZR = np.where(ok[:, None], XR, XC), np.where(ok[:, None], YR, YC), np.where(ok[:, None], ZR, ZC)
        i0, i1 = np.triu_indices(n, 1)
        near = np.hypot(x0[i0] - x0[i1], y0[i0] - y0[i1]) <= args.pair_radius_nm
        i0, i1 = i0[near], i1[near]
        if not len(i0):
            continue
        lab = labels[lo:hi]
        pair_lab = lab[i0] | lab[i1]
        n_pairs = len(i0)
        for p, (X, Y, Z) in zip(PREDICTORS, ((XR, YR, ZR), (XC, YC, ZC))):
            keep, feat = pair_features(X, Y, Z, i0, i1, grid, rules, stride, args.keep_below)
            kept_lab = pair_lab[keep]
            mbin = np.clip(np.digitize(feat["m_min"], M_EDGES) - 1, 0, nM - 1)
            tbin = np.clip(np.digitize(feat["tau"], TAU_EDGES) - 1, 0, nT - 1)
            key = (day, p)
            dbin = int(np.digitize(n, DENS_EDGES) - 1)
            dd = np.full(len(mbin), dbin)
            tab = cells.setdefault(key, np.zeros((nD, nM, nT + 1, 1 + len(LABEL_W))))
            np.add.at(tab, (dd, mbin, tbin, 0), 1)
            for j in range(len(LABEL_W)):
                np.add.at(tab, (dd, mbin, tbin, 1 + j), kept_lab[:, j])
            far = n_pairs - int(keep.sum())
            tab[dbin, nM - 1, nT, 0] += far
            tab[dbin, nM - 1, nT, 1:] += pair_lab.sum(0) - kept_lab.sum(0)
            st = states_tab.setdefault(key, np.zeros((nD, len(STATE_H), 4, 1 + len(LABEL_W))))
            for hi_, h in enumerate(STATE_H):
                sh = feat[f"state_{h}"]
                np.add.at(st, (dd, np.full(len(sh), hi_), sh, 0), 1)
                for j in range(len(LABEL_W)):
                    np.add.at(st, (dd, np.full(len(sh), hi_), sh, 1 + j), kept_lab[:, j])
                st[dbin, hi_, 0, 0] += far
                st[dbin, hi_, 0, 1:] += pair_lab.sum(0) - kept_lab.sum(0)
            if p == "R":
                take = (feat["m_min"] < args.sample_below) | (rng.random(len(feat["m_min"])) < args.sample_rate)
                if take.any():
                    ia, ib = i0[keep][take], i1[keep][take]
                    frame = pd.DataFrame({k: v[take] for k, v in feat.items()})
                    frame["week"], frame["T"] = week, T
                    frame["entity_a"] = [names[c] for c in arrays["entity_code"][rows[ia]]]
                    frame["entity_b"] = [names[c] for c in arrays["entity_code"][rows[ib]]]
                    frame["route_a"], frame["route_b"] = ok[ia], ok[ib]
                    frame["weight"] = np.where(frame.m_min < args.sample_below, 1.0, 1.0 / args.sample_rate)
                    for j, w in enumerate(LABEL_W):
                        frame[f"label_{w}"] = kept_lab[take][:, j]
                    sample_rows.append(frame)
        # ---- prediction-error validation on a few aircraft with a route
        cand = np.flatnonzero(ok)
        for c in rng.choice(cand, size=min(args.val_per_snapshot, len(cand)), replace=False) if len(cand) else []:
            code = int(arrays["entity_code"][rows[c]])
            erows = arrays["entity_rows"][code]
            et = times[erows]
            for h in VAL_H:
                k = np.searchsorted(et, T + h)
                for kk in (k - 1, k):
                    if 0 <= kk < len(et) and abs(et[kk] - (T + h)) <= 10:
                        r = erows[kk]
                        gi = int(round(h / args.grid_s))
                        validation.append(dict(week=week, horizon_s=h,
                                               err_R=math.hypot(XR[c, gi] - px[r], YR[c, gi] - py[r]),
                                               err_C=math.hypot(XC[c, gi] - px[r], YC[c, gi] - py[r]),
                                               verr_R=abs(ZR[c, gi] - arrays["altitude_ft"][r]),
                                               verr_C=abs(ZC[c, gi] - arrays["altitude_ft"][r])))
                        break
    return dict(cells=cells, states=states_tab, sample=sample_rows, validation=validation,
                audit=dict(snapshots=int((counts >= 2).sum()), aircraft_snapshots=int(A),
                           with_route=float(has_route.mean()) if A else float("nan"),
                           with_prediction_record=float((pred_pos >= 0).mean()) if A else float("nan")))


def long_cells(cells: dict, week: str) -> pd.DataFrame:
    rows = []
    for (day, p), tab in cells.items():
        for d, m, t in zip(*np.nonzero(tab[..., 0])):
            rows.append(dict(week=week, day=day, predictor=p, dens_bin=int(d), m_bin=int(m), tau_bin=int(t),
                             n_pairs=tab[d, m, t, 0], **{f"n_lab_{w}": tab[d, m, t, 1 + j] for j, w in enumerate(LABEL_W)}))
    return pd.DataFrame(rows)


def long_states(states_tab: dict, week: str) -> pd.DataFrame:
    rows = []
    for (day, p), tab in states_tab.items():
        for d, h, st in zip(*np.nonzero(tab[..., 0])):
            rows.append(dict(week=week, day=day, predictor=p, dens_bin=int(d), horizon_s=STATE_H[h], state=int(st),
                             n_pairs=tab[d, h, st, 0], **{f"n_lab_{w}": tab[d, h, st, 1 + j] for j, w in enumerate(LABEL_W)}))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", default="all")
    parser.add_argument("--snapshot-step-s", type=float, default=120.0)
    parser.add_argument("--horizon-s", type=float, default=1200.0)
    parser.add_argument("--grid-s", type=float, default=10.0)
    parser.add_argument("--pair-radius-nm", type=float, default=350.0)
    parser.add_argument("--keep-below", type=float, default=12.0)
    parser.add_argument("--max-pred-age-s", type=float, default=900.0)
    parser.add_argument("--sample-below", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument("--val-per-snapshot", type=int, default=3)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--max-snapshots", type=int, default=0, help="Smoke test; needs --out-dir")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.max_snapshots and Path(args.out_dir) == OUT:
        parser.error("--max-snapshots requires --out-dir different from the production folder")
    out = Path(args.out_dir)
    for sub in ("tables", "diagnostics", "reports", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rules = s15.load_rules()
    states_pf, pred_pf = pq.ParquetFile(STATES), pq.ParquetFile(PREDICTIONS)
    events, _ = s15.load_events()
    all_weeks = sorted(events.archive_week.unique())
    weeks = all_weeks if args.weeks == "all" else args.weeks.split(",")
    start = time.monotonic()
    for week in weeks:
        done = out / "diagnostics" / f"week_{week}.json"
        if done.exists() and not args.force:
            print("CACHED", week, flush=True)
            continue
        t0 = time.monotonic()
        states, audit = s15.load_week_states_lean(states_pf, week)
        result = analyse_week(week, states, pred_pf, rules, args, events)
        long_cells(result["cells"], week).to_parquet(out / "tables" / f"cells_{week}.parquet", index=False)
        long_states(result["states"], week).to_parquet(out / "tables" / f"colour_states_{week}.parquet", index=False)
        pd.DataFrame(result["validation"]).to_parquet(out / "tables" / f"validation_{week}.parquet", index=False)
        sample = pd.concat(result["sample"], ignore_index=True) if result["sample"] else pd.DataFrame()
        sample.to_parquet(out / "tables" / f"pair_sample_{week}.parquet", index=False, compression="zstd")
        audit.update(result["audit"], elapsed_seconds=round(time.monotonic() - t0, 1),
                     peak_rss_mib=s15.peak_rss_mib(), pairs_sample_rows=int(len(sample)))
        done.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(week, json.dumps(audit), flush=True)
        del states, result
    (out / "diagnostics" / "run.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "weeks": weeks, "args": {k: str(v) for k, v in vars(args).items()},
        "m_edges": [str(x) for x in M_EDGES], "tau_edges": TAU_EDGES.tolist(), "label_windows_s": LABEL_W,
        "state_horizons_s": STATE_H, "density_edges": [str(x) for x in DENS_EDGES], "elapsed_seconds": round(time.monotonic() - start, 1)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
