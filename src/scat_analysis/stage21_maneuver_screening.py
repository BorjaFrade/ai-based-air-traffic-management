#!/usr/bin/env python3
"""Retrospective Stage 21 validation of Stage 20 non-green encounters.

Observed geometry uses synchronized native surveillance points. The inertial
projection uses only states available before an order. Neither an association
nor the projection identifies the cause of a maneuver or a resolved conflict.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import stage13_level_change_geometry as s13

EARTH_NM = 3440.065
COLORS = ("rojo", "naranja", "amarillo")
RULES = {"horizon_min": 20, "step_sec": 10, "max_track_gap_sec": 30,
         "min_coverage": 0.80, "heading_threshold_deg": 10.0,
         "heading_target_tolerance_deg": 30.0, "level_threshold_ft": 300.0,
         "pre_state_max_age_sec": 15.0, "nearby_order_sec": 300,
         "full_coverage_required_for_absence": True}


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage20", type=Path, default=Path("results/stage20"))
    p.add_argument("--tracks", type=Path, default=Path("data/processed/scat/02_traffic/scat_states.parquet"))
    p.add_argument("--clearances", type=Path, default=Path("data/processed/scat/03_clearances/scat_clearances.parquet"))
    p.add_argument("--out", type=Path, default=Path("results/stage21"))
    p.add_argument("--alt-unit", choices=["ft", "m", "fl"], required=True)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--pilot-week", default="20161015_20161021")
    p.add_argument("--pilot-per-stratum", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pilot-summary", type=Path)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def entity(week, flight, member):
    return f"{week}\x1f{flight}\x1f{member}"


def great_circle_nm(lat1, lon1, lat2, lon2):
    a1, a2 = np.radians(lat1), np.radians(lat2)
    da, dl = a2 - a1, np.radians(lon2 - lon1)
    q = np.sin(da / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_NM * np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def interpolate(track, times, max_gap):
    q = np.asarray(times, dtype=float)
    out = np.full((len(q), 3), np.nan)
    if track is None or len(track) == 0:
        return out
    t = track[:, 0]
    j = np.searchsorted(t, q, side="left")
    exact = (j < len(t)) & (t[np.minimum(j, len(t) - 1)] == q)
    if exact.any():
        out[exact] = track[j[exact], 1:4]
    use = ~exact & (j > 0) & (j < len(t))
    ii = np.flatnonzero(use)
    if len(ii):
        left, right = j[ii] - 1, j[ii]
        gaps = t[right] - t[left]
        good = (gaps > 0) & (gaps <= max_gap)
        ii, left, right, gaps = ii[good], left[good], right[good], gaps[good]
        f = ((q[ii] - t[left]) / gaps)[:, None]
        out[ii] = track[left, 1:4] + f * (track[right, 1:4] - track[left, 1:4])
    return out


def observed_pair(a, b, start, rules):
    sec = np.arange(0, rules["horizon_min"] * 60 + 1, rules["step_sec"], dtype=float)
    aa = interpolate(a, start + sec, rules["max_track_gap_sec"])
    bb = interpolate(b, start + sec, rules["max_track_gap_sec"])
    valid = np.isfinite(aa).all(1) & np.isfinite(bb).all(1)
    h = np.full(len(sec), np.nan)
    v = np.full(len(sec), np.nan)
    h[valid] = great_circle_nm(aa[valid, 0], aa[valid, 1], bb[valid, 0], bb[valid, 1])
    v[valid] = np.abs(aa[valid, 2] - bb[valid, 2])
    m = np.maximum(h / 5.0, v / 1000.0)
    coverage = float(valid.mean())
    evaluable = bool(valid[0] and valid[-1] and coverage >= rules["min_coverage"])
    result = {"coverage": coverage, "valid_grid_points": int(valid.sum()),
              "total_grid_points": len(sec), "start_observed": bool(valid[0]),
              "end_observed": bool(valid[-1]), "full_coverage": bool(valid.all()),
              "observed_evaluable": evaluable,
              "m_min_observed": np.nan, "observed_min_time_sec": np.nan,
              "h_at_observed_min_nm": np.nan, "v_at_observed_min_ft": np.nan,
              "observed_below_1": False}
    if evaluable:
        k = int(np.nanargmin(m))
        result.update(m_min_observed=float(m[k]), observed_min_time_sec=float(sec[k]),
                      h_at_observed_min_nm=float(h[k]), v_at_observed_min_ft=float(v[k]),
                      observed_below_1=bool(m[k] < 1))
    return result, (sec, m)


def local_xy(lat, lon, lat0, lon0):
    return np.array([(lon - lon0) * 60 * math.cos(math.radians(lat0)),
                     (lat - lat0) * 60], dtype=float)


def pre_state(track, order_t, rules):
    if track is None or len(track) < 2:
        return None
    t = track[:, 0]
    last = np.searchsorted(t, order_t - 5, side="right") - 1
    if last < 1 or order_t - t[last] > rules["pre_state_max_age_sec"]:
        return None
    candidates = np.arange(max(0, last - 30), last)
    age = t[last] - t[candidates]
    candidates = candidates[(age >= 25) & (age <= 90)]
    if not len(candidates):
        return None
    prior = candidates[np.argmin(np.abs(t[last] - t[candidates] - 45))]
    dt = t[last] - t[prior]
    lat0 = (track[prior, 1] + track[last, 1]) / 2
    lon0 = (track[prior, 2] + track[last, 2]) / 2
    delta = local_xy(track[last, 1], track[last, 2], lat0, lon0) - \
            local_xy(track[prior, 1], track[prior, 2], lat0, lon0)
    return {"time": float(t[last]), "lat": float(track[last, 1]),
            "lon": float(track[last, 2]), "alt": float(track[last, 3]),
            "vx": float(delta[0] / dt), "vy": float(delta[1] / dt),
            "vz": float((track[last, 3] - track[prior, 3]) / dt),
            "age_sec": float(order_t - t[last])}


def inertial_pair(a, b, order_t, rules):
    sa, sb = pre_state(a, order_t, rules), pre_state(b, order_t, rules)
    observed, curve_obs = observed_pair(a, b, order_t, rules)
    base = {"m_no_maneuver": np.nan, "cf_min_time_sec": np.nan,
            "m_observed_after_order": observed["m_min_observed"],
            "observed_coverage": observed["coverage"],
            "observed_full_coverage": observed["full_coverage"],
            "cf_evaluable": False, "cf_status": "insufficient_pre_order_state_or_tracks",
            "pre_state_age_a_sec": sa["age_sec"] if sa else np.nan,
            "pre_state_age_b_sec": sb["age_sec"] if sb else np.nan}
    if sa is None or sb is None or not observed["observed_evaluable"]:
        return base, None, curve_obs
    lat0, lon0 = (sa["lat"] + sb["lat"]) / 2, (sa["lon"] + sb["lon"]) / 2
    pa = local_xy(sa["lat"], sa["lon"], lat0, lon0)
    pb = local_xy(sb["lat"], sb["lon"], lat0, lon0)
    sec = curve_obs[0]
    xa = pa[None, :] + (order_t - sa["time"] + sec)[:, None] * np.array([sa["vx"], sa["vy"]])
    xb = pb[None, :] + (order_t - sb["time"] + sec)[:, None] * np.array([sb["vx"], sb["vy"]])
    za = sa["alt"] + (order_t - sa["time"] + sec) * sa["vz"]
    zb = sb["alt"] + (order_t - sb["time"] + sec) * sb["vz"]
    m = np.maximum(np.linalg.norm(xa - xb, axis=1) / 5, np.abs(za - zb) / 1000)
    k = int(np.argmin(m))
    base.update(m_no_maneuver=float(m[k]), cf_min_time_sec=float(sec[k]),
                cf_evaluable=True, cf_status="screening_only")
    return base, (sec, m), curve_obs


def bearing(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


def maneuver(track, event, rules):
    t = float(event.clearance_epoch)
    sec = np.arange(-90, 301, 10, dtype=float)
    pts = interpolate(track, t + sec, rules["max_track_gap_sec"])
    valid = np.isfinite(pts).all(1)
    coverage = float(valid.mean())
    result = {"maneuver_class": "indeterminate", "maneuver_onset_sec": np.nan,
              "trajectory_coverage": coverage, "heading_change_deg": np.nan,
              "altitude_change_ft": np.nan, "target_value": np.nan,
              "plan_context_available": bool(isinstance(event.fpl_route, str) and event.fpl_route.strip()),
              "plan_could_explain": "unresolved"}
    if coverage < rules["min_coverage"] or not valid[(sec >= -90) & (sec <= -10)].all():
        return result
    kind = str(event.clearance_type)
    if "HEADING" in kind and np.isfinite(event.clearance_heading_deg):
        target = float(event.clearance_heading_deg)
        bearings = np.full(len(sec), np.nan)
        for i in range(len(sec) - 3):
            if valid[i] and valid[i + 3]:
                bearings[i] = bearing(*pts[i, :2], *pts[i + 3, :2])
        pre = bearings[(sec >= -80) & (sec <= -40)]
        post = bearings[(sec >= 60) & (sec <= 240)]
        if not np.isfinite(pre).any() or not np.isfinite(post).any():
            return result
        pre_b = float(np.angle(np.mean(np.exp(1j * np.radians(pre[np.isfinite(pre)]))), deg=True) % 360)
        post_b = float(np.angle(np.mean(np.exp(1j * np.radians(post[np.isfinite(post)]))), deg=True) % 360)
        change = abs(angle_delta(post_b, pre_b))
        result.update(target_value=target, heading_change_deg=change)
        hit = (sec >= 0) & (sec <= 270) & np.isfinite(bearings) & \
              (np.abs(angle_delta(bearings, pre_b)) >= rules["heading_threshold_deg"])
        # Each bearing spans 30 seconds; report its midpoint as onset proxy.
        onset = float(sec[np.flatnonzero(hit)[0]] + 15) if hit.any() else np.nan
        result["maneuver_onset_sec"] = onset
        before_error = abs(angle_delta(pre_b, target))
        after_error = abs(angle_delta(post_b, target))
        if np.isfinite(onset) and before_error > rules["heading_threshold_deg"] and \
           after_error <= rules["heading_target_tolerance_deg"] and after_error < before_error - 10:
            result["maneuver_class"] = "compatible_with_clearance"
        elif np.isfinite(onset):
            result["maneuver_class"] = "change_without_clear_correspondence"
        else:
            result["maneuver_class"] = "no_observable_execution"
    elif "LEVEL" in kind and np.isfinite(event.clearance_cfl):
        unit = str(event.clearance_cfl_unit)
        if unit not in {"F", "A"}:
            return result
        target = float(event.clearance_cfl) * (100 if unit == "F" else 1)
        pre_values = pts[(sec >= -30) & (sec <= -10), 2]
        post_values = pts[(sec >= 240) & (sec <= 300), 2]
        if not np.isfinite(pre_values).any() or not np.isfinite(post_values).any():
            return result
        pre_alt = float(np.nanmedian(pre_values))
        post_alt = float(np.nanmedian(post_values))
        if not np.isfinite(pre_alt) or not np.isfinite(post_alt):
            return result
        direction = np.sign(target - pre_alt)
        delta = post_alt - pre_alt
        result.update(target_value=target, altitude_change_ft=delta)
        if result["plan_context_available"] and re.search(rf"F{int(round(target / 100)):03d}\b", event.fpl_route):
            result["plan_could_explain"] = "planned_level_mentioned_not_time_resolved"
        rate = np.full(len(sec), np.nan)
        rate[1:] = np.diff(pts[:, 2]) * 6  # feet per minute on a 10-second grid
        pre_rate = float(np.nanmedian(rate[(sec >= -70) & (sec <= -10)]))
        hit = (sec >= 0) & (sec <= 270) & (direction * rate >= 300)
        onset = float(sec[np.flatnonzero(hit)[0]]) if hit.any() else np.nan
        result["maneuver_onset_sec"] = onset
        if abs(target - pre_alt) < rules["level_threshold_ft"]:
            result["maneuver_class"] = "indeterminate"
        elif direction * pre_rate >= 300:
            result["maneuver_class"] = "change_without_clear_correspondence"
        elif np.isfinite(onset) and direction * delta >= rules["level_threshold_ft"]:
            result["maneuver_class"] = "compatible_with_clearance"
        elif np.isfinite(onset):
            result["maneuver_class"] = "change_without_clear_correspondence"
        else:
            result["maneuver_class"] = "no_observable_execution"
    return result


def index_events(frame):
    groups = {}
    for key, g in frame.groupby("entity", sort=False):
        g = g.sort_values("clearance_epoch")
        groups[key] = g
    return groups


def select_links(ep, observation_times, eligible):
    start, end = pd.Timestamp(ep.start_utc).timestamp(), pd.Timestamp(ep.end_utc).timestamp()
    links = {}
    older_eligible = False
    for key in (ep.entity1, ep.entity2):
        g = eligible.get(key)
        if g is None:
            continue
        for row in g.itertuples(index=False):
            t = float(row.clearance_epoch)
            if start - 1800 <= t < start - 600:
                older_eligible = True
            prior = 0 < start - t <= 600
            if t <= start or t > end + 300:
                after = False
            else:
                j = np.searchsorted(observation_times, t, side="left") - 1
                after = j >= 0 and 0 < t - observation_times[j] <= 300
            if prior or after:
                links[int(row.event_num)] = (row, "prior" if prior else "after")
    return list(links.values()), older_eligible


def load_tracks_week(parquet, week, keys, alt_unit):
    flights = {k.split("\x1f")[1] for k in keys}
    cols = ["archive_week", "flight_key", "source_member", "time_epoch",
            "latitude", "longitude", "altitude_ft"]
    parts = defaultdict(list)
    scanned = selected = 0
    for rg in s13.week_row_groups(parquet, week):
        raw = parquet.read_row_group(rg, columns=cols).to_pandas()
        scanned += len(raw)
        r = raw.loc[raw.archive_week.eq(week) & raw.flight_key.isin(flights)].copy()
        if r.empty:
            continue
        r["entity"] = [entity(week, f, m) for f, m in zip(r.flight_key, r.source_member)]
        r = r.loc[r.entity.isin(keys)]
        selected += len(r)
        for key, group in r.groupby("entity", sort=False):
            arr = group[["time_epoch", "latitude", "longitude", "altitude_ft"]].apply(
                pd.to_numeric, errors="coerce").to_numpy(float)
            if alt_unit == "m":
                arr[:, 3] /= 0.3048
            elif alt_unit == "fl":
                arr[:, 3] *= 100
            ok = np.isfinite(arr).all(1) & (np.abs(arr[:, 1]) <= 90) & \
                 (np.abs(arr[:, 2]) <= 180) & (arr[:, 3] >= 0) & (arr[:, 3] <= 60000)
            if ok.any():
                parts[key].append(arr[ok])
    tracks = {}
    for key, chunks in parts.items():
        arr = np.concatenate(chunks)
        arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
        tracks[key] = arr[np.r_[True, np.diff(arr[:, 0]) > 0]]
    return tracks, {"rows_scanned": scanned, "rows_selected": selected,
                    "tracks_loaded": len(tracks)}


def sample_pilot(episodes, week, n, seed):
    e = episodes.loc[episodes.week.eq(week) & episodes.split.eq("calibration")].copy()
    e["order_group"] = np.select([e.prior_to_episode_start & e.any_next_clearance,
                                   e.prior_to_episode_start, e.any_next_clearance],
                                  ["both", "prior_only", "after_only"], default="none")
    groups = [("rojo", x) for x in ("none", "prior_only", "after_only", "both")] + \
             [("naranja", "any"), ("amarillo", "any")]
    selected = []
    for i, (color, order) in enumerate(groups):
        subset = e.loc[e.highest_color.eq(color)]
        if order != "any":
            subset = subset.loc[subset.order_group.eq(order)]
        if len(subset) < n:
            raise ValueError(f"Pilot stratum {(color, order)} has only {len(subset)} episodes")
        selected.append(subset.sample(n=n, random_state=seed + i))
    return pd.concat(selected).sort_values("episode_id").drop(columns="order_group")


def load_clearances(stage20, source):
    attr = pd.read_csv(stage20 / "stage20_clearance_attribution.csv.gz",
                       usecols=["event_num", "clearance_id", "week", "flight", "source_member"])
    cols = ["clearance_id", "archive_week", "flight_key", "source_member", "clearance_epoch",
            "clearance_type", "clearance_heading_deg", "clearance_cfl", "clearance_cfl_unit",
            "fpl_route"]
    raw = pd.read_parquet(source, columns=cols)
    raw["entity"] = [entity(w, f, m) for w, f, m in zip(
        raw.archive_week, raw.flight_key, raw.source_member)]
    selected = attr.merge(raw, on="clearance_id", validate="one_to_one")
    if len(selected) != len(attr) or \
       (selected.week != selected.archive_week).any() or \
       (selected.flight != selected.flight_key).any() or \
       (selected.source_member_x != selected.source_member_y).any():
        raise ValueError("Stage 20 and Stage 03 clearance identities do not match")
    selected = selected.rename(columns={"source_member_y": "source_member"})
    all_events = raw[["entity", "clearance_epoch", "clearance_type", "clearance_id"]].copy()
    return selected, all_events


def build_figures(cases, out):
    if not cases:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, data in cases.items():
        sec, observed, predicted, cf, observed_after_order, order_sec, onset_sec, episode_id = data
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sec / 60, observed, color="#174a7e", label="Observed synchronized separation")
        ax.scatter([predicted[1] / 60], [predicted[0]], color="#d13c33", zorder=4,
                   label="Stage 16 predicted minimum")
        if cf is not None:
            ax.plot((cf[0] + order_sec) / 60, cf[1], color="#d18c1f", alpha=.8,
                    label="Pre-order inertial projection")
        if observed_after_order is not None:
            ax.plot((observed_after_order[0] + order_sec) / 60,
                    observed_after_order[1], color="#174a7e", ls="--", alpha=.75,
                    label="Observed after clearance")
        if np.isfinite(order_sec):
            ax.axvline(order_sec / 60, color="#7b3894", ls="--", label="Clearance")
        if np.isfinite(onset_sec):
            ax.axvline(onset_sec / 60, color="#2b7c57", ls=":", label="Maneuver onset")
        ax.axhline(1, color="black", lw=.8, alpha=.6)
        values = [observed, np.array([predicted[0]])]
        if cf is not None:
            values.append(cf[1])
        if observed_after_order is not None:
            values.append(observed_after_order[1])
        if max(float(np.nanmax(v)) for v in values if np.isfinite(v).any()) > 6:
            ax.set_ylim(0, 5)
            ax.text(.99, .02, "m > 5 outside plotted range", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8)
        ax.set(xlabel="Minutes from representative pair snapshot", ylabel="Normalized separation m",
               title=f"{group}: episode {episode_id}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = figdir / f"{group}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def markdown_table(frame):
    names = list(frame.columns)
    rows = ["| " + " | ".join(names) + " |",
            "| " + " | ".join("---" for _ in names) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join("" if pd.isna(x) else
                    f"{x:.3f}" if isinstance(x, float) else str(x)
                    for x in values) + " |")
    return "\n".join(rows)


def validation_report(summary, encounters, maneuvers, counterfactuals, bycolor):
    if summary["pilot"]:
        return "\n".join([
            "# SCAT Stage 21 calibration pilot", "",
            f"Sampled episodes: {len(encounters):,}; synchronized observed minima: "
            f"{summary['evaluable']:,}; excluded: {summary['excluded']:,}.",
            f"Stage 20 order-category disagreements: "
            f"{summary['order_category_disagreements_with_stage20']}.",
            "", "## Observed geometry", "", markdown_table(bycolor), "",
            "This stratified sample checks data linkage, interpolation and figure quality; "
            "it is not a population estimate. The calibration uncertainty threshold is "
            "computed only in the full run when the reference sample is large enough.",
            "", "## Figures", "",
            *[f"- `{path}`" for path in summary["figures"]], ""
        ])
    unique_orders = maneuvers.drop_duplicates("clearance_id")
    compatible = unique_orders.loc[unique_orders.maneuver_class.eq("compatible_with_clearance")]
    observed_change = unique_orders.loc[unique_orders.maneuver_class.isin(
        ["compatible_with_clearance", "change_without_clear_correspondence"])]
    red = encounters.loc[encounters.split.eq("test") & encounters.color.eq("rojo")]
    red_none = red.loc[red.order_group.eq("none")]
    red_none_eval = red_none.loc[red_none.observed_evaluable]
    red_none_safe = red_none_eval.loc[red_none_eval.predicted_loss_not_observed]
    no_recorded_order_window = red_none_safe.loc[
        red_none_safe.stage03_orders_prior_30_min.eq(0) &
        red_none_safe.stage03_orders_in_episode_followup.eq(0)]
    screen = counterfactuals.loc[counterfactuals.screening_pattern]
    test_screen = screen.loc[screen.split.eq("test")]
    calibration_null = counterfactuals.loc[
        counterfactuals.split.eq("calibration") &
        counterfactuals.case_type.eq("no_order_reference") &
        counterfactuals.cf_evaluable & counterfactuals.observed_full_coverage &
        counterfactuals.nearby_stage03_orders_5_min.eq(0)]
    onset = compatible.maneuver_onset_sec.quantile([.25, .5, .75])
    lines = ["# SCAT Stage 21 validation", "",
             f"Mode: {'calibration pilot' if summary['pilot'] else 'full 13-week analysis'}.",
             "", "## Design and data quality", "",
             f"Stage 20 supplies {len(encounters):,} non-green episodes; "
             f"{summary['evaluable']:,} ({summary['evaluable'] / len(encounters):.1%}) have "
             "a synchronized observed minimum and "
             f"{summary['excluded']:,} ({summary['excluded'] / len(encounters):.1%}) "
             "are excluded for inadequate future surveillance. "
             f"{int(encounters.full_coverage.sum()):,} have every 10-second grid point. "
             f"Stage 20 order-category disagreements: {summary['order_category_disagreements_with_stage20']}.",
             "", "Observed horizontal and vertical distances are calculated at the same UTC instants. "
             "The 20-minute grid starts at the first snapshot bearing the episode's highest color. "
             "Interpolation spans at most 30 seconds; it never extrapolates. An observed minimum "
             "needs at least 80% coverage and both endpoints. A claim that predicted loss was "
             "not observed requires complete coverage. Altitude is in feet.",
             "", "## Predicted versus observed geometry", "",
             "Here m = max(horizontal distance / 5 NM, vertical distance / 1,000 ft). "
             "The Stage 16 m minimum and tau come from the same anchor snapshot as the following "
             "observed 20-minute window. An observed m below 1 is a geometric screen, not a "
             "confirmed operational loss of separation or incident. Error is observed minus predicted m.",
             "", markdown_table(bycolor), "",
             "The table retains calibration and test separately. Prediction errors are conditional "
             "on evaluable tracks; missing surveillance can bias them. Colors are Stage 19 "
             "prediction classes, not observed outcome classes.",
             "", "## Linked clearances and observed maneuvers", "",
             f"There are {len(maneuvers):,} episode-clearance links but only "
             f"{len(unique_orders):,} distinct linked clearances. Of the distinct clearances, "
             f"{len(compatible):,} ({len(compatible) / len(unique_orders):.1%}) have a trajectory "
             "change compatible with the target heading or level; "
             f"{len(observed_change):,} ({len(observed_change) / len(unique_orders):.1%}) "
             "show some observable change, including changes without clear correspondence; "
             f"{int(unique_orders.maneuver_class.eq('no_observable_execution').sum()):,} "
             "have no observable execution and "
             f"{int(unique_orders.maneuver_class.eq('indeterminate').sum()):,} are indeterminate. "
             "The compatible-onset proxy has 25th, 50th and 75th percentiles of "
             f"{onset.loc[.25]:.0f}, {onset.loc[.5]:.0f} and {onset.loc[.75]:.0f} seconds "
             "after issuance. A heading proxy spans a 30-second bearing window.",
             "", f"{int((unique_orders.nearby_stage03_orders_5_min > 0).sum()):,} distinct "
             "clearances have another Stage 03 order within five minutes; "
             f"{int((unique_orders.episodes_linked_to_same_clearance > 1).sum()):,} "
             "are linked to more than one episode. Maneuver compatibility therefore neither "
             "identifies the conflict an order addressed nor proves its effect. Flight-plan "
             "route strings provide context but no time-resolved planned turn or level change.",
             "", "## Pre-order inertial screening", "",
             f"{summary['counterfactual_evaluable']:,} of "
             f"{len(counterfactuals):,} projection rows have adequate pre-order states and "
             "observed follow-up. The projection holds each aircraft's pre-order horizontal "
             "and vertical velocity constant for 20 minutes. Its minimum and time to minimum "
             "are in `stage21_counterfactuals.csv`, alongside the observed minimum, "
             "surveillance coverage, maneuver class and other-order indicators.",
             "", f"A calibration reference of {len(calibration_null):,} full-coverage no-linked-order "
             "rows without Stage 03 orders within five minutes has 90th-percentile absolute "
             f"projection error {summary['calibration_no_order_null_error_p90_m']:.3f} m units. "
             "This is a sensitivity threshold, not a confidence interval. "
             f"{len(screen):,} compatible episode-clearance rows ({len(test_screen):,} test) "
             "combine projected m below 1 and fully observed m at least 1; "
             f"only {int(screen.difference_exceeds_calibration_p90.sum()):,} "
             f"({int(test_screen.difference_exceeds_calibration_p90.sum()):,} test) "
             "exceed the calibration error threshold. "
             f"{int((screen.nearby_stage03_orders_5_min > 0).sum()):,} have nearby "
             "orders, and "
             f"{int((screen.other_aircraft_orders_next_20_min > 0).sum()):,} have an "
             "order to the other aircraft during follow-up. All test rows exceeding the "
             "threshold have a nearby order. The inertial projection ignores planned "
             "maneuvers and cannot establish a causal resolution.",
             "", "## Test red episodes without a linked order", "",
             f"The {len(red):,} test red episodes divide into "
             f"{int(red.order_group.eq('after_only').sum()):,} after-only, "
             f"{int(red.order_group.eq('both').sum()):,} both, "
             f"{int(red.order_group.eq('prior_only').sum()):,} prior-only and "
             f"{len(red_none):,} no linked order. Among the latter, "
             f"{len(red_none_eval):,} are evaluable and "
             f"{len(red_none) - len(red_none_eval):,} lack adequate follow-up. "
             f"{int(red_none_eval.observed_below_1.sum()):,} evaluable episodes have observed "
             "m below 1; "
             f"{len(red_none_safe):,} have complete coverage with observed m at least 1. "
             f"{int((red_none_eval.prediction_error_observed_minus_predicted.abs() > 1).sum()):,} "
             "evaluable episodes differ by more than one normalized m unit.",
             "", f"Among all {len(red_none):,} no-linked-order red episodes, "
             f"{int(red_none.older_eligible_order_10_to_30_min.sum()):,} have an eligible "
             "order 10–30 minutes before onset and "
             f"{int((red_none.stage03_orders_prior_30_min > 0).sum()):,} have any Stage 03 "
             "order in the preceding 30 minutes. Of the "
             f"{len(red_none_safe):,} fully observed episodes with m at least 1, "
             f"{int((red_none_safe.stage03_orders_prior_30_min > 0).sum()):,} have a Stage 03 "
             "order in that preceding interval and "
             f"{len(no_recorded_order_window):,} have none in the preceding 30 minutes or "
             "episode/five-minute follow-up. Their median prediction discrepancy is "
             f"{no_recorded_order_window.prediction_error_observed_minus_predicted.median():.3f} "
             "m units; only "
             f"{int((no_recorded_order_window.prediction_error_observed_minus_predicted > .1).sum()):,} "
             "exceeds 0.1. These are residual cases without an identified Stage 03 order "
             "in this enlarged window, not proven false alarms: planned maneuvers, other "
             "interventions, and threshold proximity remain unresolved.",
             "", "## Suitability for a scientific article", "",
             "The full-population accounting, calibration/test split, order categories, "
             "track coverage and descriptive synchronized geometry can be reported with "
             "their denominators and censoring. The maneuver classes and inertial screening "
             "are exploratory. Claims of ATC necessity, prevented loss or causal effect "
             "require time-resolved plans, operational separation rules, independent "
             "trajectory review and a stronger validated counterfactual.",
             "", "## Figures", ""]
    lines += [f"- `{path}`" for path in summary["figures"]]
    return "\n".join(lines) + "\n"


def main():
    a = cli()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.inspect:
        for path in (a.tracks, a.clearances):
            pf = pq.ParquetFile(path)
            print(path, "rows", pf.metadata.num_rows, "columns", pf.schema.names)
        return
    if (a.out / "stage21_summary.json").exists() and not a.force:
        raise FileExistsError("Stage 21 output exists; use a new --out or explicit --force")
    stage20 = json.loads((a.stage20 / "stage20_summary.json").read_text())
    if stage20["pairs_read"] != 21_567_488 or stage20["stage16_label_300_disagreements"]:
        raise ValueError("Stage 20 input failed full-population or label validation")
    if not a.pilot:
        if a.pilot_summary is None:
            raise ValueError("Full run requires --pilot-summary with frozen rules")
        pilot = json.loads(a.pilot_summary.read_text())
        if pilot["rules"] != RULES or pilot["alt_unit"] != a.alt_unit:
            raise ValueError("Full-run rules or altitude unit differ from the calibration pilot")
    ep = pd.read_csv(a.stage20 / "stage20_episodes.csv")
    obs = pd.read_csv(a.stage20 / "stage20_non_green_observations.csv.gz", low_memory=False)
    if a.pilot:
        ep = sample_pilot(ep, a.pilot_week, a.pilot_per_stratum, a.seed)
    ep.sort_values(["split", "week", "episode_id"], inplace=True)
    obs = obs.loc[obs.episode_id.isin(set(ep.episode_id))].copy()
    obs_times = {int(i): np.sort(g.time_ns.to_numpy(np.int64) / 1e9)
                 for i, g in obs.groupby("episode_id", sort=False)}
    # Anchor observed geometry to the first snapshot at the episode's highest
    # color. This gives red episodes a red forecast with the same 20-minute
    # horizon as the subsequent observed trajectory.
    severity = {"rojo": 3, "naranja": 2, "amarillo": 1}
    obs["severity"] = obs.color.map(severity)
    representative = (obs.sort_values(["episode_id", "severity", "time_ns"],
                                      ascending=[True, False, True])
                      .drop_duplicates("episode_id").set_index("episode_id"))
    eligible, all_orders = load_clearances(a.stage20, a.clearances)
    pf = pq.ParquetFile(a.tracks)
    encounter_rows, maneuver_rows, cf_rows = [], [], []
    figures = {}
    track_audit = {}
    category_disagreements = 0
    for week in list(stage20["calibration_weeks"]) + list(stage20["test_weeks"]):
        sub = ep.loc[ep.week.eq(week)]
        if sub.empty:
            continue
        keys = set(sub.entity1) | set(sub.entity2)
        tracks, scan = load_tracks_week(pf, week, keys, a.alt_unit)
        track_audit[week] = scan
        eg = index_events(eligible.loc[eligible.week.eq(week) & eligible.entity.isin(keys)])
        ag = index_events(all_orders.loc[all_orders.entity.isin(keys)])
        print(f"{week}: {len(sub)} episodes, {scan}", flush=True)
        for episode in sub.itertuples(index=False):
            start = pd.Timestamp(episode.start_utc).timestamp()
            first = representative.loc[int(episode.episode_id)]
            anchor = float(first.time_ns) / 1e9
            t1, t2 = tracks.get(episode.entity1), tracks.get(episode.entity2)
            observed, curve = observed_pair(t1, t2, anchor, RULES)
            observation_times = obs_times[int(episode.episode_id)]
            links, older_eligible = select_links(episode, observation_times, eg)
            prior = any(rel == "prior" for _, rel in links)
            after = any(rel == "after" for _, rel in links)
            category_disagreements += int(prior != bool(episode.prior_to_episode_start) or
                                          after != bool(episode.any_next_clearance))
            older_any = False
            near_all = 0
            last_any_order_age = np.nan
            all_prior_30 = 0
            all_followup = 0
            for key in (episode.entity1, episode.entity2):
                g = ag.get(key)
                if g is None:
                    continue
                times = g.clearance_epoch.to_numpy(float)
                older_any |= bool(((times >= start - 1800) & (times < start - 600)).any())
                near_all += int(((times >= start - 300) & (times <= start + 300)).sum())
                all_prior_30 += int(((times < start) & (times >= start - 1800)).sum())
                all_followup += int(((times > start) &
                                     (times <= pd.Timestamp(episode.end_utc).timestamp() + 300)).sum())
                before = times[times < start]
                if len(before):
                    age = float(start - before.max())
                    last_any_order_age = min(last_any_order_age, age) if np.isfinite(last_any_order_age) else age
            pred_m, pred_tau = float(first.mmin), float(first.tau_min) * 60
            order_group = ("both" if prior and after else "prior_only" if prior else
                           "after_only" if after else "none")
            row = {"episode_id": int(episode.episode_id), "week": week, "split": episode.split,
                   "color": episode.highest_color, "start_utc": episode.start_utc,
                   "end_utc": episode.end_utc, "flight1": episode.flight1,
                   "flight2": episode.flight2, "anchor_time_utc": first.time_utc,
                   "order_group": order_group,
                   "eligible_orders_linked": len(links), "older_eligible_order_10_to_30_min": older_eligible,
                   "older_any_stage03_order_10_to_30_min": older_any,
                   "any_stage03_orders_near_start_5_min": near_all,
                   "stage03_orders_prior_30_min": all_prior_30,
                   "stage03_orders_in_episode_followup": all_followup,
                   "last_stage03_order_before_start_sec": last_any_order_age,
                   "m_min_predicted_stage16_start": pred_m,
                   "tau_predicted_stage16_start_sec": pred_tau,
                   **observed}
            row["prediction_error_observed_minus_predicted"] = (
                observed["m_min_observed"] - pred_m if observed["observed_evaluable"] else np.nan)
            row["predicted_loss_not_observed"] = bool(
                observed["observed_evaluable"] and observed["full_coverage"] and
                pred_m < 1 and observed["m_min_observed"] >= 1)
            encounter_rows.append(row)
            candidate_cf = None
            candidate_obs_after = None
            candidate_onset = np.nan
            candidate_order_sec = np.nan
            for event, relation in links:
                event_t = float(event.clearance_epoch)
                if not np.isfinite(candidate_order_sec):
                    candidate_order_sec = event_t - anchor
                own = episode.entity1 if event.entity == episode.entity1 else episode.entity2
                other = episode.entity2 if own == episode.entity1 else episode.entity1
                man = maneuver(tracks.get(own), event, RULES)
                nearby = 0
                other_intervention = 0
                for key in (own, other):
                    group = ag.get(key)
                    if group is None:
                        continue
                    times = group.clearance_epoch.to_numpy(float)
                    nearby += int(((np.abs(times - event_t) <= RULES["nearby_order_sec"]) &
                                   (group.clearance_id.to_numpy(str) != event.clearance_id)).sum())
                    if key == other:
                        other_intervention = int(((times > event_t) &
                                                  (times <= event_t + RULES["horizon_min"] * 60)).sum())
                mr = {"episode_id": int(episode.episode_id), "week": week,
                      "split": episode.split, "color": episode.highest_color,
                      "event_num": int(event.event_num), "clearance_id": event.clearance_id,
                      "clearance_flight": event.flight_key, "clearance_type": event.clearance_type,
                      "clearance_epoch": event_t,
                      "clearance_time_utc": pd.Timestamp(event_t, unit="s", tz="UTC").isoformat(),
                      "relative_order": relation, "nearby_stage03_orders_5_min": nearby,
                      "other_aircraft_orders_next_20_min": other_intervention, **man}
                maneuver_rows.append(mr)
                if np.isfinite(man["maneuver_onset_sec"]) and not np.isfinite(candidate_onset):
                    candidate_onset = event_t - anchor + man["maneuver_onset_sec"]
                if man["maneuver_class"] in {"compatible_with_clearance",
                                              "change_without_clear_correspondence"}:
                    cf, cf_curve, cf_obs_curve = inertial_pair(t1, t2, event_t, RULES)
                    cr = {"episode_id": int(episode.episode_id), "week": week,
                          "split": episode.split, "color": episode.highest_color,
                          "clearance_id": event.clearance_id,
                          "case_type": "pre_clearance_inertial",
                          "maneuver_class": man["maneuver_class"],
                          "nearby_stage03_orders_5_min": nearby,
                          "other_aircraft_orders_next_20_min": other_intervention, **cf}
                    cr["inertial_loss_observed_safe"] = bool(
                        cf["cf_evaluable"] and cf["observed_full_coverage"] and
                        cf["m_no_maneuver"] < 1 and cf["m_observed_after_order"] >= 1)
                    cr["screening_pattern"] = bool(
                        cr["inertial_loss_observed_safe"] and
                        man["maneuver_class"] == "compatible_with_clearance")
                    cf_rows.append(cr)
                    if candidate_cf is None and cf_curve is not None:
                        candidate_cf = cf_curve
                        candidate_obs_after = cf_obs_curve
                        candidate_onset = event_t - anchor + man["maneuver_onset_sec"]
                        candidate_order_sec = event_t - anchor
            if order_group == "none":
                cf, _, _ = inertial_pair(t1, t2, anchor, RULES)
                cf_rows.append({"episode_id": int(episode.episode_id), "week": week,
                                "split": episode.split, "color": episode.highest_color,
                                "clearance_id": "", "case_type": "no_order_reference",
                                "maneuver_class": "not_applicable",
                                "nearby_stage03_orders_5_min": near_all,
                                "other_aircraft_orders_next_20_min": np.nan,
                                **cf, "inertial_loss_observed_safe": False,
                                "screening_pattern": False})
            fig_group = f"{episode.highest_color}_{order_group}"
            if episode.split == "test" and episode.highest_color == "rojo" and \
               order_group == "none" and row["predicted_loss_not_observed"] and \
               all_prior_30 == 0 and all_followup == 0:
                fig_group = "test_rojo_none_residual"
            selected_figures = {"rojo_none", "rojo_prior_only", "rojo_after_only",
                                "rojo_both", "naranja_none", "naranja_after_only",
                                "amarillo_none", "amarillo_after_only",
                                "test_rojo_none_residual"}
            replace = fig_group in figures and figures[fig_group][3] is None and candidate_cf is not None
            if observed["observed_evaluable"] and fig_group in selected_figures and \
               (replace or fig_group not in figures):
                figures[fig_group] = (curve[0], curve[1], (pred_m, pred_tau), candidate_cf,
                                      candidate_obs_after, candidate_order_sec, candidate_onset,
                                      int(episode.episode_id))
        del tracks
    if category_disagreements:
        raise ValueError(f"{category_disagreements} Stage 20 order-category disagreements")
    encounters = pd.DataFrame(encounter_rows)
    maneuvers = pd.DataFrame(maneuver_rows)
    counterfactuals = pd.DataFrame(cf_rows)
    if not maneuvers.empty:
        multiplicity = maneuvers.groupby("clearance_id").episode_id.nunique()
        maneuvers["episodes_linked_to_same_clearance"] = maneuvers.clearance_id.map(multiplicity)
        maneuvers["ambiguous_pair_attribution"] = maneuvers.episodes_linked_to_same_clearance > 1
    calibration_errors = counterfactuals.loc[
        counterfactuals.split.eq("calibration") &
        counterfactuals.case_type.eq("no_order_reference") &
        counterfactuals.cf_evaluable &
        counterfactuals.observed_full_coverage &
        counterfactuals.nearby_stage03_orders_5_min.eq(0),
        ["m_no_maneuver", "m_observed_after_order"]]
    uncertainty = (float(np.quantile(np.abs(calibration_errors.m_observed_after_order -
                                        calibration_errors.m_no_maneuver), .9))
                   if len(calibration_errors) >= 30 else None)
    if not counterfactuals.empty:
        counterfactuals["calibration_null_error_p90_m"] = uncertainty
        counterfactuals["difference_exceeds_calibration_p90"] = (
            counterfactuals.cf_evaluable & (counterfactuals.m_observed_after_order -
                                             counterfactuals.m_no_maneuver > uncertainty)) if uncertainty is not None else False
    by = []
    for split in ("calibration", "test", "all"):
        for color in COLORS:
            g = encounters.loc[encounters.color.eq(color)]
            if split != "all":
                g = g.loc[g.split.eq(split)]
            ev = g.loc[g.observed_evaluable]
            by.append({"split": split, "color": color, "encounters": len(g),
                       "observed_evaluable": len(ev), "insufficient_tracks": len(g) - len(ev),
                       "full_coverage": int(ev.full_coverage.sum()),
                       "median_coverage": float(g.coverage.median()) if len(g) else np.nan,
                       "observed_below_1": int(ev.observed_below_1.sum()),
                       "predicted_loss_not_observed": int(ev.predicted_loss_not_observed.sum()),
                       "median_observed_minus_predicted_m": float(ev.prediction_error_observed_minus_predicted.median()) if len(ev) else np.nan,
                       "median_abs_prediction_error_m": float(ev.prediction_error_observed_minus_predicted.abs().median()) if len(ev) else np.nan,
                       "no_order": int(g.order_group.eq("none").sum()),
                       "no_order_evaluable": int((g.order_group.eq("none") & g.observed_evaluable).sum()),
                       "no_order_predicted_loss_not_observed": int((g.order_group.eq("none") & g.predicted_loss_not_observed).sum()),
                       "no_order_older_eligible_10_to_30_min": int((g.order_group.eq("none") & g.older_eligible_order_10_to_30_min).sum())})
    bycolor = pd.DataFrame(by)
    encounters.to_csv(a.out / "stage21_encounters.csv", index=False)
    maneuvers.to_csv(a.out / "stage21_clearance_maneuvers.csv", index=False)
    counterfactuals.to_csv(a.out / "stage21_counterfactuals.csv", index=False)
    bycolor.to_csv(a.out / "stage21_by_color.csv", index=False)
    figure_paths = build_figures(figures, a.out)
    summary = {"stage": 21, "pilot": a.pilot, "rules": RULES, "alt_unit": a.alt_unit,
               "pilot_week": a.pilot_week if a.pilot else None,
               "pilot_per_stratum": a.pilot_per_stratum if a.pilot else None,
               "seed": a.seed if a.pilot else None,
               "episodes": len(encounters), "evaluable": int(encounters.observed_evaluable.sum()),
               "excluded": int((~encounters.observed_evaluable).sum()),
               "order_category_disagreements_with_stage20": category_disagreements,
               "maneuver_rows": len(maneuvers),
               "maneuver_classes": maneuvers.maneuver_class.value_counts().to_dict() if len(maneuvers) else {},
               "counterfactual_rows": len(counterfactuals),
               "counterfactual_evaluable": int(counterfactuals.cf_evaluable.sum()) if len(counterfactuals) else 0,
               "calibration_no_order_null_error_p90_m": uncertainty,
               "track_audit": track_audit, "figures": figure_paths,
               "limitations": ["Temporal association and observed maneuver do not prove causality.",
                               "A constant-velocity projection ignores planned turns, level changes and other controls.",
                               "Observed m<1 is a geometric screen, not a confirmed incident.",
                               "Flight-plan route strings identify context but do not time-resolve planned maneuvers."]}
    (a.out / "stage21_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.out / "STAGE21_VALIDATION_REPORT.md").write_text(
        validation_report(summary, encounters, maneuvers, counterfactuals, bycolor))
    print(json.dumps({k: summary[k] for k in ("episodes", "evaluable", "excluded", "maneuver_rows",
                                               "counterfactual_evaluable", "calibration_no_order_null_error_p90_m")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
