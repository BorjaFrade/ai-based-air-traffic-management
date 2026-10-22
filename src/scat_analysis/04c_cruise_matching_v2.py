#!/usr/bin/env python3
"""Context-only, first-week cruise rematching from existing V2 window rows.

This script does not build windows, recompute interaction features or use any
separation, CPA, bearing, closing-rate, neighbour or TopSky field for matching.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK, assert_pilot_frame


BAND = "CRUISE_FL290_410"
WINDOW_SECONDS = 600.0  # Stage 04's clearance-free exclusion interval.
# Small, predeclared grid informed by V1 cruise pair-difference quantiles.
CALIPERS = (
    {"name": "strict", "speed_kt": 10, "altitude_ft": 500, "density": 0, "time_s": 1800},
    {"name": "moderate", "speed_kt": 15, "altitude_ft": 1000, "density": 1, "time_s": 2400},
    {"name": "balanced", "speed_kt": 20, "altitude_ft": 1000, "density": 1, "time_s": 3600},
    {"name": "broad", "speed_kt": 25, "altitude_ft": 1500, "density": 2, "time_s": 3600},
)
NUMERIC_CONTEXT = {
    "ground_speed_at_anchor_kt": ("speed_kt", "ground_speed_at_anchor_kt"),
    "analysis_altitude_ft": ("altitude_ft", "analysis_altitude_ft"),
    "local_traffic_density": ("density", "local_traffic_density"),
}


def smd(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype=float)
    if min(len(a), len(b)) < 2:
        return float("nan")
    scale = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    delta = np.mean(a) - np.mean(b)
    return float(delta / scale) if scale > 0 else (0.0 if abs(delta) < 1e-12 else float("inf"))


def status(value: float) -> str:
    if not np.isfinite(value):
        return "UNDETERMINED"
    if abs(value) < .10:
        return "PASS"
    if abs(value) <= .20:
        return "WARNING"
    return "FAIL"


def audit_clearance_free(controls: pd.DataFrame, clearances_path: Path) -> None:
    source = pd.read_parquet(clearances_path, columns=["archive_week", "flight_key", "clearance_epoch"],
                             filters=[("archive_week", "==", PILOT_WEEK)])
    assert_pilot_frame(source)
    events = {key: np.sort(group.clearance_epoch.dropna().to_numpy(dtype=float))
              for key, group in source.groupby("flight_key", sort=False)}
    for row in controls.itertuples(index=False):
        times = events.get(row.focal_flight_key)
        if times is None or not len(times):
            continue
        idx = int(np.searchsorted(times, row.event_epoch))
        nearest = min((abs(times[j] - row.event_epoch) for j in (idx - 1, idx) if 0 <= j < len(times)), default=float("inf"))
        if nearest <= WINDOW_SECONDS:
            raise SystemExit(f"Control {row.window_id} violates Stage 04 clearance exclusion")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    root = DEFAULT_OUTPUT_ROOT
    cohort = pd.read_parquet(root / "04_windows" / f"primary_matching_cohort_{PILOT_WEEK}.parquet")
    windows = pd.read_parquet(root / "05_features" / f"window_features_v2_{PILOT_WEEK}.parquet")
    assert_pilot_frame(cohort)
    assert_pilot_frame(windows)
    if windows.window_id.duplicated().any() or cohort.control_window_id.duplicated().any():
        raise SystemExit("Pilot window/control IDs are not unique")
    if not cohort.matched.all() or len(cohort) != 10097:
        raise SystemExit("Existing Stage 04 primary matched cohort changed")
    lookup = windows.set_index("window_id")
    v1_mask = cohort.intervention_window_id.map(lookup.altitude_band).eq(BAND) & cohort.control_window_id.map(lookup.altitude_band).eq(BAND)
    v1 = cohort.loc[v1_mask].copy()
    if len(v1) != 5128:
        raise SystemExit("Expected 5,128 V1 same-band cruise pairs")
    controls_meta = cohort.loc[cohort.control_window_id.map(lookup.altitude_band).eq(BAND)].copy()
    i = windows.set_index("window_id").loc[v1.intervention_window_id].reset_index()
    c = windows.set_index("window_id").loc[controls_meta.control_window_id].reset_index()
    i["local_traffic_density"] = v1.intervention_local_traffic_density.to_numpy(dtype=float)
    c["local_traffic_density"] = controls_meta.control_local_traffic_density.to_numpy(dtype=float)
    i["aircraft_type"] = v1.intervention_aircraft_type.fillna("").to_numpy()
    c["aircraft_type"] = controls_meta.control_aircraft_type.fillna("").to_numpy()
    c["source_clearance_id"] = controls_meta.clearance_id.to_numpy()
    for name, frame, label in (("intervention", i, 1), ("control", c, 0)):
        if frame.archive_week.ne(PILOT_WEEK).any() or frame.altitude_band.ne(BAND).any():
            raise SystemExit(f"{name} pool is outside the authorized pilot cruise band")
        if frame.ATC_INTERVENTION.ne(label).any() or frame.altitude_anchor_relative_seconds.gt(0).any() or frame.anchor_relative_seconds.gt(0).any():
            raise SystemExit(f"{name} has a label or pre-event anchor violation")
        if frame.analysis_altitude_ft.lt(29000).any() or frame.analysis_altitude_ft.gt(41000).any():
            raise SystemExit(f"{name} has out-of-band altitude")
        required = ["analysis_altitude_ft", "ground_speed_at_anchor_kt", "local_traffic_density", "event_epoch", "hour_utc", "flight_phase", "centre", "focal_flight_key"]
        if frame[required].isna().any().any():
            raise SystemExit(f"{name} has missing required context")
    audit_clearance_free(c, root / "03_clearances" / "scat_clearances.parquet")
    return v1, i.reset_index(drop=True), c.reset_index(drop=True), len(cohort)


def caliper_match(interventions: pd.DataFrame, controls: pd.DataFrame, config: dict) -> pd.DataFrame:
    groups = {}
    for key, part in controls.groupby(["flight_phase", "centre", "hour_utc"], sort=False):
        indices = part.index.to_numpy(dtype=int)
        groups[key] = indices[np.argsort(controls.event_epoch.to_numpy()[indices], kind="mergesort")]
    ctime = controls.event_epoch.to_numpy(dtype=float)
    cspeed = controls.ground_speed_at_anchor_kt.to_numpy(dtype=float)
    calt = controls.analysis_altitude_ft.to_numpy(dtype=float)
    cdens = controls.local_traffic_density.to_numpy(dtype=float)
    cflight = controls.focal_flight_key.to_numpy(dtype=str)
    ctype = controls.aircraft_type.fillna("").to_numpy(dtype=str)
    rows, cols, costs = [], [], []
    candidate_counts = np.zeros(len(interventions), dtype=int)
    for row_idx, row in enumerate(interventions.itertuples(index=False)):
        key = (row.flight_phase, row.centre, row.hour_utc)
        group = groups.get(key)
        if group is None:
            continue
        times = ctime[group]
        lo = np.searchsorted(times, row.event_epoch - config["time_s"], side="left")
        hi = np.searchsorted(times, row.event_epoch + config["time_s"], side="right")
        candidates = group[lo:hi]
        if not len(candidates):
            continue
        speed_diff = np.abs(cspeed[candidates] - row.ground_speed_at_anchor_kt)
        altitude_diff = np.abs(calt[candidates] - row.analysis_altitude_ft)
        density_diff = np.abs(cdens[candidates] - row.local_traffic_density)
        time_diff = np.abs(ctime[candidates] - row.event_epoch)
        keep = (speed_diff <= config["speed_kt"]) & (altitude_diff <= config["altitude_ft"]) & (density_diff <= config["density"]) & (cflight[candidates] != row.focal_flight_key)
        candidates = candidates[keep]
        if not len(candidates):
            continue
        candidate_counts[row_idx] = len(candidates)
        speed_diff, altitude_diff, density_diff, time_diff = speed_diff[keep], altitude_diff[keep], density_diff[keep], time_diff[keep]
        aircraft_penalty = np.where(ctype[candidates] == str(row.aircraft_type), 0.0, .05)
        cost = 2 * speed_diff / config["speed_kt"] + altitude_diff / config["altitude_ft"]
        if config["density"]:
            cost += density_diff / config["density"]
        cost += .25 * time_diff / config["time_s"] + aircraft_penalty + candidates * 1e-9 + 1e-6
        rows.extend([row_idx] * len(candidates))
        cols.extend(candidates.tolist())
        costs.extend(cost.tolist())
    n_i, n_c = len(interventions), len(controls)
    rows.extend(range(n_i))
    cols.extend((n_c + idx for idx in range(n_i)))
    costs.extend([100.0] * n_i)  # Dummy edges maximize cardinality before minimizing context distance.
    graph = csr_matrix((np.asarray(costs), (rows, cols)), shape=(n_i, n_c + n_i))
    left, right = min_weight_full_bipartite_matching(graph)
    keep = right < n_c
    chosen_i, chosen_c = left[keep], right[keep]
    if len(np.unique(chosen_c)) != len(chosen_c):
        raise SystemExit("Control reused in one-to-one matching")
    matched = pd.DataFrame({"intervention_index": chosen_i, "control_index": chosen_c,
                            "candidate_count": candidate_counts[chosen_i]})
    for label, source, index in (("intervention", interventions, chosen_i), ("control", controls, chosen_c)):
        for field in ("window_id", "focal_flight_key", "event_epoch", "analysis_altitude_ft", "ground_speed_at_anchor_kt", "local_traffic_density", "flight_phase", "centre", "hour_utc", "aircraft_type"):
            matched[f"{label}_{field}"] = source.iloc[index][field].to_numpy()
    matched["control_source_clearance_id"] = controls.iloc[chosen_c].source_clearance_id.to_numpy()
    matched["clearance_id"] = matched.intervention_window_id.str.removeprefix("I:")
    matched["control_time_difference_s"] = np.abs(matched.intervention_event_epoch - matched.control_event_epoch)
    matched["archive_week"] = PILOT_WEEK
    matched["altitude_band"] = BAND
    matched["analysis_scope"] = "FIRST_WEEK_PILOT"
    matched["matching_config"] = config["name"]
    matched["control_clearance_free_verified"] = True
    return matched.sort_values("intervention_window_id").reset_index(drop=True)


def balance_table(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_CONTEXT:
        a, b = matched[f"intervention_{feature}"], matched[f"control_{feature}"]
        value = smd(a, b)
        rows.append({"covariate": feature, "category": "", "intervention_n": len(a), "control_n": len(b),
                     "intervention_value": a.mean(), "control_value": b.mean(), "SMD": value, "status": status(value)})
    for feature in ("flight_phase", "centre", "hour_utc"):
        a = matched[f"intervention_{feature}"].fillna("UNKNOWN").astype(str)
        b = matched[f"control_{feature}"].fillna("UNKNOWN").astype(str)
        for category in sorted(set(a) | set(b)):
            left, right = a.eq(category).astype(int), b.eq(category).astype(int)
            value = smd(left, right)
            rows.append({"covariate": feature, "category": category, "intervention_n": len(left), "control_n": len(right),
                         "intervention_value": left.mean(), "control_value": right.mean(), "SMD": value, "status": status(value)})
    return pd.DataFrame(rows)


def summary_row(matched: pd.DataFrame, balance: pd.DataFrame, config: dict, v1_n: int) -> dict:
    indexed = balance.set_index(["covariate", "category"])
    categorical = balance.loc[balance.covariate.isin(["flight_phase", "centre", "hour_utc"])]
    time = matched.control_time_difference_s
    numeric = {name: float(indexed.loc[(name, ""), "SMD"]) for name in NUMERIC_CONTEXT}
    all_pass = all(status(value) == "PASS" for value in numeric.values()) and categorical.status.eq("PASS").all()
    return {**config, "n_pairs": len(matched), "retention_pct": len(matched) / v1_n * 100,
            "ground_speed_SMD": numeric["ground_speed_at_anchor_kt"], "altitude_SMD": numeric["analysis_altitude_ft"],
            "density_SMD": numeric["local_traffic_density"],
            **{f"{name}_max_abs_SMD": categorical.loc[categorical.covariate.eq(name), "SMD"].abs().max() for name in ("flight_phase", "centre", "hour_utc")},
            "phase_balance": "PASS" if categorical.loc[categorical.covariate.eq("flight_phase"), "status"].eq("PASS").all() else "FAIL",
            "centre_balance": "PASS" if categorical.loc[categorical.covariate.eq("centre"), "status"].eq("PASS").all() else "FAIL",
            "UTC_hour_balance": "PASS" if categorical.loc[categorical.covariate.eq("hour_utc"), "status"].eq("PASS").all() else "FAIL",
            "median_temporal_distance_s": time.median(), "p95_temporal_distance_s": time.quantile(.95),
            "all_context_PASS": bool(all_pass)}


def choose_configuration(sensitivity: pd.DataFrame) -> str:
    good = sensitivity.loc[sensitivity.all_context_PASS]
    if len(good):
        near_best = good.loc[good.n_pairs.ge(.95 * good.n_pairs.max())]
        return str(near_best.iloc[0]["name"])  # Grid order: strict -> broad.
    ordered = sensitivity.assign(max_abs_SMD=sensitivity[["ground_speed_SMD", "altitude_SMD", "density_SMD",
                                                    "phase_max_abs_SMD", "centre_max_abs_SMD", "hour_utc_max_abs_SMD"]].abs().max(axis=1))
    return str(ordered.sort_values(["max_abs_SMD", "n_pairs"], ascending=[True, False]).iloc[0]["name"])


def write_report(path: Path, v1_n: int, sensitivity: pd.DataFrame, chosen: pd.DataFrame, balance: pd.DataFrame) -> None:
    selected = sensitivity.loc[sensitivity.name.eq(chosen.matching_config.iloc[0])].iloc[0]
    prior = pd.read_csv(DEFAULT_OUTPUT_ROOT / "06_exploration" / "altitude_band_matching_summary.csv")
    v1 = prior.loc[prior.band.eq(BAND)].iloc[0]
    lines = ["# CRUISE_MATCHING_V2 — first-week context rematching", "",
             f"- Archive week: `{PILOT_WEEK}`; band: FL290–FL410 inclusive.",
             f"- V1 same-band pairs: **{v1_n:,}**; V2 pairs: **{len(chosen):,}**; retention: **{len(chosen)/v1_n*100:.1f}%**.",
             "- Stage 04 V1 and Stage 05 outputs were preserved. Stage 07 and other weeks were not run.", "",
             "## Method and control eligibility", "",
             "The intervention pool is the 5,128 V1 same-band cruise intervention IDs. The control pool is every existing first-week, cruise-band, primary-cohort control window (5,133 IDs), not new Stage 04 candidates. Their Stage 04 source anchors obey the ±600 s clearance exclusion rule; this pass independently rechecks all candidate control flight/event times against all 80,264 first-week clearance timestamps. Each control is used at most once and never for its own flight.", "",
             "Matching is minimum-cost 1:1 bipartite assignment with calipers and individual unmatched dummy edges. The dummy penalty maximizes retained pairs first; among those, cost favours speed, altitude, local density, temporal proximity and (secondarily) aircraft type. Exact strata are flight phase, centre and UTC hour. No physical interaction feature enters eligibility, cost or configuration selection. This is a reproducible nearest-context equivalent, not outcome-optimized matching.", "",
             "## Predeclared caliper sensitivity", "",
             "| Config | Speed kt | Altitude ft | Density | Time s | Pairs | Retention | Speed SMD | Altitude SMD | Density SMD | Phase / centre / UTC | Median / p95 time s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in sensitivity.itertuples(index=False):
        lines.append(f"| {row.name} | ±{row.speed_kt} | ±{row.altitude_ft} | ±{row.density} | ±{row.time_s} | {row.n_pairs:,} | {row.retention_pct:.1f}% | {row.ground_speed_SMD:+.3f} | {row.altitude_SMD:+.3f} | {row.density_SMD:+.3f} | {row.phase_balance}/{row.centre_balance}/{row.UTC_hour_balance} | {row.median_temporal_distance_s:.0f}/{row.p95_temporal_distance_s:.0f} |")
    lines.extend(["", f"Selected **{selected['name']}**: choose the simplest all-context-PASS grid point within 5% of the best PASS retention; if none passes, choose the smallest worst absolute context SMD and mark the gate failed. The 5% rule controls needless sample loss, not a minimum sample count. The full [sensitivity CSV](cruise_matching_v2_sensitivity.csv) records all metrics.", "",
                  "## V1 versus V2 matching balance", "",
                  "V1 cruise anchor ground-speed SMD was **−0.324**. V2 numeric and category-level balance is in [the CSV](cruise_matching_v2_balance.csv). |SMD| <0.10 = PASS; 0.10–0.20 = WARNING; >0.20 = FAIL. These are matching diagnostics, not conflict definitions.", "",
                  f"- V2 ground-speed SMD: **{selected.ground_speed_SMD:+.3f}** ({status(selected.ground_speed_SMD)}).",
                  f"- V2 altitude SMD: **{selected.altitude_SMD:+.3f}** ({status(selected.altitude_SMD)}).",
                  f"- V2 local-density SMD: **{selected.density_SMD:+.3f}** ({status(selected.density_SMD)}).",
                  f"- Phase / centre / UTC-hour balance: **{selected.phase_balance} / {selected.centre_balance} / {selected.UTC_hour_balance}**.",
                  f"- Median / p95 absolute intervention-control time distance: **{selected.median_temporal_distance_s:.0f} / {selected.p95_temporal_distance_s:.0f} s**.", "",
                  "| Context diagnostic | V1 | V2 |", "|---|---:|---:|",
                  f"| Ground-speed SMD | {v1.ground_speed_at_anchor_kt_SMD:+.3f} | {selected.ground_speed_SMD:+.3f} |",
                  f"| Altitude SMD | {v1.analysis_altitude_ft_SMD:+.3f} | {selected.altitude_SMD:+.3f} |",
                  f"| Local-density SMD | {v1.local_traffic_density_SMD:+.3f} | {selected.density_SMD:+.3f} |",
                  f"| Median time distance, s | {v1.control_time_distance_median_s:.0f} | {selected.median_temporal_distance_s:.0f} |",
                  f"| p95 time distance, s | {v1.control_time_distance_p95_s:.0f} | {selected.p95_temporal_distance_s:.0f} |", "",
                  "The Stage 04 local traffic density is a context covariate; the V2 neighbourhood/nearby-aircraft features are deliberately excluded from matching. Ground speed is horizontal motion; near-zero vertical rate in level cruise is not treated as anomalous. All Stage 05 physical variables remain untouched for the descriptive validation.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = DEFAULT_OUTPUT_ROOT
    out = root / "04_windows"
    v1, interventions, controls, total = load_inputs()
    print(f"Cruise V1={len(v1):,} interventions; eligible pre-existing cruise controls={len(controls):,}; first-week cohort={total:,}", flush=True)
    trials = {}
    summaries = []
    for config in CALIPERS:
        matched = caliper_match(interventions, controls, config)
        balance = balance_table(matched)
        summary = summary_row(matched, balance, config, len(v1))
        trials[config["name"]] = (matched, balance)
        summaries.append(summary)
        print(f"{config['name']}: pairs={len(matched):,}, speed_SMD={summary['ground_speed_SMD']:+.3f}, altitude={summary['altitude_SMD']:+.3f}, density={summary['density_SMD']:+.3f}, PASS={summary['all_context_PASS']}", flush=True)
    sensitivity = pd.DataFrame(summaries)
    choice = choose_configuration(sensitivity)
    chosen, balance = trials[choice]
    if chosen.control_window_id.duplicated().any() or chosen.intervention_window_id.duplicated().any():
        raise SystemExit("V2 does not have unique 1:1 matching IDs")
    if chosen.intervention_focal_flight_key.eq(chosen.control_focal_flight_key).any():
        raise SystemExit("V2 matched an intervention to the same flight")
    if chosen.control_time_difference_s.gt(next(c["time_s"] for c in CALIPERS if c["name"] == choice)).any():
        raise SystemExit("Temporal caliper violated")
    ids = set(chosen.intervention_window_id) | set(chosen.control_window_id)
    if len(ids) != 2 * len(chosen):
        raise SystemExit("V2 matched IDs overlap or are duplicated")
    full_windows = pd.read_parquet(root / "05_features" / f"window_features_v2_{PILOT_WEEK}.parquet")
    subset = full_windows.loc[full_windows.window_id.isin(ids)].copy()
    if len(subset) != 2 * len(chosen) or set(subset.window_id) != ids:
        raise SystemExit("Feature subset does not exactly cover V2 matched IDs")
    if subset.archive_week.ne(PILOT_WEEK).any() or subset.altitude_band.ne(BAND).any() or subset.anchor_relative_seconds.gt(0).any() or subset.altitude_anchor_relative_seconds.gt(0).any():
        raise SystemExit("Feature subset violates first-week cruise pre-event scope")
    if subset.window_id.duplicated().any() or int(subset.ATC_INTERVENTION.sum()) != len(chosen):
        raise SystemExit("Feature subset is not one row per labelled window")
    chosen.to_parquet(out / "cruise_matching_v2.parquet", index=False, compression="zstd")
    balance.to_csv(out / "cruise_matching_v2_balance.csv", index=False)
    sensitivity.to_csv(out / "cruise_matching_v2_sensitivity.csv", index=False)
    subset.to_parquet(root / "05_features" / "window_features_cruise_matching_v2.parquet", index=False, compression="zstd")
    write_report(out / "cruise_matching_v2_report.md", len(v1), sensitivity, chosen, balance)
    (out / "cruise_matching_v2_run_config.json").write_text(json.dumps({"archive_week": PILOT_WEEK, "band": BAND,
        "v1_pairs": len(v1), "v2_pairs": len(chosen), "selected_config": choice, "calipers": CALIPERS,
        "source_intervention_pool": "V1 same-band cruise interventions", "source_control_pool": "all existing primary-cohort cruise controls",
        "clearance_free_audit": "all first-week clearance timestamps ±600 s", "without_replacement": True,
        "matching_variables": ["archive_week", "altitude_band", "analysis_altitude_ft", "flight_phase", "centre", "local_traffic_density", "ground_speed_at_anchor_kt", "hour_utc", "event_epoch", "aircraft_type_secondary"],
        "stage05_rerun": False, "stage07_executed": False, "other_scat_weeks_used": False}, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {choice}: {len(chosen):,}/{len(v1):,} pairs ({len(chosen)/len(v1)*100:.1f}%); exact V2 feature rows={len(subset):,}", flush=True)


if __name__ == "__main__":
    main()
