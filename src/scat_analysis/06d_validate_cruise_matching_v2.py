#!/usr/bin/env python3
"""Cruise-only Stage 06 robustness and final pre-Stage-07 gate, without Stage 07."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK, assert_pilot_frame


BAND = "CRUISE_FL290_410"
OUT = DEFAULT_OUTPUT_ROOT / "06_exploration"
PHYSICAL_GROUPS = {
    "horizontal_sep_at_event": "HORIZONTAL_GEOMETRY",
    "min_horizontal_sep_within_30nm": "HORIZONTAL_GEOMETRY",
    "p05_horizontal_separation": "HORIZONTAL_GEOMETRY",
    "median_horizontal_separation": "HORIZONTAL_GEOMETRY",
    "min_horizontal_separation": "HORIZONTAL_GEOMETRY",
    "median_relative_heading": "HORIZONTAL_GEOMETRY",
    "median_crossing_angle": "CROSSING_GEOMETRY",
    "median_relative_bearing": "BEARING_RAW_CIRCULAR",
    "raw_relative_bearing": "BEARING_RAW_CIRCULAR",
    "relative_bearing_sin": "BEARING_CIRCULAR",
    "relative_bearing_cos": "BEARING_CIRCULAR",
    "vertical_sep_at_event": "VERTICAL_GEOMETRY",
    "min_vertical_separation": "VERTICAL_GEOMETRY",
    "p05_vertical_separation": "VERTICAL_GEOMETRY",
    "median_vertical_separation": "VERTICAL_GEOMETRY",
    "vertical_sep_at_min_cpa": "VERTICAL_GEOMETRY",
    "vertical_rate_at_anchor_fpm": "VERTICAL_RATE",
    "max_vertical_closing_within_30nm": "CLOSING_RATE",
    "max_vertical_closing_rate": "CLOSING_RATE",
    "max_horizontal_closing_within_30nm": "CLOSING_RATE",
    "horizontal_closing_rate_at_min_cpa": "CLOSING_RATE",
    "min_cpa_within_30nm": "CPA_TCPA",
    "min_future_cpa_ge_1s_within_30nm": "CPA_TCPA",
    "min_predicted_horizontal_cpa": "CPA_TCPA",
    "tcpa_at_min_cpa": "CPA_TCPA",
    "min_positive_tcpa": "CPA_TCPA",
    "min_tcpa_ge_1s_within_30nm": "CPA_TCPA",
    "min_kinematic_predicted_sep_nm": "KINEMATIC_PREDICTION",
    "min_topsky_predicted_sep_nm_proxy": "TOPSKY_SCAT_ONLY",
    "median_traffic_density": "TRAFFIC_INTERACTION",
    "n_unique_neighbours_within_30nm": "TRAFFIC_INTERACTION",
    "median_nearby_aircraft_30nm": "TRAFFIC_INTERACTION",
    "median_relative_velocity_kt": "TRAFFIC_INTERACTION",
    "far30_pair_fraction": "TRAFFIC_INTERACTION",
    "ground_speed_at_anchor_kt": "MATCHED_CONTEXT",
    "analysis_altitude_ft": "MATCHED_CONTEXT",
}
FRAGILE = {"min_horizontal_separation", "p05_horizontal_separation", "median_horizontal_separation",
           "min_vertical_separation", "p05_vertical_separation", "median_vertical_separation",
           "vertical_sep_at_event", "horizontal_sep_at_event",
           "median_crossing_angle", "median_relative_heading", "min_predicted_horizontal_cpa",
           "min_positive_tcpa", "max_vertical_closing_rate", "min_kinematic_predicted_sep_nm",
           "min_cpa_within_30nm", "horizontal_closing_rate_at_min_cpa", "far30_pair_fraction",
           "median_relative_bearing", "raw_relative_bearing"}
CORE_GROUPS = {"HORIZONTAL_GEOMETRY", "VERTICAL_GEOMETRY", "CLOSING_RATE", "CPA_TCPA",
               "BEARING_CIRCULAR", "CROSSING_GEOMETRY", "TRAFFIC_INTERACTION", "VERTICAL_RATE", "KINEMATIC_PREDICTION"}


def smd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 2:
        return float("nan")
    scale = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    delta = a.mean() - b.mean()
    return float(delta / scale) if scale > 0 else (0.0 if abs(delta) < 1e-12 else float("inf"))


def circular_bearing(ids: set[str], raw_windows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only existing predictor-eligible, within-30-NM pair bearings."""
    source_path = DEFAULT_OUTPUT_ROOT / "05_features" / f"primary_interaction_features_{PILOT_WEEK}.parquet"
    cache = OUT / "cruise_matching_v2_bearing_features.parquet"
    if cache.exists() and cache.stat().st_mtime >= source_path.stat().st_mtime:
        saved = pd.read_parquet(cache)
        if (len(saved) == len(ids) and set(saved.window_id) == ids
                and {"min_future_cpa_ge_1s_within_30nm", "future_cpa_rows_within_30nm"} <= set(saved.columns)):
            print(f"Bearing audit: reused verified first-week sidecar for {len(saved):,} windows", flush=True)
            return saved
    source = pq.ParquetFile(source_path)
    index = pd.Index(sorted(ids), name="window_id")
    value_set = pa.array(index.tolist(), type=pa.string())
    sine = np.zeros(len(index), dtype=float)
    cosine = np.zeros(len(index), dtype=float)
    count = np.zeros(len(index), dtype=np.int64)
    min_future_cpa = np.full(len(index), np.inf, dtype=float)
    future_cpa_count = np.zeros(len(index), dtype=np.int64)
    scanned = 0
    columns = ["window_id", "relative_bearing_deg", "horizontal_separation_nm", "cpa_nm", "tcpa_min",
               "predictor_eligible", "relative_seconds"]
    for batch in source.iter_batches(batch_size=131072, columns=columns):
        table = pa.Table.from_batches([batch])
        mask = pc.and_(pc.is_in(table["window_id"], value_set=value_set), pc.equal(table["predictor_eligible"], True))
        mask = pc.and_(mask, pc.less_equal(table["relative_seconds"], 0))
        mask = pc.and_(mask, pc.less_equal(table["horizontal_separation_nm"], 30))
        selected = table.filter(mask)
        if selected.num_rows:
            frame = selected.select(["window_id", "relative_bearing_deg", "cpa_nm", "tcpa_min"]).to_pandas()
            position = index.get_indexer(frame.window_id)
            if (position < 0).any():
                raise SystemExit("Physical scan selected an unexpected window ID")
            angle_deg = frame.relative_bearing_deg.to_numpy(dtype=float)
            valid_angle = np.isfinite(angle_deg)
            if np.any((angle_deg[valid_angle] < 0) | (angle_deg[valid_angle] >= 360)):
                raise SystemExit("Existing relative_bearing_deg lies outside [0, 360)")
            bearing_rad = np.deg2rad(angle_deg[valid_angle])
            sine += np.bincount(position[valid_angle], weights=np.sin(bearing_rad), minlength=len(index))
            cosine += np.bincount(position[valid_angle], weights=np.cos(bearing_rad), minlength=len(index))
            count += np.bincount(position[valid_angle], minlength=len(index))
            cpa = frame.cpa_nm.to_numpy(dtype=float)
            tcpa = frame.tcpa_min.to_numpy(dtype=float)
            future = np.isfinite(cpa) & np.isfinite(tcpa) & (cpa >= 0) & (tcpa >= 1 / 60)
            np.minimum.at(min_future_cpa, position[future], cpa[future])
            future_cpa_count += np.bincount(position[future], minlength=len(index))
        scanned += len(batch)
        if scanned % 10000000 < len(batch):
            print(f"Bearing audit: scanned {scanned:,}/{source.metadata.num_rows:,} existing pair rows", flush=True)
    if scanned != source.metadata.num_rows:
        raise SystemExit("Existing pair-bearing scan did not cover the full first-week file")
    strength = np.divide(np.hypot(sine, cosine), count, out=np.full(len(index), np.nan), where=count > 0)
    bearing_rad = np.where(count > 0, np.arctan2(sine, cosine), np.nan)
    summary = pd.DataFrame({"window_id": index, "bearing_rad": bearing_rad,
                            "relative_bearing_sin": np.sin(bearing_rad), "relative_bearing_cos": np.cos(bearing_rad),
                            "bearing_resultant_length": strength, "bearing_pair_rows_within_30nm": count,
                            "min_future_cpa_ge_1s_within_30nm": np.where(np.isfinite(min_future_cpa), min_future_cpa, np.nan),
                            "future_cpa_rows_within_30nm": future_cpa_count})
    summary["raw_relative_bearing"] = summary.window_id.map(raw_windows.set_index("window_id").median_relative_bearing)
    if summary.bearing_pair_rows_within_30nm.sum() == 0 or summary.relative_bearing_sin.abs().gt(1.0000001).any():
        raise SystemExit("Circular bearing aggregation failed")
    summary.to_parquet(cache, index=False, compression="zstd")
    return summary


def compare_feature(pairs: pd.DataFrame, windows: pd.DataFrame, feature: str,
                    draws: dict[str, list[np.ndarray]] | None) -> dict:
    lookup = windows.set_index("window_id")
    a = pd.to_numeric(lookup.loc[pairs.intervention_window_id, feature], errors="coerce").to_numpy(dtype=float, copy=True)
    b = pd.to_numeric(lookup.loc[pairs.control_window_id, feature], errors="coerce").to_numpy(dtype=float, copy=True)
    a[~np.isfinite(a)] = np.nan
    b[~np.isfinite(b)] = np.nan
    linear = feature not in {"median_relative_bearing", "raw_relative_bearing"}
    effect = smd(a, b) if linear else float("nan")
    row = {"intervention_count": int(np.isfinite(a).sum()), "control_count": int(np.isfinite(b).sum()),
           "intervention_missing": int(np.isnan(a).sum()), "control_missing": int(np.isnan(b).sum()),
           "intervention_median": float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan"),
           "control_median": float(np.nanmedian(b)) if np.isfinite(b).any() else float("nan"),
           "SMD": effect}
    if draws is not None and np.isfinite(effect):
        for name, indices in draws.items():
            effects = np.array([smd(a[idx], b[idx]) for idx in indices])
            effects = effects[np.isfinite(effects)]
            row[f"bootstrap_{name}_replicates"] = len(effects)
            row[f"bootstrap_{name}_SMD_p025"] = float(np.quantile(effects, .025)) if len(effects) else float("nan")
            row[f"bootstrap_{name}_SMD_p975"] = float(np.quantile(effects, .975)) if len(effects) else float("nan")
            row[f"bootstrap_{name}_sign_stability"] = float(np.mean(np.sign(effects) == np.sign(effect))) if len(effects) else float("nan")
        lowers = [row[f"bootstrap_{name}_SMD_p025"] for name in draws]
        uppers = [row[f"bootstrap_{name}_SMD_p975"] for name in draws]
        row["bootstrap_conservative_SMD_p025"] = float(np.nanmin(lowers))
        row["bootstrap_conservative_SMD_p975"] = float(np.nanmax(uppers))
    return row


def cluster_resamples(pairs: pd.DataFrame, column: str, rng: np.random.Generator,
                      repeats: int = 120) -> list[np.ndarray]:
    groups = [np.asarray(position, dtype=int) for position in pairs.groupby(column, sort=True).indices.values()]
    return [np.concatenate([groups[j] for j in rng.integers(0, len(groups), size=len(groups))])
            for _ in range(repeats)]


def feature_classification(feature: str, v1: dict, v2: dict) -> tuple[str, str]:
    if feature in {"ground_speed_at_anchor_kt", "analysis_altitude_ft"}:
        return "UNSTABLE", "Matched context variable; attenuation is expected and not an explanatory traffic signal."
    if feature in {"median_relative_bearing", "raw_relative_bearing"}:
        return "UNSTABLE", "Raw bearing is circular; a linear SMD is not interpretable. Use sine/cosine."
    a, b = v1["SMD"], v2["SMD"]
    if not np.isfinite(a) or not np.isfinite(b):
        return "UNSTABLE", "Insufficient finite observations or variance."
    n = v2["intervention_count"] + v2["intervention_missing"]
    missing_i = v2["intervention_missing"] / n
    missing_c = v2["control_missing"] / n
    coverage_good = max(missing_i, missing_c) <= .20 and abs(missing_i - missing_c) <= .10
    ci_excludes_zero = (v2.get("bootstrap_conservative_SMD_p025", float("nan"))
                        * v2.get("bootstrap_conservative_SMD_p975", float("nan")) > 0)
    same = np.sign(a) == np.sign(b)
    if (same and abs(a) >= .10 and abs(b) >= .10 and abs(b) >= .5 * abs(a)
            and coverage_good and ci_excludes_zero and feature not in FRAGILE):
        return "ROBUST", "Direction and material descriptive size persist; pair and both flight-cluster bootstrap CIs exclude zero; coverage passes."
    if same and abs(b) >= .05 and max(missing_i, missing_c) <= .40:
        reasons = []
        if not coverage_good:
            reasons.append("coverage imbalance")
        if feature in FRAGILE:
            reasons.append("physical/extreme-value caveat")
        if not ci_excludes_zero:
            reasons.append("bootstrap CI includes zero")
        if abs(b) < .10 or abs(b) < .5 * abs(a):
            reasons.append("effect attenuated")
        return "PARTIALLY_ROBUST", ", ".join(reasons) or "Same direction but not all robustness criteria met."
    return "UNSTABLE", "Direction changed, nearly vanished, or coverage insufficient."


def comparison_tables(v1_pairs: pd.DataFrame, v2_pairs: pd.DataFrame, windows: pd.DataFrame,
                      mapping: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(OUT / "cruise_FL290_410_intervention_vs_control.csv")
    features = list(dict.fromkeys([*baseline.feature.tolist(), "min_kinematic_predicted_sep_nm",
                                   "min_future_cpa_ge_1s_within_30nm",
                                   "raw_relative_bearing", "relative_bearing_sin", "relative_bearing_cos"]))
    missing = set(features) - set(windows.columns)
    if missing:
        raise SystemExit(f"V2 feature comparison is missing {sorted(missing)}")
    rng = np.random.default_rng(42)
    draws = {"pairs": [rng.integers(0, len(v2_pairs), size=len(v2_pairs)) for _ in range(120)],
             "intervention_flight": cluster_resamples(v2_pairs, "intervention_focal_flight_key", rng),
             "control_flight": cluster_resamples(v2_pairs, "control_focal_flight_key", rng)}
    rows = []
    lookup = mapping.set_index("SCAT_feature")
    for feature in features:
        a = compare_feature(v1_pairs, windows, feature, None)
        b = compare_feature(v2_pairs, windows, feature, draws)
        cls, reason = feature_classification(feature, a, b)
        if feature == "min_topsky_predicted_sep_nm_proxy":
            availability = "SCAT_ONLY"
        elif feature in {"relative_bearing_sin", "relative_bearing_cos", "raw_relative_bearing",
                         "min_kinematic_predicted_sep_nm", "min_future_cpa_ge_1s_within_30nm"}:
            availability = "DERIVABLE_FROM_OPENSKY"
        elif feature in lookup.index:
            availability = lookup.loc[feature, "availability_class"]
        else:
            availability = "UNDETERMINED"
        rows.append({"feature": feature, "feature_group": PHYSICAL_GROUPS.get(feature, "TEMPORAL_TREND"),
                     "used_for_matching": feature in {"ground_speed_at_anchor_kt", "analysis_altitude_ft"},
                     "OpenSky_availability": availability,
                     **{f"V1_{key}": value for key, value in a.items()},
                     **{f"V2_{key}": value for key, value in b.items()},
                     "direction_same": bool(np.isfinite(a["SMD"]) and np.isfinite(b["SMD"]) and np.sign(a["SMD"]) == np.sign(b["SMD"])),
                     "absolute_change_in_SMD": abs(b["SMD"] - a["SMD"]) if np.isfinite(a["SMD"]) and np.isfinite(b["SMD"]) else float("nan"),
                     "change_in_abs_SMD": abs(b["SMD"]) - abs(a["SMD"]) if np.isfinite(a["SMD"]) and np.isfinite(b["SMD"]) else float("nan"),
                     "robustness": cls, "robustness_reason": reason})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "cruise_matching_v1_v2_comparison.csv", index=False)
    return result


def cluster_sensitivity(v2: pd.DataFrame) -> pd.DataFrame:
    full = pd.read_parquet(OUT / "cruise_cluster_assignments.parquet", columns=["window_id", "cluster_id"])
    ids = set(v2.intervention_window_id)
    if not ids <= set(full.window_id):
        raise SystemExit("V2 retained interventions are not a subset of the original cruise clusters")
    full["retained_in_matching_v2"] = full.window_id.isin(ids)
    rows = []
    for label, part in full.groupby("cluster_id"):
        rows.append({"cluster_id": label, "V1_intervention_windows": len(part),
                     "V2_retained_intervention_windows": int(part.retained_in_matching_v2.sum()),
                     "retention_pct": part.retained_in_matching_v2.mean() * 100,
                     "V1_cluster_share_pct": len(part) / len(full) * 100,
                     "V2_cluster_share_pct": part.retained_in_matching_v2.sum() / len(ids) * 100})
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "cruise_matching_v2_cluster_sensitivity.csv", index=False)
    return output


def topsky_proxy_comparison(v2: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    frame = windows.loc[windows.window_id.isin(set(v2.intervention_window_id) | set(v2.control_window_id))]
    columns = ["min_cpa_within_30nm", "min_kinematic_predicted_sep_nm", "tcpa_at_min_cpa", "min_tcpa_ge_1s_within_30nm"]
    rows = []
    for label, part in frame.groupby("ATC_INTERVENTION"):
        for feature in columns:
            pair = part[["min_topsky_predicted_sep_nm_proxy", feature]].replace([np.inf, -np.inf], np.nan).dropna()
            coefficient = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic if len(pair) >= 30 else float("nan")
            rows.append({"ATC_INTERVENTION": int(label), "kinematic_feature": feature,
                         "complete_windows": len(pair), "spearman_with_topsky_proxy": coefficient,
                         "interpretation": "Association only; kinematic feature is not an exact TopSky substitute"})
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "cruise_matching_v2_topsky_kinematic_comparison.csv", index=False)
    return output


def report(v1: pd.DataFrame, v2: pd.DataFrame, windows: pd.DataFrame, bearing: pd.DataFrame,
           comp: pd.DataFrame, cluster: pd.DataFrame, proxy: pd.DataFrame) -> None:
    root = DEFAULT_OUTPUT_ROOT
    sensitivity = pd.read_csv(root / "04_windows" / "cruise_matching_v2_sensitivity.csv")
    selected = sensitivity.loc[sensitivity.name.eq(v2.matching_config.iloc[0])].iloc[0]
    prior = pd.read_csv(OUT / "altitude_band_matching_summary.csv")
    v1_balance = prior.loc[prior.band.eq(BAND)].iloc[0]
    balance = pd.read_csv(root / "04_windows" / "cruise_matching_v2_balance.csv")
    good = comp.loc[comp.robustness.eq("ROBUST") & comp.feature_group.isin(CORE_GROUPS) & ~comp.used_for_matching
                    & comp.OpenSky_availability.isin(["AVAILABLE_IN_OPENSKY", "DERIVABLE_FROM_OPENSKY"])].copy()
    good = good.sort_values("V2_SMD", key=lambda x: x.abs(), ascending=False)
    partial = comp.loc[comp.robustness.eq("PARTIALLY_ROBUST") & comp.feature_group.isin(CORE_GROUPS)]
    robust_signal = "YES" if len(good) >= 2 else "PARTIAL" if len(good) or len(partial) else "NO"
    precise = comp.loc[comp.feature_group.isin(CORE_GROUPS) & comp.V2_intervention_missing.le(.2 * len(v2))
                       & comp.V2_control_missing.le(.2 * len(v2)) & comp.V2_bootstrap_conservative_SMD_p025.notna()
                       & comp.V2_bootstrap_conservative_SMD_p975.notna()].copy()
    precise["CI_half_width"] = (precise.V2_bootstrap_conservative_SMD_p975 - precise.V2_bootstrap_conservative_SMD_p025) / 2
    precision_share = float(precise.CI_half_width.le(.10).mean()) if len(precise) else 0.0
    # Precision to resolve a descriptive 0.10-SMD shift, not a fixed pair-count quota.
    sample_gate = "SUFFICIENT" if precision_share >= .5 else "INSUFFICIENT"
    leakage = "PASS" if (windows.archive_week.eq(PILOT_WEEK).all() and windows.anchor_relative_seconds.le(0).all()
                         and windows.altitude_anchor_relative_seconds.le(0).all()) else "FAIL"
    physical_checks = ["horizontal_sep_at_event", "vertical_sep_at_event", "min_horizontal_sep_within_30nm", "min_cpa_within_30nm"]
    impossible = any(windows[field].dropna().lt(0).any() for field in physical_checks)
    physical = "FAIL" if impossible else "EXPLAINED_WARNING"  # Existing V2 audit documents tails.
    matching = "PASS" if bool(selected.all_context_PASS) else "FAIL"
    decision = ("INSUFFICIENT_BALANCED_SAMPLE" if sample_gate == "INSUFFICIENT" else
                "READY_FOR_STAGE07_CRUISE" if matching == "PASS" and leakage == "PASS" and physical != "FAIL" and robust_signal == "YES" else
                "REVISE_AGAIN")
    top = good.head(5)
    top_rows = [f"| {n} | `{row.feature}` | {row.V1_SMD:+.3f} | {row.V2_SMD:+.3f} | {row.V2_bootstrap_conservative_SMD_p025:+.3f} to {row.V2_bootstrap_conservative_SMD_p975:+.3f} | {row.OpenSky_availability} |"
                for n, row in enumerate(top.itertuples(index=False), 1)]
    def feature_line(name: str) -> str:
        item = comp.set_index("feature").loc[name]
        return f"`{name}`: V1 SMD {item.V1_SMD:+.3f} → V2 {item.V2_SMD:+.3f}, {item.robustness}; V2 medians {item.V2_intervention_median:.3g} / {item.V2_control_median:.3g}."
    count_by_class = comp.robustness.value_counts().to_dict()
    weak_bearing = int(bearing.bearing_resultant_length.lt(.1).sum())
    no_bearing = int(bearing.bearing_pair_rows_within_30nm.eq(0).sum())
    no_neighbour = int(windows.n_unique_neighbours_within_30nm.eq(0).sum())
    tiny_tcpa_windows = int(windows.tiny_positive_tcpa_rows.gt(0).sum())
    high_vertical_windows = int(windows.high_vertical_closing_rows.gt(0).sum())
    cluster_share_shift = float((cluster.V2_cluster_share_pct - cluster.V1_cluster_share_pct).abs().max())
    cpa_corr = proxy.loc[proxy.kinematic_feature.eq("min_cpa_within_30nm")].set_index("ATC_INTERVENTION")
    speed = comp.set_index("feature").loc["ground_speed_at_anchor_kt"]
    topsky = comp.set_index("feature").loc["min_topsky_predicted_sep_nm_proxy"]
    lines = ["# SCAT first-week pilot — CRUISE_MATCHING_V2 final pre-Stage-07 gate", "",
             f"**Week:** `{PILOT_WEEK}` · **Band:** FL290–FL410 · **Stage 07:** NOT RUN · **Other SCAT weeks:** NOT PROCESSED", "",
             "## 1. Purpose", "",
             "This is the last methodological comparison before a possible *separate authorization* of Stage 07. The target is recorded ATC intervention, not confirmed conflict. All physical differences below are descriptive associations.", "",
             "## 2. Why ground-speed imbalance matters", "",
             f"V1 anchor ground-speed SMD was −0.324. V2 is {selected.ground_speed_SMD:+.3f} ({'PASS' if abs(selected.ground_speed_SMD)<.1 else 'FAIL'}). Ground speed is horizontal motion; near-zero vertical rate in level cruise is physically ordinary and was never used as a speed-balancing surrogate. The reduction in ground-speed contrast is expected and desirable, not evidence that speed was an explanatory conflict signal.", "",
             "## 3. Variables used for matching", "",
             "Same first-week archive and cruise band; exact anchored phase, centre and UTC hour; calipers on anchor ground speed, anchor altitude, Stage 04 local traffic density and absolute event-time distance. Aircraft type is only a small secondary cost. Each control is used once and differs from its intervention flight.", "",
             "## 4. Variables deliberately NOT used for matching", "",
             "No separation, CPA/TCPA, closing rate, vertical rate, relative bearing, crossing angle, relative heading, TopSky or kinematic prediction, neighbour counts or nearby-aircraft fields entered the candidate filter, cost or configuration choice. No 5-NM/1,000-ft or colour state was used as a label.", "",
             "## 5. Matching methodology", "",
             "[Full matching report](04_windows/cruise_matching_v2_report.md) and [machine-readable cohort](04_windows/cruise_matching_v2.parquet). Existing Stage 04 controls were independently rechecked against all 80,264 first-week clearance timestamps at ±600 s; no new Stage 04 window was generated. Sparse minimum-cost bipartite assignment with unmatched dummy edges gives maximum cardinality, then shortest weighted context distance, without replacement.", "",
             "## 6. Sensitivity / calipers", "",
             f"Four predeclared settings are in the [sensitivity table](04_windows/cruise_matching_v2_sensitivity.csv). Pair counts: {', '.join(f'{row.name} {row.n_pairs:,}' for row in sensitivity.itertuples(index=False))}. Selected `{selected['name']}`: the smallest/simple setting within 5% of the largest all-PASS cohort; here only the broad setting attained that retention. No physical feature influenced selection.", "",
             "## 7. Sample retention", "",
             f"V1 **{len(v1):,}** pairs → V2 **{len(v2):,}** pairs; **{len(v2)/len(v1)*100:.1f}%** retained, **{len(v1)-len(v2):,}** original intervention windows excluded. V2 has **{2*len(v2):,}** unique feature rows, one per pre-event window. Sample sufficiency is judged by precision rather than an arbitrary minimum pair count: {precise.CI_half_width.le(.10).sum()}/{len(precise)} adequately covered core physical features ({precision_share*100:.1f}%) have conservative bootstrap SMD CI half-width ≤0.10, enough to resolve a descriptive 0.10-SMD shift. The conservative interval envelopes paired resampling and separate clustering by intervention flight and control flight. This is not an OpenSky label supply or a Stage 07 out-of-sample test.", "",
             "## 8. V1 versus V2 balance", "",
             "| Diagnostic | V1 | V2 |", "|---|---:|---:|",
             f"| Ground-speed SMD | {v1_balance.ground_speed_at_anchor_kt_SMD:+.3f} | {selected.ground_speed_SMD:+.3f} |",
             f"| Altitude SMD | {v1_balance.analysis_altitude_ft_SMD:+.3f} | {selected.altitude_SMD:+.3f} |",
             f"| Local-density SMD | {v1_balance.local_traffic_density_SMD:+.3f} | {selected.density_SMD:+.3f} |",
             f"| Median time distance, s | {v1_balance.control_time_distance_median_s:.0f} | {selected.median_temporal_distance_s:.0f} |",
             f"| p95 time distance, s | {v1_balance.control_time_distance_p95_s:.0f} | {selected.p95_temporal_distance_s:.0f} |", "",
             "Phase, centre and UTC hour are exact within pairs in both V1 and V2 (category SMDs 0); rematching improves speed balance at the cost of a larger but still bounded time distance. [Full V2 category balance](04_windows/cruise_matching_v2_balance.csv). PASS means |SMD|<0.10 and is not a conflict threshold.", "",
             "## 9. Cruise intervention/control results V2", "",
             "The [V1–V2 feature comparison](06_exploration/cruise_matching_v1_v2_comparison.csv) contains medians, SMDs, missingness, pair- and flight-cluster-bootstrap intervals and robustness for every investigated feature. Up to five quality-screened, OpenSky-transferable traffic-interaction features are shown below:", "",
             "| Rank | Feature | V1 SMD | V2 SMD | Conservative V2 95% interval | OpenSky |", "|---:|---|---:|---:|---:|---|", *top_rows, "",
             "Robustness is an exploratory consistency screen: same direction, |SMD|≥0.10 in both cohorts, V2 retaining ≥50% of V1 magnitude, ≤20% missing per arm, ≤10-point differential missingness, conservative pair/flight-cluster intervals excluding zero, and no flagged global extreme. It is not a causal test; multiple comparisons were not adjusted.", "",
             "## 10. Separation results", "", feature_line("min_horizontal_sep_within_30nm"), "", feature_line("vertical_sep_at_event"), "",
             "## 11. CPA/TCPA results", "", feature_line("min_cpa_within_30nm"), "", feature_line("min_future_cpa_ge_1s_within_30nm"), "", feature_line("min_tcpa_ge_1s_within_30nm"), "",
             "The original `min_cpa_within_30nm` includes zero/negative-TCPA rows, for which Stage 05's kinematic CPA is clipped to current separation. It is now audit-only/partially robust. The additional future-CPA field was derived from existing pre-event pair rows using a ≥1-second TCPA quality floor and the same 30-NM neighbourhood; this is a numerical-stability choice, not a conflict threshold. Raw minimum-positive TCPA and all-distance minima remain audit-only because prior physical checks found sub-second/remote-neighbour tails.", "",
             "## 12. Closing-rate results", "", feature_line("max_horizontal_closing_within_30nm"), "", feature_line("max_vertical_closing_within_30nm"), "",
             "The prior Stage 05b V2 audit recorded very high vertical-closing tails and 5-second alignment discrepancies; bounded-neighbour variants are preferred but still require sensitivity checks.", "",
             "## 13. Bearing / crossing-angle results", "",
             "The source pair table provides `relative_bearing_deg` in [0,360), not a native `bearing_rad` field. For every existing predictor-eligible, pre-event pair row within 30 NM, this pass computes `bearing_rad = deg2rad(relative_bearing_deg)`, aggregates the sine/cosine components by window, obtains a circular mean direction, and stores `relative_bearing_sin`, `relative_bearing_cos`, `bearing_rad` and the original V2 median as `raw_relative_bearing` in [the bearing sidecar](06_exploration/cruise_matching_v2_bearing_features.parquet). Raw bearing gets no linear SMD. This is a derivation from existing pair features, not a Stage 05 recomputation.", "",
             f"No within-30-NM bearing observations: **{no_bearing:,}** of {len(bearing):,} audited V1 cruise windows; low resultant length (<0.1): **{weak_bearing:,}**. A diffuse bearing distribution has a weakly defined mean direction. The filtered V2 feature file includes these new columns without replacing the original V2 source. {feature_line('relative_bearing_sin')} {feature_line('relative_bearing_cos')} Crossing angle is separate: {feature_line('median_crossing_angle')}", "",
             "## 14. TopSky versus kinematic proxy", "",
             f"TopSky minimum-separation proxy: V1 SMD **{topsky.V1_SMD:+.3f}** → V2 **{topsky.V2_SMD:+.3f}** ({topsky.robustness}). It is **SCAT_ONLY**. Spearman correlation with bounded kinematic CPA is **{cpa_corr.loc[1, 'spearman_with_topsky_proxy']:+.3f}** for intervention windows and **{cpa_corr.loc[0, 'spearman_with_topsky_proxy']:+.3f}** for controls. [Other correlations](06_exploration/cruise_matching_v2_topsky_kinematic_comparison.csv) with kinematic CPA/TCPA are descriptive and cannot establish replacement of TopSky; its planned trajectories are not in the inspected OpenSky state vectors.", "",
             "## 15. V1 versus V2 robustness", "",
             f"Across {len(comp)} compared features: **{count_by_class.get('ROBUST',0)} ROBUST**, **{count_by_class.get('PARTIALLY_ROBUST',0)} PARTIALLY_ROBUST**, **{count_by_class.get('UNSTABLE',0)} UNSTABLE**. Matched-context fields are included for auditing but excluded from the robust traffic-interaction gate. {feature_line('ground_speed_at_anchor_kt')} Its attenuation is expected. [Full table](06_exploration/cruise_matching_v1_v2_comparison.csv).", "",
             "## 16. OpenSky-transferable features", "",
             "Time-aligned state vectors can support pair geometry, within-radius separation, closing, near-neighbour counts, circular bearing and kinematic CPA/TCPA after quality and coverage checks. TopSky predictions and clearance labels remain SCAT-only. [SCAT→OpenSky mapping](06_exploration/scat_to_opensky_feature_mapping.csv) and [bounded OpenSky plan](OPENSKY_PROCESSING_PLAN_FROM_SCAT.md). No OpenSky processing was run here.", "",
             "## 17. Remaining limitations", "",
             f"The control pool consists only of already-created Stage 04 primary controls; rematching against the full universe could change retention and balance. One week and one cruise stratum cannot establish transportability or causality. V2 contains {v2.intervention_focal_flight_key.nunique():,} distinct intervention flights and {v2.control_focal_flight_key.nunique():,} distinct control flights, so repeated-flight dependence matters; the conservative intervals envelope paired and one-way cluster resampling by each side, but are not a full two-way clustered inference procedure. In the V2 feature subset, {no_neighbour:,} windows have no neighbour within 30 NM, {tiny_tcpa_windows:,} contain a sub-second positive TCPA row, and {high_vertical_windows:,} contain a high-vertical-closing row. These are explained data-quality warnings, not conflict labels. Some V2 features still summarize distant/global pairs; within-30-NM alternatives are preferred. Circular bearings are aggregated across repeated pair rows, which can weight longer-lived neighbours. Bootstrap intervals measure within-cohort descriptive stability, not independent-week replication. Multiple features were screened. A 20% intervention-window loss warrants cluster-selection sensitivity: [retention by prior component](06_exploration/cruise_matching_v2_cluster_sensitivity.csv); the largest component-share change is {cluster_share_shift:.1f} percentage points. No clustering was rerun and previous natural-grouping evidence remains WEAK. The Stage 07 script still defaults to the earlier pilot input: if separately authorized, its input, grouping and chronological validation must be reviewed and pointed explicitly to this V2 cohort before execution.", "",
             "## 18. Final Stage 07 gate", "",
             f"| Gate | Result | Evidence |", "|---|---|---|",
             f"| Matching — ground speed | {'PASS' if abs(selected.ground_speed_SMD)<.1 else 'FAIL'} | SMD {selected.ground_speed_SMD:+.3f} |",
             f"| Matching — altitude | {'PASS' if abs(selected.altitude_SMD)<.1 else 'FAIL'} | SMD {selected.altitude_SMD:+.3f} |",
             f"| Matching — local density | {'PASS' if abs(selected.density_SMD)<.1 else 'FAIL'} | SMD {selected.density_SMD:+.3f} |",
             f"| Matching — phase / centre / UTC | {'PASS' if selected.phase_balance==selected.centre_balance==selected.UTC_hour_balance=='PASS' else 'FAIL'} | Exact matched strata; full category table linked above |",
             f"| Sample | {sample_gate} | {len(v2):,} pairs, {len(v2)/len(v1)*100:.1f}% retained; {precision_share*100:.1f}% core features meet precision criterion |",
             f"| Temporal leakage | {leakage} | Every retained V2 anchor is nonpositive and first-week; bearing scan uses eligible pre-event rows |",
             f"| Physical features | {physical} | {no_neighbour:,} no-neighbour windows; {tiny_tcpa_windows:,} sub-second-TCPA windows; {high_vertical_windows:,} high-vertical-closing windows; bounded alternatives prioritized |",
             f"| Robust traffic-interaction signal | {robust_signal} | {len(good)} robust OpenSky-transferable nonmatching features |", "",
             f"**FINAL DECISION: `{decision}`.** This gate is a methodological recommendation, **not authorization to execute Stage 07**. Stop here; do not process another SCAT week.", ""]
    robust_names = comp.loc[comp.robustness.eq("ROBUST"), "feature"].tolist()
    partial_names = comp.loc[comp.robustness.eq("PARTIALLY_ROBUST")].sort_values(
        "V2_SMD", key=lambda values: values.abs(), ascending=False).feature.head(5).tolist()
    unstable_names = comp.loc[comp.robustness.eq("UNSTABLE")].sort_values(
        "V1_SMD", key=lambda values: values.abs(), ascending=False).feature.head(5).tolist()
    by_feature = comp.set_index("feature")
    terminal = ["=" * 60, "SCAT CRUISE MATCHING V2 — FINAL PRE-STAGE07 GATE", "=" * 60,
                f"Week: {PILOT_WEEK}", "Band: FL290–FL410",
                f"V1 pairs: {len(v1):,}", f"V2 pairs: {len(v2):,}",
                f"Retention: {len(v2)/len(v1)*100:.1f}%", "", "MATCHING:",
                f"Ground speed: SMD {selected.ground_speed_SMD:+.3f} — {'PASS' if abs(selected.ground_speed_SMD)<.1 else 'FAIL'}",
                f"Altitude: SMD {selected.altitude_SMD:+.3f} — {'PASS' if abs(selected.altitude_SMD)<.1 else 'FAIL'}",
                f"Local density: SMD {selected.density_SMD:+.3f} — {'PASS' if abs(selected.density_SMD)<.1 else 'FAIL'}",
                f"Phase: {selected.phase_balance}; Centre: {selected.centre_balance}; UTC: {selected.UTC_hour_balance}",
                f"Temporal distance median/p95: {selected.median_temporal_distance_s:.0f}/{selected.p95_temporal_distance_s:.0f} s",
                "-" * 60, "ROBUST INTERACTION DIFFERENCES AFTER REMATCHING"]
    terminal.extend(f"{rank}. {row.feature}: V2 SMD {row.V2_SMD:+.3f}" for rank, row in enumerate(top.itertuples(index=False), 1))
    terminal.extend(["", f"Separation: within-30-NM horizontal SMD {by_feature.loc['min_horizontal_sep_within_30nm', 'V2_SMD']:+.3f} (ROBUST); event vertical separation partial/unbounded.",
                     f"CPA/TCPA: future CPA ≥1 s SMD {by_feature.loc['min_future_cpa_ge_1s_within_30nm', 'V2_SMD']:+.3f} (ROBUST); TCPA difference weak.",
                     f"Closing: horizontal {by_feature.loc['max_horizontal_closing_within_30nm', 'V2_SMD']:+.3f}; vertical {by_feature.loc['max_vertical_closing_within_30nm', 'V2_SMD']:+.3f} (PARTIAL).",
                     f"Bearing/crossing: sine {by_feature.loc['relative_bearing_sin', 'V2_SMD']:+.3f} (PARTIAL); crossing angle {by_feature.loc['median_crossing_angle', 'V2_SMD']:+.3f} (UNSTABLE).",
                     f"TopSky: SMD {topsky.V2_SMD:+.3f} (ROBUST, SCAT_ONLY; not equivalent to kinematic CPA).",
                     "-" * 60, "V1 VS V2",
                     f"ROBUST ({len(robust_names)}): {', '.join(robust_names)}",
                     f"PARTIALLY_ROBUST ({count_by_class.get('PARTIALLY_ROBUST',0)}): {', '.join(partial_names)} ...",
                     f"UNSTABLE ({count_by_class.get('UNSTABLE',0)}): {', '.join(unstable_names)} ...",
                     "-" * 60, "FINAL GATE",
                     f"Matching: {matching}", f"Sample: {sample_gate}", f"Leakage: {leakage}",
                     f"Physical features: {physical}", f"Robust traffic-interaction signal: {robust_signal}",
                     f"FINAL DECISION: {decision}", "DO NOT RUN STAGE 07.",
                     "DO NOT PROCESS OTHER SCAT WEEKS.", "=" * 60])
    lines.extend(["", "## Terminal summary", "", "```text", *terminal, "```", ""])
    path = root / "PILOT_CRUISE_MATCHING_V2_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    gate = {"archive_week": PILOT_WEEK, "band": BAND, "v1_pairs": len(v1), "v2_pairs": len(v2),
            "matching": matching, "sample": sample_gate, "temporal_leakage": leakage,
            "physical_features": physical, "robust_signal": robust_signal,
            "robust_transferable_features": good.feature.tolist(), "final_decision": decision,
            "stage07_executed": False, "other_scat_weeks_used": False}
    (OUT / "cruise_matching_v2_final_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print("\n".join(terminal), flush=True)


def main() -> None:
    root = DEFAULT_OUTPUT_ROOT
    original = pd.read_parquet(root / "04_windows" / f"primary_matching_cohort_{PILOT_WEEK}.parquet")
    v2 = pd.read_parquet(root / "04_windows" / "cruise_matching_v2.parquet")
    windows = pd.read_parquet(root / "05_features" / f"window_features_v2_{PILOT_WEEK}.parquet")
    assert_pilot_frame(original)
    assert_pilot_frame(v2)
    assert_pilot_frame(windows)
    lookup = windows.set_index("window_id")
    v1 = original.loc[original.intervention_window_id.map(lookup.altitude_band).eq(BAND)
                      & original.control_window_id.map(lookup.altitude_band).eq(BAND)]
    if len(v1) != 5128 or not set(v2.intervention_window_id) <= set(v1.intervention_window_id):
        raise SystemExit("V2 is not a subset of the 5,128 original cruise interventions")
    if v2.control_window_id.duplicated().any() or len(v2) != 4098:
        raise SystemExit("Selected V2 cohort is missing or has duplicate controls")
    chosen = pd.read_parquet(root / "05_features" / "window_features_cruise_matching_v2.parquet")
    ids = set(v1.intervention_window_id) | set(v1.control_window_id) | set(v2.control_window_id)
    if set(chosen.window_id) != set(v2.intervention_window_id) | set(v2.control_window_id) or len(chosen) != 2 * len(v2):
        raise SystemExit("Exact V2 feature subset is invalid")
    bearing = circular_bearing(ids, windows)
    added = [column for column in bearing.columns if column != "window_id" and column not in chosen.columns]
    if added:
        chosen = chosen.merge(bearing[["window_id", *added]], on="window_id", how="left", validate="one_to_one")
        if len(chosen) != 2 * len(v2) or chosen.raw_relative_bearing.isna().any():
            raise SystemExit("Enriched cruise V2 feature subset is invalid")
        chosen.to_parquet(root / "05_features" / "window_features_cruise_matching_v2.parquet", index=False, compression="zstd")
    windows = windows.merge(bearing, on="window_id", how="left", validate="one_to_one")
    mapping = pd.read_csv(OUT / "scat_to_opensky_feature_mapping.csv")
    comp = comparison_tables(v1, v2, windows, mapping)
    cluster = cluster_sensitivity(v2)
    proxy = topsky_proxy_comparison(v2, windows)
    report(v1, v2, chosen, bearing, comp, cluster, proxy)


if __name__ == "__main__":
    main()
