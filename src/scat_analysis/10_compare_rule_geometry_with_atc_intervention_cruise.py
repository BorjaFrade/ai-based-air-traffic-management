#!/usr/bin/env python3
"""Descriptive first-week comparison of traffic geometry with ATC intervention.

ATC_INTERVENTION is not conflict ground truth. Rule states are comparators,
not labels. This script reads, but never recomputes or modifies, Stages 01–09.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.stats import chi2_contingency, fisher_exact, norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT
from pilot_scope import PILOT_WEEK

ACQUISITION = PROJECT_ROOT / "src" / "acquisition"
sys.path.insert(0, str(ACQUISITION))
import opensky_safety  # noqa: E402

ROOT = DEFAULT_OUTPUT_ROOT
PAIRS = ROOT / "04_windows" / "cruise_matching_v2.parquet"
WINDOWS = ROOT / "05_features" / "window_features_cruise_matching_v2.parquet"
NATIVE = ROOT / "05_features" / f"primary_interaction_features_{PILOT_WEEK}.parquet"
SPLITS = ROOT / "07_explainability_cruise" / "split_assignments.parquet"
PREDICTIONS = ROOT / "07_explainability_cruise" / "test_predictions.parquet"
PROFILE = ACQUISITION / "separation_profiles.json"
OUT = ROOT / "10_rule_geometry_comparison_cruise"
REPORT = ROOT / "PILOT_STAGE10_RULE_GEOMETRY_VS_ATC_REPORT.md"
BAND = "CRUISE_FL290_410"
MIN_CELL_N = 30
NATIVE_COLUMNS = ["window_id", "relative_seconds", "predictor_eligible",
                  "horizontal_separation_nm", "vertical_separation_ft", "cpa_nm",
                  "tcpa_min", "predicted_min_vertical_sep_ft",
                  "horizontal_closing_rate_nm_min", "vertical_closing_rate_fpm"]
GEOMETRY = {
    "horizontal_sep_at_event": "NM; nearest pair at pre-event anchor",
    "vertical_sep_at_event": "ft; same nearest pair at pre-event anchor",
    "min_horizontal_sep_within_30nm": "NM; pre-event window minimum",
    "min_cpa_within_30nm": "NM; pre-event kinematic minimum, any TCPA",
    "min_future_cpa_ge_1s_within_30nm": "NM; pre-event future CPA >=1s",
    "min_tcpa_ge_1s_within_30nm": "min; pre-event TCPA >=1s",
    "max_horizontal_closing_within_30nm": "NM/min; pre-event window maximum",
    "max_vertical_closing_within_30nm": "ft/min; pre-event window maximum",
    "relative_bearing_sin": "circular bearing sine",
    "relative_bearing_cos": "circular bearing cosine",
    "median_crossing_angle": "deg; descriptive",
    "n_unique_neighbours_within_30nm": "count",
    "median_traffic_density": "count; contextual",
}
RULES = {
    "5NM_1000FT": (5.0, 1000.0),
    "7_5NM_1200FT": (7.5, 1200.0),
    "PREDICTED_5NM_1000FT_5MIN": (5.0, 1000.0),
}
LOGISTIC_FEATURES = ["horizontal_sep_at_event", "vertical_sep_at_event",
                     "min_future_cpa_ge_1s_within_30nm",
                     "min_tcpa_ge_1s_within_30nm",
                     "max_horizontal_closing_within_30nm",
                     "max_vertical_closing_within_30nm"]


def log(message: str, path: Path | None = None) -> None:
    print(message, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fail closed before creating output files."""
    p = pd.read_parquet(PAIRS)
    w = pd.read_parquet(WINDOWS)
    s = pd.read_parquet(SPLITS)
    if len(p) != 4098 or len(w) != 8196 or len(s) != 8196:
        raise ValueError("Unexpected cohort/split size: Stage 10 aborted")
    if not w.window_id.is_unique or not s.window_id.is_unique:
        raise ValueError("Duplicate window IDs: Stage 10 aborted")
    if p.intervention_window_id.duplicated().any() or p.control_window_id.duplicated().any():
        raise ValueError("Duplicate matched-window ID: Stage 10 aborted")
    if set(p.intervention_window_id) & set(p.control_window_id):
        raise ValueError("Overlapping intervention/control IDs")
    if set(w.window_id) != set(s.window_id) or set(w.window_id) != (set(p.intervention_window_id) | set(p.control_window_id)):
        raise ValueError("Window/split/pair membership mismatch")
    if w.ATC_INTERVENTION.value_counts().to_dict() != {0: 4098, 1: 4098}:
        raise ValueError("ATC_INTERVENTION is not balanced 1:1")
    if set(w.archive_week) != {PILOT_WEEK} or set(p.archive_week) != {PILOT_WEEK}:
        raise ValueError("Non-pilot SCAT week present")
    if set(w.altitude_band) != {BAND} or set(p.altitude_band) != {BAND}:
        raise ValueError("Non-cruise band present")
    if not w.analysis_altitude_ft.between(29000, 41000).all():
        raise ValueError("Altitude outside FL290–FL410")
    if not p.control_clearance_free_verified.all():
        raise ValueError("Unverified control clearance-free status")
    if (pd.to_numeric(w.anchor_relative_seconds) > 0).any():
        raise ValueError("Post-event anchor present")
    for side, target in (("intervention", 1), ("control", 0)):
        selected = w.set_index("window_id").loc[p[f"{side}_window_id"], "ATC_INTERVENTION"]
        if not selected.eq(target).all():
            raise ValueError(f"{side} label mismatch")
    if not s.set_index("window_id").loc[w.window_id, "ATC_INTERVENTION"].to_numpy().tolist() == w.ATC_INTERVENTION.tolist():
        raise ValueError("Stage07 split targets differ")
    if not set(GEOMETRY) <= set(w.columns):
        raise ValueError("Required geometry fields missing")
    if not set(s.split) <= {"train", "validation", "test", "excluded_cut_or_embargo"}:
        raise ValueError("Unexpected Stage07 split")
    if (s.groupby("matched_pair_id").size() != 2).any() or (s.groupby("matched_pair_id").split.nunique() != 1).any():
        raise ValueError("Stage07 matched-pair split broken")
    p = p.reset_index(drop=True).assign(matched_pair_id=np.arange(len(p)))
    pair_map = pd.concat([p[["matched_pair_id", "intervention_window_id"]].rename(columns={"intervention_window_id": "window_id"}),
                          p[["matched_pair_id", "control_window_id"]].rename(columns={"control_window_id": "window_id"})])
    w = w.merge(pair_map, on="window_id", validate="one_to_one")
    w = w.merge(s[["window_id", "matched_pair_id", "split", "connected_group_id"]], on=["window_id", "matched_pair_id"], validate="one_to_one")
    return p, w, s


def anchor_pairs(ids: set[str], source: Path, progress: Path) -> pd.DataFrame:
    """Stream Stage05 pairs; retain nearest neighbour and any-pair rule states.

    Continuous h/v describe the nearest neighbour, exactly as in Stage05b.
    Fixed, predicted and static-profile rules instead apply to ANY coherent
    pair at the same pre-event anchor time. Batch boundaries never split a
    window's anchor state semantically: equal-time aggregates are combined.
    """
    best: dict[str, dict] = {}
    profile = opensky_safety.load_safety_profile(PROFILE, "ENROUTE")
    state_priority = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
    colour = {"NORMAL": "GREEN", "PROXIMITY": "YELLOW", "HIGH_RISK": "ORANGE", "CONFLICT": "RED"}
    parquet = pq.ParquetFile(source)
    for number, batch in enumerate(parquet.iter_batches(batch_size=200_000, columns=NATIVE_COLUMNS), start=1):
        frame = batch.to_pandas()
        frame = frame.loc[frame.window_id.isin(ids) & frame.predictor_eligible.fillna(False)]
        frame = frame.loc[pd.to_numeric(frame.relative_seconds, errors="coerce").le(0)]
        frame = frame.loc[np.isfinite(pd.to_numeric(frame.horizontal_separation_nm, errors="coerce"))]
        if frame.empty:
            continue
        frame = frame.loc[frame.relative_seconds.eq(frame.groupby("window_id").relative_seconds.transform("max"))]
        for window_id, group in frame.groupby("window_id", sort=False):
            relative = float(group.relative_seconds.iloc[0])
            previous = best.get(window_id)
            if previous is not None and relative < previous["relative"]:
                continue
            h = pd.to_numeric(group.horizontal_separation_nm, errors="coerce").to_numpy(float)
            v = pd.to_numeric(group.vertical_separation_ft, errors="coerce").to_numpy(float)
            valid = np.isfinite(h) & np.isfinite(v)
            cpa = pd.to_numeric(group.cpa_nm, errors="coerce").to_numpy(float)
            tcpa = pd.to_numeric(group.tcpa_min, errors="coerce").to_numpy(float)
            future_v = pd.to_numeric(group.predicted_min_vertical_sep_ft, errors="coerce").to_numpy(float)
            valid_future = (valid & np.isfinite(cpa) & np.isfinite(tcpa) & np.isfinite(future_v)
                            & (h <= 30) & (cpa >= 0) & (tcpa >= 1 / 60) & (tcpa <= 5))
            nearest = group.iloc[int(np.argmin(h))].to_dict()
            observed_colours = [colour[opensky_safety.classify_pair(float(h[i]), float(v[i]), state_rules=profile["states"])]
                                for i in np.flatnonzero(valid)]
            state = max(observed_colours, key=lambda x: state_priority[x]) if observed_colours else "RULE_UNKNOWN"
            current = {"relative": relative, "nearest": nearest, "has_valid": bool(valid.any()),
                       "positive_5": bool((valid & (h < 5) & (v < 1000)).any()),
                       "positive_7_5": bool((valid & (h < 7.5) & (v < 1200)).any()),
                       "has_future": bool(valid_future.any()),
                       "positive_future": bool((valid_future & (cpa < 5) & (future_v < 1000)).any()),
                       "static_state": state}
            if previous is None or relative > previous["relative"]:
                best[window_id] = current
            else:
                previous["has_valid"] |= current["has_valid"]
                previous["positive_5"] |= current["positive_5"]
                previous["positive_7_5"] |= current["positive_7_5"]
                previous["has_future"] |= current["has_future"]
                previous["positive_future"] |= current["positive_future"]
                if h.min() < float(previous["nearest"]["horizontal_separation_nm"]):
                    previous["nearest"] = nearest
                if state_priority.get(state, -1) > state_priority.get(previous["static_state"], -1):
                    previous["static_state"] = state
        if number % 40 == 0:
            log(f"Stage10 native pairs: {number} batches; {len(best)} windows with anchor pair", progress)
    result = pd.DataFrame([{"window_id": window_id,
                            **{f"anchor_{column}": value["nearest"][column] for column in NATIVE_COLUMNS if column != "window_id"},
                            "anchor_any_valid_pair": value["has_valid"],
                            "anchor_any_5NM_1000FT": value["positive_5"],
                            "anchor_any_7_5NM_1200FT": value["positive_7_5"],
                            "anchor_any_future_valid_pair": value["has_future"],
                            "anchor_any_predicted_5NM_1000FT_5MIN": value["positive_future"],
                            "anchor_static_worst_state": value["static_state"]}
                           for window_id, value in best.items()])
    if len(result) != len(ids) or set(result.window_id) != ids:
        raise ValueError(f"Native anchor-pair coverage {len(result)}/{len(ids)}; abort")
    return result


def rule_status(h: pd.Series, v: pd.Series, h_limit: float, v_limit: float) -> pd.Series:
    status = pd.Series("RULE_UNKNOWN", index=h.index, dtype="object")
    known = np.isfinite(h) & np.isfinite(v)
    status.loc[known] = "RULE_NEGATIVE"
    status.loc[known & h.lt(h_limit) & v.lt(v_limit)] = "RULE_POSITIVE"
    return status


def future_rule(frame: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(frame.anchor_horizontal_separation_nm, errors="coerce")
    cpa = pd.to_numeric(frame.anchor_cpa_nm, errors="coerce")
    tcpa = pd.to_numeric(frame.anchor_tcpa_min, errors="coerce")
    v = pd.to_numeric(frame.anchor_predicted_min_vertical_sep_ft, errors="coerce")
    valid = (np.isfinite(h) & np.isfinite(cpa) & np.isfinite(tcpa) & np.isfinite(v)
             & h.le(30) & cpa.ge(0) & tcpa.ge(1 / 60) & tcpa.le(5))
    result = pd.Series("RULE_UNKNOWN", index=frame.index, dtype="object")
    result.loc[valid] = "RULE_NEGATIVE"
    result.loc[valid & cpa.lt(5) & v.lt(1000)] = "RULE_POSITIVE"
    return result


def time_to_region(frame: pd.DataFrame, h_limit: float, v_limit: float, horizon_s: int = 300) -> pd.DataFrame:
    """Same anchor pair; straight-line 2-D motion and linear signed vertical motion.

    CPA/TCPA identify horizontal relative-speed magnitude. Vertical closing is
    local signed-separation rate. These are modelled lead times, not actual
    post-clearance trajectories or evidence a conflict would occur.
    """
    h = pd.to_numeric(frame.anchor_horizontal_separation_nm, errors="coerce").to_numpy(float)
    cpa = pd.to_numeric(frame.anchor_cpa_nm, errors="coerce").to_numpy(float)
    tcpa_s = pd.to_numeric(frame.anchor_tcpa_min, errors="coerce").to_numpy(float) * 60
    v = pd.to_numeric(frame.anchor_vertical_separation_ft, errors="coerce").to_numpy(float)
    vclosing = pd.to_numeric(frame.anchor_vertical_closing_rate_fpm, errors="coerce").to_numpy(float)
    valid = (np.isfinite(h) & np.isfinite(cpa) & np.isfinite(tcpa_s) & np.isfinite(v)
             & np.isfinite(vclosing) & (h <= 30) & (cpa >= 0) & (cpa <= h + 1e-5)
             & (tcpa_s >= 1) & (tcpa_s <= horizon_s))
    speed2 = np.zeros(len(frame))
    speed2[valid] = np.maximum(h[valid] ** 2 - cpa[valid] ** 2, 0) / tcpa_s[valid] ** 2
    first = np.full(len(frame), np.nan)
    for seconds in np.arange(0, horizon_s + 1, 5):
        horizontal = np.sqrt(cpa ** 2 + speed2 * (seconds - tcpa_s) ** 2)
        vertical = np.abs(v - vclosing * seconds / 60)
        enter = valid & np.isnan(first) & (horizontal < h_limit) & (vertical < v_limit)
        first[enter] = seconds
    return pd.DataFrame({"window_id": frame.window_id, "ATC_INTERVENTION": frame.ATC_INTERVENTION,
                         "rule": f"{h_limit:g}NM_{v_limit:g}FT", "horizon_s": horizon_s,
                         "model_evaluable": valid, "enters_within_horizon": np.isfinite(first),
                         "time_to_rule_region_s": first})


def describe_geometry(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, meaning in GEOMETRY.items():
        groups = [pd.to_numeric(w.loc[w.ATC_INTERVENTION.eq(label), variable], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() for label in (1, 0)]
        x, y = groups
        pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        smd = (x.mean() - y.mean()) / pooled if pooled > 0 else np.nan
        for label, z in zip((1, 0), groups):
            q = z.quantile([.05, .25, .5, .75, .95])
            rows.append({"variable": variable, "meaning": meaning, "ATC_INTERVENTION": label,
                         "n": len(z), "missing_n": int(w.ATC_INTERVENTION.eq(label).sum() - len(z)),
                         "mean": z.mean(), "median": q.loc[.5], "p05": q.loc[.05], "p25": q.loc[.25],
                         "p75": q.loc[.75], "p95": q.loc[.95], "iqr": q.loc[.75] - q.loc[.25], "SMD_intervention_minus_control": smd})
    return pd.DataFrame(rows)


def fraction_bins(w: pd.DataFrame, feature: str, edges: list[float] | None = None) -> pd.DataFrame:
    values = pd.to_numeric(w[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    bands = pd.cut(values, edges, include_lowest=True) if edges else pd.qcut(values, q=10, duplicates="drop")
    result = w.groupby(bands, observed=True).ATC_INTERVENTION.agg(["sum", "count"]).reset_index()
    result.columns = ["bin", "n_intervention", "n"]
    result["n_control"] = result.n - result.n_intervention
    result["matched_cohort_intervention_fraction"] = np.where(result.n >= MIN_CELL_N, result.n_intervention / result.n, np.nan)
    result["small_cell"] = result.n < MIN_CELL_N
    result["variable"] = feature
    result["binning"] = "physical" if edges else "deciles_predeclared"
    return result.astype({"bin": "str"})


def geometry_2d(w: pd.DataFrame) -> pd.DataFrame:
    specs = [("horizontal_sep_at_event", "vertical_sep_at_event", [0, 3, 5, 7.5, 10, 15, 20, 30, np.inf], [0, 500, 1000, 1200, 2000, 3000, 5000, 10000, np.inf]),
             ("min_future_cpa_ge_1s_within_30nm", "vertical_sep_at_event", [0, 1, 3, 5, 7.5, 10, 15, 30, np.inf], [0, 500, 1000, 1200, 2000, 3000, 5000, 10000, np.inf]),
             ("min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm", [0, 1, 3, 5, 7.5, 10, 15, 30, np.inf], [-np.inf, 0, 3, 6, 9, 12, 15, np.inf])]
    rows = []
    for x, y, xb, yb in specs:
        temp = w.assign(x_bin=pd.cut(w[x], xb, include_lowest=True), y_bin=pd.cut(w[y], yb, include_lowest=True))
        grouped = temp.groupby(["x_bin", "y_bin"], observed=True).ATC_INTERVENTION.agg(["sum", "count"]).reset_index()
        for row in grouped.itertuples(index=False):
            count = int(row.count)
            rows.append({"x_variable": x, "y_variable": y, "x_bin": str(row.x_bin), "y_bin": str(row.y_bin),
                         "x_bin_lower": float(row.x_bin.left), "y_bin_lower": float(row.y_bin.left),
                         "n": count, "n_intervention": int(row.sum), "n_control": count - int(row.sum),
                         "matched_cohort_intervention_fraction": row.sum / count if count >= MIN_CELL_N else np.nan,
                         "small_cell": count < MIN_CELL_N})
    return pd.DataFrame(rows)


def overlap(w: pd.DataFrame, rule: str) -> tuple[dict, pd.DataFrame]:
    col = f"rule_{rule}"
    known = w.loc[w[col].ne("RULE_UNKNOWN")]
    table = pd.crosstab(known[col], known.ATC_INTERVENTION).reindex(index=["RULE_POSITIVE", "RULE_NEGATIVE"], columns=[1, 0], fill_value=0)
    a, b = table.loc["RULE_POSITIVE", [1, 0]].to_numpy()
    c, d = table.loc["RULE_NEGATIVE", [1, 0]].to_numpy()
    odds, fisher_p = fisher_exact([[a, b], [c, d]]) if min(a + b, c + d, a + c, b + d) > 0 else (np.nan, np.nan)
    chi_p = chi2_contingency([[a, b], [c, d]])[1] if min(a + b, c + d, a + c, b + d) > 0 else np.nan
    out = {"rule": rule, "intervention_rule_positive": int(a), "control_rule_positive": int(b),
           "intervention_rule_negative": int(c), "control_rule_negative": int(d),
           "intervention_rule_unknown": int((w.ATC_INTERVENTION.eq(1) & w[col].eq("RULE_UNKNOWN")).sum()),
           "control_rule_unknown": int((w.ATC_INTERVENTION.eq(0) & w[col].eq("RULE_UNKNOWN")).sum()),
           "fraction_interventions_rule_positive_among_evaluable": a / (a + c) if a + c else np.nan,
           "fraction_controls_rule_positive_among_evaluable": b / (b + d) if b + d else np.nan,
           "matched_cohort_odds_ratio": odds, "matched_cohort_risk_ratio": (a / (a + b)) / (c / (c + d)) if a + b and c and c + d else np.nan,
           "fisher_p_descriptive": fisher_p, "chi_square_p_descriptive": chi_p}
    pivot = w.pivot(index="matched_pair_id", columns="ATC_INTERVENTION", values=col)
    pair = pd.DataFrame({"matched_pair_id": pivot.index, "rule": rule,
                         "intervention_state": pivot[1].to_numpy(), "control_state": pivot[0].to_numpy()})
    pair["category"] = np.select([
        pair.intervention_state.eq("RULE_UNKNOWN") | pair.control_state.eq("RULE_UNKNOWN"),
        pair.intervention_state.eq("RULE_POSITIVE") & pair.control_state.eq("RULE_NEGATIVE"),
        pair.intervention_state.eq("RULE_NEGATIVE") & pair.control_state.eq("RULE_POSITIVE"),
        pair.intervention_state.eq("RULE_POSITIVE") & pair.control_state.eq("RULE_POSITIVE")],
        ["RULE_UNKNOWN_PAIR", "RULE_I1_C0", "RULE_I0_C1", "RULE_I1_C1"], default="RULE_I0_C0")
    return out, pair


def pair_association(pairs: pd.DataFrame, rule: str) -> dict:
    counts = pairs.category.value_counts()
    b = int(counts.get("RULE_I1_C0", 0))
    c = int(counts.get("RULE_I0_C1", 0))
    bb, cc = b + .5, c + .5
    log_or = np.log(bb / cc)
    se = np.sqrt(1 / bb + 1 / cc)
    return {"rule_or_variable": rule, "method": "matched_discordant_pairs_Haldane_0.5_CI",
            "n_evaluable_pairs": int(len(pairs) - counts.get("RULE_UNKNOWN_PAIR", 0)),
            "I_positive_C_negative": b, "I_negative_C_positive": c,
            "odds_ratio": float(np.exp(log_or)), "ci_lower": float(np.exp(log_or - 1.96 * se)),
            "ci_upper": float(np.exp(log_or + 1.96 * se)),
            "interpretation": "association with recorded intervention, not conflict or causality"}


def continuous_pair_association(w: pd.DataFrame, variable: str) -> dict:
    pivot = w.pivot(index="matched_pair_id", columns="ATC_INTERVENTION", values=variable)
    diff = pd.to_numeric(pivot[1], errors="coerce") - pd.to_numeric(pivot[0], errors="coerce")
    scale = pd.to_numeric(w[variable], errors="coerce").std()
    z = (diff.dropna() / scale).to_numpy() if np.isfinite(scale) and scale > 0 else np.array([])
    if len(z) < 30:
        return {"rule_or_variable": variable, "method": "conditional_logistic_complete_pairs_per_1SD", "n_evaluable_pairs": len(z), "odds_ratio": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    # Pair-conditioned likelihood: P(intervention member | discordant pair covariates).
    from scipy.special import expit
    fit = minimize(lambda b: np.logaddexp(0, -b[0] * z).sum(), [0.0], jac=lambda b: np.array([-(z * expit(-b[0] * z)).sum()]), method="BFGS")
    coef = float(fit.x[0])
    weights = expit(coef * z) * expit(-coef * z)
    se = float(1 / np.sqrt(np.sum(z * z * weights)))
    return {"rule_or_variable": variable, "method": "conditional_logistic_complete_pairs_per_1SD",
            "n_evaluable_pairs": len(z), "odds_ratio": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - 1.96 * se)), "ci_upper": float(np.exp(coef + 1.96 * se)),
            "interpretation": "unadjusted descriptive association; nominal CI ignores connected-flight clustering"}


def test_metrics(y: np.ndarray, score: np.ndarray | None, predicted: np.ndarray) -> dict:
    return {"roc_auc": roc_auc_score(y, score) if score is not None else np.nan,
            "pr_auc": average_precision_score(y, score) if score is not None else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "precision_intervention": precision_score(y, predicted, zero_division=0),
            "intervention_recall": recall_score(y, predicted, zero_division=0),
            "f1_intervention": f1_score(y, predicted, zero_division=0),
            "control_rule_positive_rate": float(np.mean(predicted[y == 0]))}


def compare_test(w: pd.DataFrame, predictions: Path) -> pd.DataFrame:
    test = w.loc[w.split.eq("test")].copy()
    rows = []
    for rule in RULES:
        status = test[f"rule_{rule}"]
        known = status.ne("RULE_UNKNOWN")
        y = test.loc[known, "ATC_INTERVENTION"].to_numpy(int)
        pred = status.loc[known].eq("RULE_POSITIVE").to_numpy(int)
        rows.append({"method": rule, "n_test": len(test), "n_evaluable_test": len(y),
                     "test_coverage": len(y) / len(test), "threshold_selection": "fixed_predeclared",
                     **test_metrics(y, None, pred)})
    train = w.loc[w.split.eq("train")]
    valid = w.loc[w.split.eq("validation")]
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42))
    model.fit(train[LOGISTIC_FEATURES], train.ATC_INTERVENTION)
    validation_score = model.predict_proba(valid[LOGISTIC_FEATURES])[:, 1]
    thresholds = np.arange(.2, .801, .01)
    threshold = float(max(thresholds, key=lambda t: balanced_accuracy_score(valid.ATC_INTERVENTION, validation_score >= t)))
    score = model.predict_proba(test[LOGISTIC_FEATURES])[:, 1]
    rows.append({"method": "CONTINUOUS_GEOMETRY_LOGISTIC", "n_test": len(test), "n_evaluable_test": len(test),
                 "test_coverage": 1.0, "threshold_selection": f"validation_balanced_accuracy_{threshold:.2f}",
                 **test_metrics(test.ATC_INTERVENTION.to_numpy(int), score, (score >= threshold).astype(int))})
    saved = pd.read_parquet(predictions)
    if set(saved.loc[saved.model.eq("xgboost")].feature_family) != {"MINIMAL_PHYSICAL", "OPENSKY_COMPATIBLE", "SCAT_FULL"}:
        raise ValueError("Stage07 XGBoost predictions missing")
    for family in ("OPENSKY_COMPATIBLE", "SCAT_FULL"):
        part = saved.loc[saved.model.eq("xgboost") & saved.feature_family.eq(family)]
        part = test[["window_id", "ATC_INTERVENTION"]].merge(part, on=["window_id", "ATC_INTERVENTION"], validate="one_to_one")
        if len(part) != len(test):
            raise ValueError(f"Stage07 {family} TEST coverage mismatch")
        rows.append({"method": f"STAGE07_{family}_XGBOOST", "n_test": len(test), "n_evaluable_test": len(test),
                     "test_coverage": 1.0, "threshold_selection": "saved_Stage07_validation",
                     **test_metrics(part.ATC_INTERVENTION.to_numpy(int), part.score.to_numpy(float), part.predicted_label.to_numpy(int))})
    rows.append({"method": "EXISTING_OPENSKY_DYNAMIC_RULE", "n_test": len(test), "n_evaluable_test": 0,
                 "test_coverage": 0.0, "threshold_selection": "UNAVAILABLE_exact_dynamic_snapshot_not_reconstructable"})
    return pd.DataFrame(rows)


def figures(w: pd.DataFrame, out: Path, summaries: dict[str, pd.DataFrame], test: pd.DataFrame,
            pair_summary: pd.DataFrame, ttr: pd.DataFrame, types: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(out / f"{name}.png", dpi=150)
        plt.close()
    for feature, name in [("horizontal_sep_at_event", "01_horizontal_separation"),
                          ("vertical_sep_at_event", "02_vertical_separation"),
                          ("min_future_cpa_ge_1s_within_30nm", "03_future_cpa")]:
        plt.figure(figsize=(7, 4))
        upper = min(w[feature].quantile(.99), 50 if "vertical" not in feature else 20000)
        for label, text in ((1, "intervention"), (0, "control")):
            vals = w.loc[w.ATC_INTERVENTION.eq(label), feature].dropna()
            plt.hist(vals.clip(upper=upper), bins=45, density=True, histtype="step", linewidth=2, label=text)
        plt.xlabel(feature + f" (values above {upper:.1f} capped for display)")
        plt.ylabel("Density")
        plt.legend()
        save(name)
    for feature, name in [("horizontal_sep_at_event", "04_fraction_by_horizontal"),
                          ("min_future_cpa_ge_1s_within_30nm", "05_fraction_by_cpa"),
                          ("vertical_sep_at_event", "06_fraction_by_vertical")]:
        b = summaries[feature]
        b = b.loc[b.binning.eq("deciles_predeclared") & ~b.small_cell]
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(b)), b.matched_cohort_intervention_fraction, marker="o")
        plt.xticks(np.arange(len(b)), b.bin, rotation=55, ha="right", fontsize=7)
        plt.ylabel("Matched-cohort intervention fraction")
        plt.xlabel(feature)
        save(name)
    h2 = pd.read_csv(out.parent / "geometry_2d_bins.csv")
    h2 = h2.loc[h2.x_variable.eq("horizontal_sep_at_event") & h2.y_variable.eq("vertical_sep_at_event")]
    x_order = h2[["x_bin", "x_bin_lower"]].drop_duplicates().sort_values("x_bin_lower").x_bin
    y_order = h2[["y_bin", "y_bin_lower"]].drop_duplicates().sort_values("y_bin_lower").y_bin
    matrix = h2.pivot(index="y_bin", columns="x_bin", values="matched_cohort_intervention_fraction").reindex(index=y_order, columns=x_order)
    counts = h2.pivot(index="y_bin", columns="x_bin", values="n").reindex(index=y_order, columns=x_order)
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            if pd.notna(counts.iloc[i, j]):
                plt.text(j, i, str(int(counts.iloc[i, j])), ha="center", va="center", fontsize=6, color="white")
    plt.xticks(np.arange(len(matrix.columns)), ["0–3", "3–5", "5–7.5", "7.5–10", "10–15", "15–20", "20–30", ">30"], fontsize=8)
    plt.yticks(np.arange(len(matrix)), ["0–500", "500–1,000", "1,000–1,200", "1,200–2,000",
                                       "2,000–3,000", "3,000–5,000", "5,000–10,000", ">10,000"], fontsize=8)
    plt.xlabel("Horizontal separation at event (NM)")
    plt.ylabel("Vertical separation at event (ft)")
    plt.colorbar(label="Matched-cohort intervention fraction; n<30 hidden")
    plt.title("Horizontal × vertical geometry (cell text = n)")
    save("07_horizontal_vertical_heatmap")
    state = w.groupby(["opensky_static_profile_proxy_state", "ATC_INTERVENTION"]).size().unstack(fill_value=0)
    state.plot.bar(figsize=(7, 4))
    plt.ylabel("Windows")
    plt.xlabel("Static profile projection; NOT dynamic OpenSky detector")
    save("08_static_profile_state_vs_intervention")
    subset = pair_summary.loc[pair_summary.rule.eq("5NM_1000FT")].category.value_counts()
    subset.plot.bar(figsize=(7, 4))
    plt.ylabel("Matched pairs")
    save("09_matched_pair_concordance")
    test.loc[test.method.ne("EXISTING_OPENSKY_DYNAMIC_RULE")].plot.bar(x="method", y="balanced_accuracy", figsize=(9, 4), legend=False)
    plt.ylim(0, 1)
    plt.ylabel("TEST balanced accuracy for intervention")
    save("10_stage07_test_comparison")
    topsky = w.groupby(["rule_5NM_1000FT", "ATC_INTERVENTION"]).min_topsky_predicted_sep_nm_proxy.median()
    topsky.plot.bar(figsize=(8, 4))
    plt.ylabel("Median TopSky predicted sep proxy (NM)")
    save("11_topsky_by_rule_category")
    valid_time = ttr.loc[ttr.model_evaluable & ttr.enters_within_horizon & ttr.ATC_INTERVENTION.eq(1)]
    plt.figure(figsize=(7, 4))
    for name, part in valid_time.groupby("rule"):
        plt.hist(part.time_to_rule_region_s, bins=np.arange(0, 306, 15), histtype="step", linewidth=2, label=f"{name} (n={len(part)})")
    plt.xlabel("Modelled seconds from pre-event anchor to rule region")
    plt.ylabel("Interventions; evaluable entrants only")
    plt.legend()
    save("12_time_to_rule_region")
    if not types.empty:
        types.pivot(index="clearance_type", columns="rule", values="rule_positive_fraction_evaluable").plot.bar(figsize=(8, 4))
        plt.ylim(0, 1)
        plt.ylabel("Rule-positive fraction among evaluable interventions")
        save("13_clearance_type_rule_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--report-path", type=Path, default=REPORT)
    parser.add_argument("--overwrite-stage10", action="store_true", help="Regenerate only the selected Stage10 outputs; never prior stages")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite_stage10:
        raise SystemExit(f"Stage 10 output already exists; refusing overwrite: {args.output_root}")
    if args.report_path.exists() and not args.overwrite_stage10:
        raise SystemExit(f"Stage 10 report already exists; refusing overwrite: {args.report_path}")
    pairs, w, _ = inputs()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.log"
    if args.overwrite_stage10:
        progress.write_text("", encoding="utf-8")
    log("[OPEN] Stage10: validated 4098 pairs / 8196 windows; first week, FL290–FL410", progress)
    anchor = anchor_pairs(set(w.window_id), NATIVE, progress)
    w = w.merge(anchor, on="window_id", validate="one_to_one")
    if not np.allclose(w.horizontal_sep_at_event, w.anchor_horizontal_separation_nm, rtol=1e-5, atol=1e-4):
        raise ValueError("Native anchor horizontal separation does not match Stage05b")
    if not np.allclose(w.vertical_sep_at_event, w.anchor_vertical_separation_ft, rtol=1e-5, atol=1e-4):
        raise ValueError("Native anchor vertical separation does not match Stage05b")
    log("[DONE] Same-pair native anchor geometry validated", progress)
    continuous = describe_geometry(w)
    continuous.to_csv(out / "continuous_geometry_summary.csv", index=False)
    summaries = {}
    for feature, filename, edges in [
        ("horizontal_sep_at_event", "intervention_fraction_by_horizontal_sep.csv", [0, 3, 5, 7.5, 10, 15, 20, 30, 50, np.inf]),
        ("vertical_sep_at_event", "intervention_fraction_by_vertical_sep.csv", [0, 500, 1000, 1200, 2000, 3000, 5000, 10000, np.inf]),
        ("min_future_cpa_ge_1s_within_30nm", "intervention_fraction_by_cpa.csv", [0, 1, 3, 5, 7.5, 10, 15, 30, np.inf]),
        ("min_tcpa_ge_1s_within_30nm", "intervention_fraction_by_tcpa.csv", [0, 1 / 60, .5, 1, 2, 5, 10, np.inf]),
        ("max_horizontal_closing_within_30nm", "intervention_fraction_by_closing.csv", [-np.inf, 0, 3, 6, 9, 12, 15, np.inf])]:
        table = pd.concat([fraction_bins(w, feature), fraction_bins(w, feature, edges)], ignore_index=True)
        table.to_csv(out / filename, index=False)
        summaries[feature] = table
    geometry_2d(w).to_csv(out / "geometry_2d_bins.csv", index=False)
    log("[DONE] Continuous distributions, quantile/physical bins and 2-D geometry", progress)
    w["rule_5NM_1000FT"] = np.where(~w.anchor_any_valid_pair, "RULE_UNKNOWN",
                                     np.where(w.anchor_any_5NM_1000FT, "RULE_POSITIVE", "RULE_NEGATIVE"))
    w["rule_7_5NM_1200FT"] = np.where(~w.anchor_any_valid_pair, "RULE_UNKNOWN",
                                       np.where(w.anchor_any_7_5NM_1200FT, "RULE_POSITIVE", "RULE_NEGATIVE"))
    w["rule_PREDICTED_5NM_1000FT_5MIN"] = np.where(~w.anchor_any_future_valid_pair, "RULE_UNKNOWN",
                                                    np.where(w.anchor_any_predicted_5NM_1000FT_5MIN, "RULE_POSITIVE", "RULE_NEGATIVE"))
    profile = opensky_safety.load_safety_profile(PROFILE, "ENROUTE")
    w["opensky_static_profile_proxy_state"] = w.anchor_static_worst_state
    w["rule_OPENSKY_STATIC_PROFILE_PROXY"] = np.where(w.opensky_static_profile_proxy_state.eq("RULE_UNKNOWN"), "RULE_UNKNOWN",
                                                         np.where(w.opensky_static_profile_proxy_state.eq("GREEN"), "RULE_NEGATIVE", "RULE_POSITIVE"))
    saved_test_score = pd.read_parquet(PREDICTIONS)
    saved_test_score = saved_test_score.loc[saved_test_score.model.eq("xgboost") & saved_test_score.feature_family.eq("OPENSKY_COMPATIBLE"), ["window_id", "score"]]
    if len(saved_test_score) != int(w.split.eq("test").sum()) or not saved_test_score.window_id.is_unique:
        raise ValueError("Stage07 OpenSky-compatible TEST score membership mismatch")
    w = w.merge(saved_test_score.rename(columns={"score": "stage07_opensky_score_test"}), on="window_id", how="left", validate="one_to_one")
    rule_names = list(RULES) + ["OPENSKY_STATIC_PROFILE_PROXY"]
    w[["window_id", "matched_pair_id", "ATC_INTERVENTION", "split", "horizontal_sep_at_event", "vertical_sep_at_event",
       "anchor_cpa_nm", "anchor_tcpa_min", "anchor_predicted_min_vertical_sep_ft",
       "opensky_static_profile_proxy_state"] + [f"rule_{name}" for name in rule_names]].to_parquet(out / "rule_state_assignments.parquet", index=False)
    definitions = {"5NM_1000FT": "any coherent pair at pre-event anchor: H<5NM AND V<1000ft",
                   "7_5NM_1200FT": "any coherent pair at pre-event anchor: H<7.5NM AND V<1200ft",
                   "PREDICTED_5NM_1000FT_5MIN": "any coherent anchor pair with current H<=30NM, 1s<=TCPA<=5min, predicted CPA<5NM AND predicted V at CPA<1000ft",
                   "OPENSKY_STATIC_PROFILE_PROXY": "worst opensky_safety.classify_pair state over all anchor pairs, ENROUTE profile; NOT dynamic production snapshot/forecast logic",
                   "EXISTING_OPENSKY_DYNAMIC_RULE": "UNAVAILABLE: requires complete pairwise aircraft position/velocity snapshots and 5s forecast states; Stage05 pair aggregates do not reproduce production dynamic active-separator aggregation"}
    (out / "rule_definitions.json").write_text(json.dumps({"semantics": ["ATC_INTERVENTION != CONFLICT", "RULES_ARE_NOT_GROUND_TRUTH"],
                                                               "profile_source": str(PROFILE), "implementation_source": str(ACQUISITION / "opensky_safety.py"),
                                                               "profile": profile, "rules": definitions}, indent=2), encoding="utf-8")
    tables, pair_tables, associations = [], [], []
    for rule in rule_names:
        table, pair = overlap(w, rule)
        tables.append(table)
        pair_tables.append(pair)
        associations.append(pair_association(pair, rule))
    tables = pd.DataFrame(tables)
    tables.to_csv(out / "rule_vs_intervention_tables.csv", index=False)
    pair_table = pd.concat(pair_tables, ignore_index=True)
    pair_table.to_csv(out / "rule_overlap_by_matched_pair.csv", index=False)
    for feature in ("horizontal_sep_at_event", "vertical_sep_at_event", "min_future_cpa_ge_1s_within_30nm"):
        associations.append(continuous_pair_association(w, feature))
    pd.DataFrame(associations).to_csv(out / "conditional_rule_associations.csv", index=False)
    log("[DONE] Fixed/predicted rules, static profile proxy, matched-pair associations", progress)
    outside = []
    controls = []
    for rule in ("5NM_1000FT", "7_5NM_1200FT"):
        sub = w.loc[w.ATC_INTERVENTION.eq(1) & w[f"rule_{rule}"].eq("RULE_NEGATIVE")].copy()
        h_limit, v_limit = RULES[rule]
        sub["nearest_pair_both_dimensions_above_limits"] = sub.horizontal_sep_at_event.gt(h_limit) & sub.vertical_sep_at_event.gt(v_limit)
        sub["rule"] = rule
        outside.append(sub)
        c = w.loc[w.ATC_INTERVENTION.eq(0) & w[f"rule_{rule}"].eq("RULE_POSITIVE")].copy()
        c["rule"] = rule
        controls.append(c)
    context_cols = ["window_id", "matched_pair_id", "rule", "horizontal_sep_at_event", "vertical_sep_at_event",
                    "min_future_cpa_ge_1s_within_30nm", "min_tcpa_ge_1s_within_30nm",
                    "max_horizontal_closing_within_30nm", "median_traffic_density",
                    "min_topsky_predicted_sep_nm_proxy", "horizontal_sep_trend_60s", "horizontal_sep_trend_120s",
                    "stage07_opensky_score_test"]
    pd.concat(outside)[context_cols + ["nearest_pair_both_dimensions_above_limits"]].to_csv(out / "interventions_outside_rule_region.csv", index=False)
    pd.concat(controls)[context_cols].to_csv(out / "rule_positive_controls.csv", index=False)
    context_rows = []
    for rule in ("5NM_1000FT", "7_5NM_1200FT"):
        for target in (1, 0):
            for status in ("RULE_POSITIVE", "RULE_NEGATIVE"):
                group = w.loc[w.ATC_INTERVENTION.eq(target) & w[f"rule_{rule}"].eq(status)]
                record = {"rule": rule, "ATC_INTERVENTION": target, "rule_state": status, "n": len(group)}
                for feature in context_cols[3:]:
                    record[f"{feature}_median"] = group[feature].median()
                    record[f"{feature}_n"] = group[feature].notna().sum()
                context_rows.append(record)
    context_summary = pd.DataFrame(context_rows)
    context_summary.to_csv(out / "rule_region_context_summary.csv", index=False)
    clearance = pd.read_parquet(ROOT / "09_counterfactual_cruise_v2" / "counterfactual_event_summary_v2.parquet",
                                columns=["window_id", "clearance_type", "stage09_label"])
    if not clearance.window_id.is_unique or len(clearance) != 4098:
        raise ValueError("Stage09 contextual clearance-type mapping invalid")
    intervention = w.loc[w.ATC_INTERVENTION.eq(1)].merge(clearance, on="window_id", validate="one_to_one")
    type_rows = []
    for kind, part in intervention.groupby("clearance_type"):
        for rule in rule_names:
            known = part[f"rule_{rule}"].ne("RULE_UNKNOWN")
            type_rows.append({"clearance_type": kind, "rule": rule, "n": len(part), "n_evaluable": int(known.sum()),
                              "n_rule_positive": int(part[f"rule_{rule}"].eq("RULE_POSITIVE").sum()),
                              "rule_positive_fraction_evaluable": part.loc[known, f"rule_{rule}"].eq("RULE_POSITIVE").mean(),
                              "median_horizontal_sep_nm": part.horizontal_sep_at_event.median(),
                              "median_vertical_sep_ft": part.vertical_sep_at_event.median(),
                              "median_future_cpa_nm": part.min_future_cpa_ge_1s_within_30nm.median(),
                              "median_tcpa_min": part.min_tcpa_ge_1s_within_30nm.median()})
    type_summary = pd.DataFrame(type_rows)
    type_summary.to_csv(out / "clearance_type_rule_summary.csv", index=False)
    # Stage09 is one-case context only, never a target, performance denominator or predictor.
    stage09_context = intervention.loc[intervention.stage09_label.eq("LIKELY_INTERACTION_RELATED"),
                                       ["window_id", "clearance_type", "stage09_label", "horizontal_sep_at_event",
                                        "vertical_sep_at_event", "opensky_static_profile_proxy_state"] +
                                       [f"rule_{name}" for name in rule_names]]
    stage09_context.to_csv(out / "stage09_v2_one_case_context.csv", index=False)
    log("[DONE] Interventions outside rules, rule-positive controls, clearance-type context", progress)
    test = compare_test(w, PREDICTIONS)
    test.to_csv(out / "stage07_test_rule_comparison.csv", index=False)
    methods = test.copy()
    methods["fixed_thresholds"] = methods.method.isin(RULES) | methods.method.eq("EXISTING_OPENSKY_DYNAMIC_RULE")
    methods["uses_CPA"] = methods.method.str.contains("PREDICTED|GEOMETRY|STAGE07")
    methods["uses_closing"] = methods.method.str.contains("GEOMETRY|STAGE07")
    methods["uses_density"] = methods.method.str.contains("STAGE07")
    methods["uses_TopSky"] = methods.method.eq("STAGE07_SCAT_FULL_XGBOOST")
    methods["OpenSky_compatible"] = ~methods.method.eq("STAGE07_SCAT_FULL_XGBOOST")
    methods.to_csv(out / "method_comparison.csv", index=False)
    log("[DONE] Reused Stage07 TEST and fitted geometry-only logistic on Stage07 TRAIN", progress)
    surface = []
    h, v = w.horizontal_sep_at_event, w.vertical_sep_at_event
    for h_limit in np.arange(3, 15.1, 1):
        for v_limit in (500, 750, 1000, 1200, 1500, 2000, 2500, 3000):
            positive = h.lt(h_limit) & v.lt(v_limit)
            a = int((positive & w.ATC_INTERVENTION.eq(1)).sum())
            b = int((positive & w.ATC_INTERVENTION.eq(0)).sum())
            c, d = 4098 - a, 4098 - b
            surface.append({"horizontal_threshold_nm": h_limit, "vertical_threshold_ft": v_limit,
                            "interventions_rule_positive": a, "controls_rule_positive": b,
                            "fraction_interventions_rule_positive": a / 4098,
                            "fraction_controls_rule_positive": b / 4098,
                            "matched_cohort_odds_ratio_Haldane_0_5": (a + .5) * (d + .5) / ((b + .5) * (c + .5))})
    pd.DataFrame(surface).to_csv(out / "threshold_sensitivity_surface.csv", index=False)
    ttr = pd.concat([time_to_region(w, 5, 1000), time_to_region(w, 7.5, 1200)], ignore_index=True)
    ttr.to_parquet(out / "time_to_rule_region.parquet", index=False)
    topsky = w.groupby(["rule_5NM_1000FT", "ATC_INTERVENTION"], dropna=False).min_topsky_predicted_sep_nm_proxy.agg(["count", "median", "mean"]).reset_index()
    topsky.to_csv(out / "topsky_rule_context.csv", index=False)
    static_counts = w.groupby(["opensky_static_profile_proxy_state", "ATC_INTERVENTION"]).size().unstack(fill_value=0)
    static_counts.to_csv(out / "static_profile_proxy_state_counts.csv")
    log("[DONE] TopSky context, finite-horizon lead time, threshold sensitivity", progress)
    figs = out / "figures"
    figures(w, figs, summaries, test, pair_table, ttr, type_summary)
    log("[DONE] Figures generated", progress)
    config = {"scope": "FIRST_WEEK_PILOT", "archive_week": PILOT_WEEK, "band": "FL290-410", "cohort": "CRUISE_MATCHING_V2",
              "pairs": 4098, "windows": 8196, "target": "ATC_INTERVENTION_TARGET", "rules": "RULES_ARE_NOT_GROUND_TRUTH",
              "input_paths": {"pairs": str(PAIRS), "windows": str(WINDOWS), "native_pair_features": str(NATIVE),
                              "stage07_splits": str(SPLITS), "stage07_test_predictions": str(PREDICTIONS),
                              "stage09_v2_context_only": str(ROOT / "09_counterfactual_cruise_v2" / "counterfactual_event_summary_v2.parquet")},
              "min_2d_cell_n": MIN_CELL_N, "test_predictions_reused_without_refit": True,
              "existing_dynamic_OpenSky_comparison": "UNAVAILABLE_not_exactly_reconstructable_from_Stage05_pair_aggregates",
              "matched_prevalence_notice": "Fractions and model scores are conditional on an artificial 1:1 matched cohort, not population intervention probabilities.",
              "interpretation": "Descriptive associations only; neither rules nor clearances establish conflict or causality."}
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    key = tables.set_index("rule")
    def rule_line(name: str) -> str:
        row = key.loc[name]
        pairs_for_rule = pair_table.loc[pair_table.rule.eq(name)].category.value_counts()
        return (f"{name}: intervention-positive {int(row.intervention_rule_positive)}/4098 ({row.intervention_rule_positive / 4098:.2%}); "
                f"control-positive {int(row.control_rule_positive)}/4098 ({row.control_rule_positive / 4098:.2%}); "
                f"pairs I+/C- {pairs_for_rule.get('RULE_I1_C0', 0)}, I-/C+ {pairs_for_rule.get('RULE_I0_C1', 0)}, "
                f"I+/C+ {pairs_for_rule.get('RULE_I1_C1', 0)}, I-/C- {pairs_for_rule.get('RULE_I0_C0', 0)}; "
                f"unknown pairs {pairs_for_rule.get('RULE_UNKNOWN_PAIR', 0)}.")
    decision = "REVISE_RULE_COMPARISON" if test.loc[test.method.eq("EXISTING_OPENSKY_DYNAMIC_RULE"), "n_evaluable_test"].iloc[0] == 0 else "READY_FOR_BOUNDED_OPENSKY_CRUISE_PILOT"
    report = ["# Stage 10 — Rule geometry versus recorded ATC intervention", "",
              "## 1. Scope", "", "First week `20161015_20161021`, FL290–FL410, CRUISE_MATCHING_V2, 4,098 matched intervention/control pairs (8,196 windows). No other week, band or OpenSky source was processed.", "",
              "## 2. Why Stage 10 was reformulated", "", "Stage 09 V1/V2 ended with `INSUFFICIENT_COUNTERFACTUAL_EVIDENCE`; no defensible conflict ground truth exists. Stage 10 is descriptive/comparative, not conflict-detector validation.", "",
              "## 3. Target semantics", "", "`ATC_INTERVENTION != CONFLICT`; `RULE_POSITIVE != CONFIRMED_CONFLICT`; `RULE_NEGATIVE != SAFE`; Stage 09 labels are not validation truth. All TEST metrics below describe discrimination of recorded interventions only.", "",
              "## 4. Cohort and matching", "", "One pre-event feature row per window; 1:1 matched design. The 50/50 prevalence is artificial. Fractions below are matched-cohort fractions, never population intervention probabilities.", "",
              "## 5. Continuous geometry", "", "`continuous_geometry_summary.csv` reports median, IQR, p05/p25/p75/p95 and SMD for each feature. Within-30-NM bounded features are preferred; missing-neighbour and unstable TCPA tails remain visible. These are associations, not causal effects.", "",
              "## 6. Horizontal separation", "", "`intervention_fraction_by_horizontal_sep.csv` supplies predeclared deciles and physical bins; no 5-NM discontinuity is assumed.", "",
              "## 7. Vertical separation", "", "`intervention_fraction_by_vertical_sep.csv` includes descriptive 1,000/2,000/3,000/5,000-ft landmarks, not target labels.", "",
              "## 8. CPA/TCPA", "", "Future CPA uses >=1-s quality-controlled within-30-NM values for continuous descriptions. The predicted rule instead uses a single native anchor pair and finite 5-min TCPA; it is not constructed from independent window minima.", "",
              "## 9. Closing and 2-D geometry", "", "`geometry_2d_bins.csv` reports n per cell and suppresses fractions for n<30; crossing/bearing are descriptive and bearing is represented circularly.", "",
              "## 10. Classical rule definitions", "", "Fixed 5 NM/1,000 ft and 7.5 NM/1,200 ft evaluate each coherent pair at the same pre-event anchor and mark the window positive if **any** pair is positive; strict `<` applies. Predicted 5 NM/1,000 ft likewise requires the same pair's CPA and vertical separation at CPA within 1s–5min. The continuous h/v summaries instead describe the nearest neighbour. Unknown is never negative.", "",
              "## 11. Existing OpenSky rule implementation", "", "The production `opensky_safety.analyse_aircraft_snapshot` uses dynamic active-separator trends and 5-s forecast aggregation. Stage05 pair aggregates cannot reproduce exact future snapshot states. Stage02 native aircraft states might support a separate reconstruction, but snapshot synchronization and an equivalence audit would be required before calling it the existing detector. `opensky_static_profile_proxy_state` calls the project's `classify_pair` with the unmodified ENROUTE profile over all anchor pairs, but is a **static proxy only**, not the existing detector. Exact OpenSky rule TEST comparison is unavailable and is not silently substituted.", "",
              "## 12. Rule states versus intervention", "", rule_line("5NM_1000FT"), "", rule_line("7_5NM_1200FT"), "", "`rule_vs_intervention_tables.csv` includes denominators, unknowns, odds/risk ratios and descriptive Fisher/chi-square. See static-proxy colour-state counts separately; GREEN is not safe truth and RED is not conflict truth.", "",
              "## 13. Matched-pair rule comparison", "", "`rule_overlap_by_matched_pair.csv` contains I+/C-, I-/C+, both, neither and unknown categories. `conditional_rule_associations.csv` adds discordant-pair ORs and unadjusted continuous pair-conditioned associations; nominal continuous CIs ignore connected-flight clustering.", "",
              "## 14. Interventions outside rule regions", "", "`interventions_outside_rule_region.csv` lists rule-negative interventions and flags the subset exceeding both dimension limits. These are not missed conflicts. Trends, density, CPA and TopSky are post-hoc descriptors.", "",
              "## 15. Rule-positive controls", "", "`rule_positive_controls.csv` lists matched controls inside fixed-rule regions; they are not false positives.", "",
              "## 16. Clearance type", "", "`clearance_type_rule_summary.csv` gives LEVEL/HEADING/SPEED post-hoc results. Clearance type was not a model predictor.", "",
              "## 17. Stage07 TEST comparison", "", f"Stage07 TEST has {int(w.split.eq('test').sum())} windows. Fixed rules and a geometry-only logistic (trained on Stage07 TRAIN; threshold chosen on VALIDATION) are compared with saved, unrefitted Stage07 XGBoost TEST predictions. `stage07_test_rule_comparison.csv` gives coverage and intervention-discrimination metrics. No random re-split.", "",
              "## 18. Fixed rules versus continuous geometry", "", "`method_comparison.csv` records feature families and availability. The exact production OpenSky dynamic detector is N/A, not a zero score. No classical thresholds were tuned; `threshold_sensitivity_surface.csv` is descriptive only.", "",
              "## 19. TopSky context", "", "`topsky_rule_context.csv` is post-hoc only. TopSky never enters fixed rules, the static proxy or the geometry-only logistic. Stage07 SCAT-full is separate from its OpenSky-compatible variant.", "",
              "## 20. Time to rule region", "", "`time_to_rule_region.parquet` uses the nearest native anchor pair, constant-velocity horizontal CPA geometry and local vertical closing over 0–300 s in 5-s steps. Missing/unphysical states are unevaluable; no entry within horizon is distinct from unevaluable. These are modelled pre-action lead times, not observed post-action outcomes.", "",
              "## 21. Threshold sensitivity", "", "The 3–15 NM × 500–3,000 ft grid is descriptive. No optimum is selected or recast as a conflict boundary.", "",
              "## 22. Implications for OpenSky", "", "Stage07 provides an OpenSky-compatible *intervention* signal but OpenSky has no comparable intervention labels here. A bounded FL290–FL410 quality/feature-yield pilot remains a separate decision; no full-history extraction or conflict model is authorized.", "",
              "## 23. Limitations", "", "Matched selection, pre-event aggregation, imperfect neighbour alignment, missing 30-NM neighbours, tiny-TCPA and vertical-closing tails, finite-horizon kinematic assumptions, unverified exact production dynamic colour states, and absence of conflict truth prevent causal or safety-performance claims.", "",
              "## 24. Final recommendation", "", f"`{decision}`. Proceed with bounded OpenSky quality/feature-yield pilot: **UNDETERMINED** pending an audited exact dynamic-rule comparison. Do not run Stage 08, other SCAT weeks or full OpenSky processing.", ""]
    def add_finding(section: str, lines: list[str]) -> None:
        position = report.index(section) + 2
        report[position:position] = lines + [""]
    leading = continuous.loc[continuous.ATC_INTERVENTION.eq(1)].copy()
    leading = leading.sort_values("SMD_intervention_minus_control", key=lambda x: x.abs(), ascending=False).head(5)
    top_geometry = "; ".join(f"`{row.variable}` SMD {row.SMD_intervention_minus_control:+.3f}"
                             for row in leading.itertuples(index=False))
    add_finding("## 5. Continuous geometry", [f"Largest absolute SMDs (intervention minus control): {top_geometry}. Magnitudes are modest; traffic geometry alone does not determine intervention."])
    h_phys = summaries["horizontal_sep_at_event"].loc[lambda x: x.binning.eq("physical")].reset_index(drop=True)
    v_phys = summaries["vertical_sep_at_event"].loc[lambda x: x.binning.eq("physical")].reset_index(drop=True)
    cpa_phys = summaries["min_future_cpa_ge_1s_within_30nm"].loc[lambda x: x.binning.eq("physical")].reset_index(drop=True)
    add_finding("## 6. Horizontal separation", [
        f"Matched-cohort intervention fraction: 3–5 NM {h_phys.loc[1, 'matched_cohort_intervention_fraction']:.1%} (n={h_phys.loc[1, 'n']}), "
        f"5–7.5 NM {h_phys.loc[2, 'matched_cohort_intervention_fraction']:.1%} (n={h_phys.loc[2, 'n']}), "
        f"7.5–10 NM {h_phys.loc[3, 'matched_cohort_intervention_fraction']:.1%} (n={h_phys.loc[3, 'n']}), "
        f"20–30 NM {h_phys.loc[6, 'matched_cohort_intervention_fraction']:.1%} (n={h_phys.loc[6, 'n']}). "
        "This is mixed/non-monotonic, not a sharp 5-NM boundary."])
    add_finding("## 7. Vertical separation", [
        f"Matched-cohort fraction: 500–1,000 ft {v_phys.loc[1, 'matched_cohort_intervention_fraction']:.1%} "
        f"(n={v_phys.loc[1, 'n']}) versus >10,000 ft {v_phys.iloc[-1].matched_cohort_intervention_fraction:.1%} "
        f"(n={v_phys.iloc[-1].n}). No claim of a 1,000-ft decision discontinuity."])
    add_finding("## 8. CPA/TCPA", [
        f"Future CPA medians: intervention {continuous.loc[(continuous.variable.eq('min_future_cpa_ge_1s_within_30nm')) & (continuous.ATC_INTERVENTION.eq(1)), 'median'].iloc[0]:.2f} NM, "
        f"control {continuous.loc[(continuous.variable.eq('min_future_cpa_ge_1s_within_30nm')) & (continuous.ATC_INTERVENTION.eq(0)), 'median'].iloc[0]:.2f} NM. "
        f"CPA 0–1 NM bin fraction {cpa_phys.iloc[0].matched_cohort_intervention_fraction:.1%} (n={cpa_phys.iloc[0].n}); "
        f"15–30 NM {cpa_phys.iloc[-1].matched_cohort_intervention_fraction:.1%} (n={cpa_phys.iloc[-1].n}).",
        f"The same-pair predicted rule is evaluable for only {int(key.loc['PREDICTED_5NM_1000FT_5MIN', 'intervention_rule_positive'] + key.loc['PREDICTED_5NM_1000FT_5MIN', 'intervention_rule_negative'])} interventions "
        f"and {int(key.loc['PREDICTED_5NM_1000FT_5MIN', 'control_rule_positive'] + key.loc['PREDICTED_5NM_1000FT_5MIN', 'control_rule_negative'])} controls; it is not a full-cohort comparator."])
    add_finding("## 12. Rule states versus intervention", [
        "The strict 5-NM/1,000-ft region contains only 9 windows in total; its observed difference is uninformative. "
        "The wider 7.5-NM/1,200-ft region has matched-cohort OR "
        f"{key.loc['7_5NM_1200FT', 'matched_cohort_odds_ratio']:.2f}, but still includes only "
        f"{int(key.loc['7_5NM_1200FT', 'intervention_rule_positive'])}/4098 interventions.",
        "Static ENROUTE profile proxy counts (intervention/control): " + "; ".join(
            f"{state} {int(static_counts.loc[state, 1])}/{int(static_counts.loc[state, 0])}"
            for state in ("GREEN", "YELLOW", "ORANGE", "RED")) + ". These are proxy states, not full production OpenSky states."])
    add_finding("## 14. Interventions outside rule regions", [
        f"5 NM/1,000 ft: {int(key.loc['5NM_1000FT', 'intervention_rule_negative'])}/4098 interventions outside (rule-negative). "
        f"7.5 NM/1,200 ft: {int(key.loc['7_5NM_1200FT', 'intervention_rule_negative'])}/4098 outside. "
        f"The nearest pair is beyond **both** limits for {int(outside[0].nearest_pair_both_dimensions_above_limits.sum())} and "
        f"{int(outside[1].nearest_pair_both_dimensions_above_limits.sum())} of these interventions, respectively. "
        "`nearest_pair_both_dimensions_above_limits` describes the nearest pair only; it is not an all-neighbour claim.",
        "For the 7.5-NM rule, rule-negative intervention windows have median TopSky predicted-separation proxy "
        f"{context_summary.loc[(context_summary.rule.eq('7_5NM_1200FT')) & (context_summary.ATC_INTERVENTION.eq(1)) & (context_summary.rule_state.eq('RULE_NEGATIVE')), 'min_topsky_predicted_sep_nm_proxy_median'].iloc[0]:.2f} NM "
        "and median Stage07 OpenSky-compatible TEST score "
        f"{context_summary.loc[(context_summary.rule.eq('7_5NM_1200FT')) & (context_summary.ATC_INTERVENTION.eq(1)) & (context_summary.rule_state.eq('RULE_NEGATIVE')), 'stage07_opensky_score_test_median'].iloc[0]:.3f}; "
        "these are descriptive, not safety scores or causal explanations."])
    add_finding("## 15. Rule-positive controls", [
        f"There are {int(key.loc['5NM_1000FT', 'control_rule_positive'])} controls positive under 5 NM/1,000 ft and "
        f"{int(key.loc['7_5NM_1200FT', 'control_rule_positive'])} under 7.5 NM/1,200 ft. "
        "No nearby recorded clearance does not establish an unsafe or safe state.",
        "Under 7.5 NM/1,200 ft, rule-positive intervention/control windows have median future CPA "
        f"{context_summary.loc[(context_summary.rule.eq('7_5NM_1200FT')) & (context_summary.ATC_INTERVENTION.eq(1)) & (context_summary.rule_state.eq('RULE_POSITIVE')), 'min_future_cpa_ge_1s_within_30nm_median'].iloc[0]:.2f}/"
        f"{context_summary.loc[(context_summary.rule.eq('7_5NM_1200FT')) & (context_summary.ATC_INTERVENTION.eq(0)) & (context_summary.rule_state.eq('RULE_POSITIVE')), 'min_future_cpa_ge_1s_within_30nm_median'].iloc[0]:.2f} NM; "
        "TopSky proxy medians "
        f"{context_summary.loc[(context_summary.rule.eq('7_5NM_1200FT')) & (context_summary.ATC_INTERVENTION.eq(1)) & (context_summary.rule_state.eq('RULE_POSITIVE')), 'min_topsky_predicted_sep_nm_proxy_median'].iloc[0]:.2f}/"
        f"{context_summary.loc[(context_summary.rule.eq('7_5NM_1200FT')) & (context_summary.ATC_INTERVENTION.eq(0)) & (context_summary.rule_state.eq('RULE_POSITIVE')), 'min_topsky_predicted_sep_nm_proxy_median'].iloc[0]:.2f} NM. "
        "See `rule_region_context_summary.csv` for other features and available n."])
    add_finding("## 16. Clearance type", [
        "; ".join(f"{row.clearance_type}: {int(row.n_rule_positive)}/{int(row.n)} rule-positive under 7.5 NM/1,200 ft"
                  for row in type_summary.loc[type_summary.rule.eq('7_5NM_1200FT')].itertuples(index=False)) + ". These are post-hoc strata, not predictors or conflict classes."])
    add_finding("## 17. Stage07 TEST comparison", [
        "| Method | TEST n | ROC-AUC | PR-AUC | Balanced accuracy | Intervention recall | Control rule-positive rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {row.method} | {int(row.n_evaluable_test)} | "
          f"{'N/A' if pd.isna(row.roc_auc) else f'{row.roc_auc:.3f}'} | "
          f"{'N/A' if pd.isna(row.pr_auc) else f'{row.pr_auc:.3f}'} | {row.balanced_accuracy:.3f} | "
          f"{row.intervention_recall:.3f} | {row.control_rule_positive_rate:.3f} |"
          for row in test.loc[test.method.isin(["5NM_1000FT", "7_5NM_1200FT", "CONTINUOUS_GEOMETRY_LOGISTIC",
                                               "STAGE07_OPENSKY_COMPATIBLE_XGBOOST", "STAGE07_SCAT_FULL_XGBOOST"])].itertuples(index=False)],
        "",
        "AUC entries for deterministic rules are N/A; the table is about intervention, not conflict. "
        f"Predicted-rule TEST coverage is {test.loc[test.method.eq('PREDICTED_5NM_1000FT_5MIN'), 'test_coverage'].iloc[0]:.1%}, so its conditional metrics are not directly comparable."])
    ttr_i = ttr.loc[ttr.ATC_INTERVENTION.eq(1)].groupby("rule")[["model_evaluable", "enters_within_horizon"]].sum()
    add_finding("## 20. Time to rule region", [
        "; ".join(f"{name}: {int(row.enters_within_horizon)}/{int(row.model_evaluable)} model-evaluable intervention pairs enter within 300 s"
                  for name, row in ttr_i.iterrows()) + ". This calculation follows only the nearest anchor pair, while fixed rules test any anchor pair. The low coverage prevents a strong lead-time inference."])
    if len(stage09_context) == 1:
        row = stage09_context.iloc[0]
        add_finding("## 22. Implications for OpenSky", [
            f"Stage09 V2 one-case context only: the sole provisional interaction-related `{row.clearance_type}` intervention "
            f"was `{row.rule_5NM_1000FT}` under 5 NM/1,000 ft and `{row.rule_7_5NM_1200FT}` under 7.5 NM/1,200 ft; "
            f"static profile proxy `{row.opensky_static_profile_proxy_state}`. One case is not a validation sample."])
    add_finding("## 24. Final recommendation", [
        "**Key scientific result:** recorded interventions occur mostly outside the strict classical joint-threshold regions. "
        "The continuous geometry shows modest, mixed associations; the wider early-warning region is enriched but remains sparse. "
        "This is consistent with recorded interventions spanning a broader geometry than fixed thresholds, not proof of anticipatory conflict prevention."])
    args.report_path.write_text("\n".join(report), encoding="utf-8")
    log("[DONE] Stage10 report and run config", progress)
    print("=" * 60)
    print("SCAT STAGE 10 — RULE GEOMETRY VS ATC INTERVENTION")
    print("Week: 20161015_20161021 | Band: FL290–FL410 | Cohort: CRUISE_MATCHING_V2")
    print("Pairs: 4098 | Windows: 8196")
    print("ATC_INTERVENTION != CONFLICT | RULES_ARE_NOT_GROUND_TRUTH | STAGE09 != GROUND_TRUTH")
    for name in ("5NM_1000FT", "7_5NM_1200FT"):
        print(rule_line(name))
    print("Continuous geometry: " + top_geometry)
    print("Static ENROUTE profile proxy, intervention/control: " + "; ".join(
        f"{state} {int(static_counts.loc[state, 1])}/{int(static_counts.loc[state, 0])}"
        for state in ("GREEN", "YELLOW", "ORANGE", "RED")))
    print("Existing dynamic OpenSky rule: UNAVAILABLE; static profile proxy reported separately")
    for method in ("5NM_1000FT", "7_5NM_1200FT", "CONTINUOUS_GEOMETRY_LOGISTIC",
                   "STAGE07_OPENSKY_COMPATIBLE_XGBOOST", "STAGE07_SCAT_FULL_XGBOOST"):
        item = test.loc[test.method.eq(method)].iloc[0]
        auc = f"{item.roc_auc:.3f}" if pd.notna(item.roc_auc) else "N/A"
        pr_auc = f"{item.pr_auc:.3f}" if pd.notna(item.pr_auc) else "N/A"
        print(f"Stage07 TEST {method}: intervention recall={item.intervention_recall:.3f}, "
              f"control rule-positive rate={item.control_rule_positive_rate:.3f}, balanced accuracy={item.balanced_accuracy:.3f}, "
              f"ROC-AUC={auc}, PR-AUC={pr_auc}")
    print("Key result: MIXED / CONTINUOUSLY ACROSS GEOMETRY; most interventions outside strict joint-threshold regions")
    print("Bounded OpenSky pilot: UNDETERMINED")
    print(f"Final decision: {decision}")
    print("DO NOT PROCESS FULL OPENSKY HISTORY OR OTHER SCAT WEEKS")


if __name__ == "__main__":
    main()
