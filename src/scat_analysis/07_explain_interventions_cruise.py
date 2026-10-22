#!/usr/bin/env python3
"""First-week, FL290–410 matched-cohort ATC-intervention explainability only."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from common import DEFAULT_OUTPUT_ROOT
from pilot_scope import PILOT_WEEK


ROOT = DEFAULT_OUTPUT_ROOT
PAIRS_PATH = ROOT / "04_windows" / "cruise_matching_v2.parquet"
WINDOWS_PATH = ROOT / "05_features" / "window_features_cruise_matching_v2.parquet"
MAPPING_PATH = ROOT / "06_exploration" / "scat_to_opensky_feature_mapping.csv"
OUT = ROOT / "07_explainability_cruise"
REPORT = ROOT / "PILOT_STAGE07_CRUISE_EXPLAINABILITY_REPORT.md"
BAND = "CRUISE_FL290_410"
SEED = 42
EMBARGO = 600  # seconds on each side of a chronological cut; 1,200 s total gap

# Explicit allowlist. Every remaining source column is non-predictive by default.
# The families and ablations are fixed before any model is fitted.
FEATURES = {
    "min_horizontal_sep_within_30nm": ("separation", "DERIVABLE_FROM_OPENSKY"),
    "vertical_sep_at_event": ("separation", "DERIVABLE_FROM_OPENSKY"),
    "median_vertical_separation": ("separation", "DERIVABLE_FROM_OPENSKY"),
    "min_future_cpa_ge_1s_within_30nm": ("cpa_tcpa", "DERIVABLE_FROM_OPENSKY"),
    "max_horizontal_closing_within_30nm": ("closing", "DERIVABLE_FROM_OPENSKY"),
    "max_vertical_closing_within_30nm": ("closing", "DERIVABLE_FROM_OPENSKY"),
    "relative_bearing_sin": ("geometry", "DERIVABLE_FROM_OPENSKY"),
    "relative_bearing_cos": ("geometry", "DERIVABLE_FROM_OPENSKY"),
    "median_crossing_angle": ("geometry", "DERIVABLE_FROM_OPENSKY"),
    "median_traffic_density": ("density_neighbours", "DERIVABLE_FROM_OPENSKY"),
    "n_unique_neighbours_within_30nm": ("density_neighbours", "DERIVABLE_FROM_OPENSKY"),
    "ground_speed_at_anchor_kt": ("context", "AVAILABLE_IN_OPENSKY"),
    "analysis_altitude_ft": ("context", "AVAILABLE_IN_OPENSKY"),
    "vertical_rate_at_anchor_fpm": ("context", "AVAILABLE_IN_OPENSKY"),
    "horizontal_sep_trend_60s": ("temporal", "DERIVABLE_FROM_OPENSKY"),
    "horizontal_sep_trend_120s": ("temporal", "DERIVABLE_FROM_OPENSKY"),
    "vertical_sep_trend_60s": ("temporal", "DERIVABLE_FROM_OPENSKY"),
    "vertical_sep_trend_120s": ("temporal", "DERIVABLE_FROM_OPENSKY"),
    "min_topsky_predicted_sep_nm_proxy": ("topsky", "SCAT_ONLY"),
}
MINIMAL = ["min_horizontal_sep_within_30nm", "vertical_sep_at_event",
           "min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
           "max_vertical_closing_within_30nm", "relative_bearing_sin",
           "relative_bearing_cos", "n_unique_neighbours_within_30nm"]
CONDITIONAL = ["min_horizontal_sep_within_30nm", "vertical_sep_at_event",
               "min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
               "relative_bearing_sin", "n_unique_neighbours_within_30nm"]
BLACKLIST = ["ATC_INTERVENTION", "window_id", "matched_pair_id", "matching_*", "matching_cost",
             "matching_distance", "clearance_*", "control_for_clearance_id", "assigned_*",
             "post_clearance_*", "future_clearance_*", "future_flight_plan_*", "GREEN",
             "YELLOW", "ORANGE", "RED", "cluster_*", "archive_week", "analysis_scope",
             "raw_relative_bearing", "median_relative_bearing", "bearing_rad", "min_positive_tcpa",
             "min_horizontal_separation", "min_predicted_horizontal_cpa", "min_cpa_within_30nm",
             "min_kinematic_predicted_sep_nm", "far30_pair_fraction", "event_epoch",
             "focal_flight_key", "flight_phase", "centre", "hour_utc"]
NOTICE = ("Predicted scores/probabilities are conditional on the matched analytical cohort and "
          "are not calibrated estimates of population ATC-intervention prevalence.")


def audit_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(PAIRS_PATH).copy()
    windows = pd.read_parquet(WINDOWS_PATH).copy()
    assert len(pairs) == 4098 and len(windows) == 8196, "Unexpected V2 cohort size; stop."
    assert windows.window_id.is_unique and not pairs.intervention_window_id.duplicated().any()
    assert not pairs.control_window_id.duplicated().any()
    assert set(pairs.intervention_window_id).isdisjoint(pairs.control_window_id)
    assert set(windows.window_id) == set(pairs.intervention_window_id) | set(pairs.control_window_id)
    assert windows.ATC_INTERVENTION.value_counts().to_dict() == {0: 4098, 1: 4098}
    assert set(windows.loc[windows.ATC_INTERVENTION == 1, "window_id"]) == set(pairs.intervention_window_id)
    assert set(windows.loc[windows.ATC_INTERVENTION == 0, "window_id"]) == set(pairs.control_window_id)
    assert set(windows.archive_week) == set(pairs.archive_week) == {PILOT_WEEK}
    assert set(windows.altitude_band) == set(pairs.altitude_band) == {BAND}
    assert windows.analysis_altitude_ft.between(29000, 41000).all()
    assert (pd.to_numeric(windows.anchor_relative_seconds) <= 0).all()
    assert (pd.to_numeric(windows.altitude_anchor_relative_seconds) <= 0).all()
    assert pairs.control_clearance_free_verified.all()
    lookup = windows.set_index("window_id")
    for side, label in (("intervention", 1), ("control", 0)):
        ids = pairs[f"{side}_window_id"]
        assert (lookup.loc[ids, "ATC_INTERVENTION"].to_numpy() == label).all()
        assert (lookup.loc[ids, "focal_flight_key"].to_numpy() == pairs[f"{side}_focal_flight_key"].to_numpy()).all()
        assert np.allclose(lookup.loc[ids, "event_epoch"].to_numpy(), pairs[f"{side}_event_epoch"].to_numpy())
    assert set(FEATURES) <= set(windows.columns), "A predeclared feature is absent."
    assert not set(FEATURES) & set(BLACKLIST)
    return pairs.reset_index(drop=True), windows.reset_index(drop=True)


def feature_sets(mapping: pd.DataFrame) -> dict[str, list[str]]:
    classes = dict(zip(mapping.SCAT_feature, mapping.availability_class))
    # V2-only derived fields inherit provenance from their documented source primitives.
    aliases = {"min_future_cpa_ge_1s_within_30nm": "min_cpa_within_30nm",
               "relative_bearing_sin": "median_relative_bearing",
               "relative_bearing_cos": "median_relative_bearing"}
    for feature, (_, availability) in FEATURES.items():
        source = aliases.get(feature, feature)
        assert classes.get(source) == availability, (feature, source, classes.get(source), availability)
    compatible = [f for f, (_, a) in FEATURES.items() if a in {"AVAILABLE_IN_OPENSKY", "DERIVABLE_FROM_OPENSKY"}]
    return {"SCAT_FULL": list(FEATURES), "OPENSKY_COMPATIBLE": compatible,
            "MINIMAL_PHYSICAL": MINIMAL, "FULL_WITHOUT_TOPSKY": compatible}


def split_pairs(pairs: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    keys = pd.Index(pd.concat([pairs.intervention_focal_flight_key, pairs.control_focal_flight_key]).unique())
    left = keys.get_indexer(pairs.intervention_focal_flight_key)
    right = keys.get_indexer(pairs.control_focal_flight_key)
    graph = coo_matrix((np.ones(2 * len(left)), (np.r_[left, right], np.r_[right, left])),
                       shape=(len(keys), len(keys))).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    p = pairs.copy()
    p["matched_pair_id"] = np.arange(len(p), dtype=np.int32)
    p["connected_group_id"] = labels[left]
    p["pair_min_epoch"] = p[["intervention_event_epoch", "control_event_epoch"]].min(axis=1)
    p["pair_max_epoch"] = p[["intervention_event_epoch", "control_event_epoch"]].max(axis=1)
    anchors = np.r_[p.intervention_event_epoch.to_numpy(), p.control_event_epoch.to_numpy()]
    cut1, cut2 = np.quantile(anchors, [.70, .85])
    group = p.groupby("connected_group_id").agg(lo=("pair_min_epoch", "min"),
                                                   hi=("pair_max_epoch", "max"), pairs=("matched_pair_id", "size"))
    group["split"] = "excluded_cut_or_embargo"
    group.loc[group.hi <= cut1 - EMBARGO, "split"] = "train"
    group.loc[(group.lo >= cut1 + EMBARGO) & (group.hi <= cut2 - EMBARGO), "split"] = "validation"
    group.loc[group.lo >= cut2 + EMBARGO, "split"] = "test"
    p["split"] = p.connected_group_id.map(group.split)
    assert set(p.split) >= {"train", "validation", "test"}, "Split lacks a required partition."
    lookup = windows.set_index("window_id")
    rows = []
    for side, label in (("intervention", 1), ("control", 0)):
        part = p[["matched_pair_id", "connected_group_id", "split", f"{side}_window_id",
                  f"{side}_focal_flight_key", f"{side}_event_epoch"]].copy()
        part.columns = ["matched_pair_id", "connected_group_id", "split", "window_id", "focal_flight_key", "event_epoch"]
        part["ATC_INTERVENTION"] = label
        rows.append(part)
    assignments = pd.concat(rows, ignore_index=True)
    assert assignments.window_id.is_unique and len(assignments) == len(windows)
    for name in ("train", "validation", "test"):
        s = assignments.loc[assignments.split == name]
        assert len(s) and set(s.ATC_INTERVENTION) == {0, 1}
        assert s.matched_pair_id.nunique() * 2 == len(s)
        assert np.array_equal(lookup.loc[s.window_id, "ATC_INTERVENTION"].to_numpy(), s.ATC_INTERVENTION.to_numpy())
    names = ("train", "validation", "test")
    for a, b in (("train", "validation"), ("validation", "test"), ("train", "test")):
        x = assignments.loc[assignments.split == a]
        y = assignments.loc[assignments.split == b]
        assert set(x.matched_pair_id).isdisjoint(y.matched_pair_id)
        assert set(x.focal_flight_key).isdisjoint(y.focal_flight_key)
        assert set(x.connected_group_id).isdisjoint(y.connected_group_id)
        assert y.event_epoch.min() - x.event_epoch.max() >= 2 * EMBARGO
    lines = ["# Stage 07 cruise split validation", "", "**PASS**: pair-disjoint, focal-flight-disjoint, connected-group-disjoint and chronologically embargoed.", "",
             f"Cutpoints: {pd.to_datetime(cut1, unit='s', utc=True)} and {pd.to_datetime(cut2, unit='s', utc=True)}.",
             f"Embargo: {EMBARGO} s on each side of each cut (at least {2*EMBARGO} s between retained splits).",
             "Groups are connected components of intervention/control focal-flight IDs joined by every matched pair; a whole group is assigned or excluded.",
             "A pair's two anchors need not be simultaneous; both anchors must fall safely within one interval. Approximate 70/15/15 is subordinate to leakage protection.", "",
             "| Split | Pairs | Windows | Unique focal flights | UTC time span |", "|---|---:|---:|---:|---|"]
    for name in (*names, "excluded_cut_or_embargo"):
        s = assignments.loc[assignments.split == name]
        span = f"{pd.to_datetime(s.event_epoch.min(), unit='s', utc=True)} to {pd.to_datetime(s.event_epoch.max(), unit='s', utc=True)}" if len(s) else "none"
        lines.append(f"| {name} | {s.matched_pair_id.nunique()} | {len(s)} | {s.focal_flight_key.nunique()} | {span} |")
    lines += ["", "The 10-minute two-sided source-window extent motivates this conservative 20-minute inter-split anchor gap. Pre-event predictor rows end at the anchor.",
              "Only focal flight IDs are present in the V2 window/pair tables; shared neighbour-flight identities across partitions cannot be asserted from these aggregate artifacts and remain a limitation."]
    return p, assignments, "\n".join(lines) + "\n"


def conditional_logit(pairs: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Pair-conditioned likelihood: L = product sigmoid(beta' (case-control))."""
    train = pairs.loc[pairs.split == "train"].copy()
    lookup = windows.set_index("window_id")
    a = lookup.loc[train.intervention_window_id, CONDITIONAL].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    b = lookup.loc[train.control_window_id, CONDITIONAL].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    complete = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[complete], b[complete]
    groups = train.connected_group_id.to_numpy()[complete]
    scale = np.std(np.vstack([a, b]), axis=0, ddof=1)
    assert len(a) > 100 and np.all(scale > 0)
    delta = (a - b) / scale
    lam = 0.1
    def objective(beta):
        eta = delta @ beta
        return -log_expit(eta).sum() + lam * np.dot(beta, beta) / 2
    def gradient(beta):
        eta = delta @ beta
        return -(delta.T @ expit(-eta)) + lam * beta
    fit = minimize(objective, np.zeros(len(CONDITIONAL)), jac=gradient, method="BFGS", options={"gtol": 1e-6})
    if not np.isfinite(fit.x).all() or np.linalg.norm(gradient(fit.x)) > 1e-3:
        raise RuntimeError("Pair-conditioned logistic optimization did not converge")
    probability = expit(delta @ fit.x)
    hessian = (delta.T * (probability * (1 - probability))) @ delta + lam * np.eye(len(CONDITIONAL))
    bread = np.linalg.inv(hessian)
    score = delta * (1 - probability)[:, None]
    cluster_scores = pd.DataFrame(score).assign(group=groups).groupby("group").sum().to_numpy()
    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    ci_lo, ci_hi = fit.x - 1.96 * se, fit.x + 1.96 * se
    result = pd.DataFrame({"feature": CONDITIONAL, "coefficient": fit.x, "odds_ratio": np.exp(fit.x),
                           "ci_lower": np.exp(ci_lo), "ci_upper": np.exp(ci_hi),
                           "direction": np.where(fit.x > 0, "positive", "negative"),
                           "p_value_exploratory": 2 * norm.sf(np.divide(np.abs(fit.x), se, out=np.full_like(se, np.nan), where=se > 0)),
                           "n_complete_train_pairs": len(a), "n_flight_connected_groups": len(np.unique(groups)),
                           "coefficient_unit": "per one train-cohort SD; pair difference", "method": "custom_pair_conditioned_logistic_likelihood_L2_0.1_cluster_sandwich"})
    return result


def make_model(kind: str):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if kind == "logistic":
        steps += [("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]
    elif kind == "tree":
        steps += [("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=80, random_state=SEED))]
    elif kind == "xgboost":
        from xgboost import XGBClassifier
        steps += [("model", XGBClassifier(n_estimators=120, max_depth=2, learning_rate=.05,
                                          subsample=.85, colsample_bytree=.85, min_child_weight=10,
                                          objective="binary:logistic", eval_metric="logloss",
                                          random_state=SEED, n_jobs=2))]
    elif kind == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        steps += [("model", ExplainableBoostingClassifier(interactions=0, max_bins=64, random_state=SEED))]
    return Pipeline(steps)


def metrics(y, score, threshold: float) -> dict:
    predicted = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
            "accuracy": accuracy_score(y, predicted), "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0),
            "specificity": tn / (tn + fp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "matched_cohort_brier": brier_score_loss(y, score)}


def choose_threshold(y, score) -> float:
    grid = np.linspace(.2, .8, 61)
    values = [balanced_accuracy_score(y, score >= t) for t in grid]
    return float(grid[int(np.argmax(values))])


def bootstrap_intervals(y, score, threshold, group, draws=200) -> dict:
    rng = np.random.default_rng(SEED)
    by_group = [np.flatnonzero(group == g) for g in np.unique(group)]
    results = {"roc_auc": [], "pr_auc": [], "f1": [], "balanced_accuracy": [], "matched_cohort_brier": []}
    for _ in range(draws):
        take = np.concatenate([by_group[i] for i in rng.integers(0, len(by_group), len(by_group))])
        row = metrics(y[take], score[take], threshold)
        for key in results:
            results[key].append(row[key])
    return {f"{key}_ci_{suffix}": float(np.quantile(values, q)) for key, values in results.items()
            for suffix, q in (("lower", .025), ("upper", .975))}


def paired_auc_delta_intervals(full_pred: pd.DataFrame, reduced_pred: pd.DataFrame,
                               draws: int = 500) -> dict:
    """Same test windows and connected-group resamples for both feature sets."""
    a = full_pred.sort_values("window_id")
    b = reduced_pred.sort_values("window_id")
    assert np.array_equal(a.window_id.to_numpy(), b.window_id.to_numpy())
    assert np.array_equal(a.ATC_INTERVENTION.to_numpy(), b.ATC_INTERVENTION.to_numpy())
    y = a.ATC_INTERVENTION.to_numpy(dtype=int)
    sa, sb = a.score.to_numpy(), b.score.to_numpy()
    g = a.connected_group_id.to_numpy()
    chunks = [np.flatnonzero(g == group_id) for group_id in np.unique(g)]
    rng = np.random.default_rng(SEED)
    roc_deltas, pr_deltas = [], []
    for _ in range(draws):
        take = np.concatenate([chunks[i] for i in rng.integers(0, len(chunks), len(chunks))])
        roc_deltas.append(roc_auc_score(y[take], sa[take]) - roc_auc_score(y[take], sb[take]))
        pr_deltas.append(average_precision_score(y[take], sa[take]) - average_precision_score(y[take], sb[take]))
    return {"delta_roc_auc_ci_lower": float(np.quantile(roc_deltas, .025)),
            "delta_roc_auc_ci_upper": float(np.quantile(roc_deltas, .975)),
            "delta_pr_auc_ci_lower": float(np.quantile(pr_deltas, .025)),
            "delta_pr_auc_ci_upper": float(np.quantile(pr_deltas, .975)),
            "paired_bootstrap_connected_groups": draws}


def importance(model, columns, X_val, y_val, folder: Path, kind: str) -> pd.DataFrame:
    permutation = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=5,
                                         random_state=SEED, n_jobs=1)
    ranked = pd.DataFrame({"feature": columns, "importance_mean": permutation.importances_mean,
                           "importance_std": permutation.importances_std, "method": "validation_permutation_roc_auc"})
    ranked = ranked.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(folder / "permutation_feature_importance.csv", index=False)
    if kind == "logistic":
        coefficients = model.named_steps["model"].coef_[0]
        names = model.named_steps["imputer"].get_feature_names_out(columns)
        pd.DataFrame({"feature": names, "standardized_coefficient": coefficients,
                      "odds_ratio_per_scaled_unit": np.exp(coefficients)}).sort_values(
                          "standardized_coefficient", key=np.abs, ascending=False).to_csv(folder / "coefficients.csv", index=False)
    elif kind == "tree":
        names = list(model.named_steps["imputer"].get_feature_names_out(columns))
        tree = model.named_steps["model"]
        (folder / "rules.txt").write_text(export_text(tree, feature_names=names), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(tree, feature_names=names, class_names=["control", "intervention"],
                  filled=True, rounded=True, fontsize=8, ax=ax)
        fig.tight_layout()
        fig.savefig(folder / "tree.png", dpi=130)
        plt.close(fig)
    elif kind == "ebm":
        terms = model.named_steps["model"].term_names_
        values = model.named_steps["model"].term_importances()
        pd.DataFrame({"feature": terms, "importance": values}).sort_values("importance", ascending=False).to_csv(folder / "global_importance.csv", index=False)
        for idx in np.argsort(values)[-3:]:
            explanation = model.named_steps["model"].explain_global()
            data = explanation.data(int(idx))
            pd.DataFrame({"name": data.get("names", []), "score": data.get("scores", [])}).to_csv(
                folder / f"shape_{idx}.csv", index=False)
    return ranked


def train_evaluate(events: pd.DataFrame, family: str, columns: list[str], kind: str,
                   save_detail: bool = True) -> tuple[dict, pd.DataFrame, pd.DataFrame, object]:
    subsets = {name: events.loc[events.split == name].copy() for name in ("train", "validation", "test")}
    x = {name: frame[columns].apply(pd.to_numeric, errors="coerce") for name, frame in subsets.items()}
    y = {name: frame.ATC_INTERVENTION.to_numpy(dtype=int) for name, frame in subsets.items()}
    model = make_model(kind)
    model.fit(x["train"], y["train"])
    val_score = model.predict_proba(x["validation"])[:, 1]
    test_score = model.predict_proba(x["test"])[:, 1]
    threshold = choose_threshold(y["validation"], val_score)
    row = {"model": kind, "feature_family": family, "n_features": len(columns),
           "n_train": len(subsets["train"]), "n_validation": len(subsets["validation"]),
           "n_test": len(subsets["test"]), "threshold_from_validation": threshold,
           "validation_roc_auc": roc_auc_score(y["validation"], val_score),
           "validation_pr_auc": average_precision_score(y["validation"], val_score),
           "validation_f1": f1_score(y["validation"], val_score >= threshold),
           **metrics(y["test"], test_score, threshold),
           **bootstrap_intervals(y["test"], test_score, threshold,
                                 subsets["test"].connected_group_id.to_numpy())}
    predicted = subsets["test"][["window_id", "matched_pair_id", "connected_group_id",
                                   "focal_flight_key", "event_epoch", "ATC_INTERVENTION",
                                   "clearance_id", "control_for_clearance_id"]].copy()
    predicted["model"] = kind
    predicted["feature_family"] = family
    predicted["score"] = test_score
    predicted["threshold"] = threshold
    predicted["predicted_label"] = (test_score >= threshold).astype(int)
    ranking = pd.DataFrame(columns=["feature", "importance_mean", "rank"])
    if save_detail:
        folder = OUT / kind / family.lower()
        folder.mkdir(parents=True, exist_ok=True)
        ranking = importance(model, columns, x["validation"], y["validation"], folder, kind)
        (folder / "validation_selection.json").write_text(json.dumps(
            {"threshold": threshold, "selection_set": "validation", "threshold_metric": "balanced_accuracy",
             "importance_set": "validation", "importance_metric": "ROC-AUC", "n_features": len(columns)}, indent=2))
    return row, predicted, ranking.assign(model=kind, feature_family=family), model


def consensus_table(conditional: pd.DataFrame, rankings: pd.DataFrame,
                    logistic_models: dict, family_features: dict) -> pd.DataFrame:
    cond_dir = dict(zip(conditional.feature, conditional.direction))
    stage06 = pd.read_csv(ROOT / "06_exploration" / "cruise_matching_v1_v2_comparison.csv")
    stage06_smd = dict(zip(stage06.feature, stage06.V2_SMD))
    rows = []
    for feature in FEATURES:
        smd = stage06_smd.get(feature, np.nan)
        descriptive_direction = "negative" if np.isfinite(smd) and smd < 0 else "positive" if np.isfinite(smd) and smd > 0 else "unavailable"
        row = {"feature": feature, "conditional_logistic_direction": cond_dir.get(feature, "not_in_compact_conditional_set"),
               "stage06_descriptive_direction": descriptive_direction}
        for kind in ("logistic", "tree", "ebm", "xgboost"):
            r = rankings.loc[(rankings.model == kind) & (rankings.feature_family == "SCAT_FULL") & (rankings.feature == feature)]
            row[f"{kind}_rank"] = int(r.iloc[0]["rank"]) if len(r) else np.nan
        for family in ("SCAT_FULL", "OPENSKY_COMPATIBLE"):
            r = rankings.loc[(rankings.feature_family == family) & (rankings.feature == feature)]
            row[f"{family}_rank"] = int(r["rank"].min()) if len(r) else np.nan
        directions = []
        for family, model in logistic_models.items():
            if feature in family_features[family]:
                names = list(model.named_steps["imputer"].get_feature_names_out(family_features[family]))
                directions.append(np.sign(model.named_steps["model"].coef_[0][names.index(feature)]))
        cond_sign = {"positive": 1, "negative": -1}.get(cond_dir.get(feature))
        if cond_sign is not None:
            directions.append(cond_sign)
        if descriptive_direction != "unavailable" and FEATURES[feature][0] != "context":
            directions.append(1 if descriptive_direction == "positive" else -1)
        row["direction_consistent"] = bool(len(directions) >= 2 and len(set(directions)) == 1)
        available_ranks = [row[f"{kind}_rank"] for kind in ("logistic", "tree", "ebm", "xgboost")]
        row["robust_across_models"] = bool(sum(np.isfinite(r) and r <= 5 for r in available_ranks) >= 2
                                           and row["direction_consistent"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["robust_across_models", "SCAT_FULL_rank"], ascending=[False, True])


def high_performance_audit(pairs: pd.DataFrame, events: pd.DataFrame, metrics_frame: pd.DataFrame) -> str:
    alarm = metrics_frame.loc[metrics_frame.roc_auc > .95]
    if alarm.empty:
        return "No model exceeded the predeclared ROC-AUC > 0.95 leakage-alarm trigger."
    duplicate_features = events.duplicated(subset=list(FEATURES)).sum()
    unique_windows = events.window_id.is_unique
    checks = ["# High-performance leakage audit", "", f"Trigger: {len(alarm)} model(s) with test ROC-AUC > 0.95.",
              f"Unique windows: {unique_windows}; exact duplicate full feature vectors: {duplicate_features}.",
              "Allowed predictor set was fixed in source code before fitting; clearance, IDs, future-clearance, cluster, matching and colour fields are excluded.",
              "Matched-pair, focal-flight, connected-group and chronological embargo assertions passed before fitting.",
              "TopSky uses source-state predicted routes, but this export does not expose a separate route-publication timestamp; residual provenance uncertainty remains.",
              "Treat high performance as provisional until independent-week and route-publication-time verification."]
    (OUT / "high_performance_leakage_audit.md").write_text("\n".join(checks) + "\n")
    return "Leakage alarm triggered; see high_performance_leakage_audit.md."


def format_metric(row: pd.Series | None) -> str:
    if row is None:
        return "SKIPPED"
    return f"ROC-AUC {row.roc_auc:.3f}, PR-AUC {row.pr_auc:.3f}, F1 {row.f1:.3f}, balanced accuracy {row.balanced_accuracy:.3f}, matched-cohort Brier {row.matched_cohort_brier:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only this Stage 07 output folder/report")
    args = parser.parse_args()
    if OUT.exists() and (OUT / "model_metrics.csv").exists() and not args.overwrite:
        raise SystemExit(f"Stage 07 output already exists: {OUT}; preserve it and inspect before rerun.")
    pairs, windows = audit_inputs()
    mapping = pd.read_csv(MAPPING_PATH)
    families = feature_sets(mapping)
    pairs, assignments, split_report = split_pairs(pairs, windows)
    # No model fit before this point; all split assertions must pass.
    OUT.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(OUT / "split_assignments.parquet", index=False)
    (OUT / "split_validation_report.md").write_text(split_report, encoding="utf-8")
    pairs[["matched_pair_id", "connected_group_id", "split", "intervention_window_id",
           "control_window_id"]].to_parquet(OUT / "pair_split_assignments.parquet", index=False)
    set_rows = [{"feature": f, "physical_family": group, "availability_class": available,
                 **{name: f in columns for name, columns in families.items()},
                 "Stage06_quality_basis": "predeclared bounded/pre-event physical or context feature"}
                for f, (group, available) in FEATURES.items()]
    pd.DataFrame(set_rows).to_csv(OUT / "feature_sets.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({
        "scope": "FIRST_WEEK_PILOT", "week": PILOT_WEEK, "band": BAND,
        "cohort": "CRUISE_MATCHING_V2", "target": "ATC_INTERVENTION",
        "pairs_path": str(PAIRS_PATH), "features_path": str(WINDOWS_PATH),
        "pairs": len(pairs), "windows": len(windows), "embargo_seconds_each_cut": EMBARGO,
        "split_cut_quantiles": [.70, .85], "random_seed": SEED, "feature_sets": families,
        "threshold_selection": "max_validation_balanced_accuracy_grid_0.20_to_0.80_step_0.01",
        "feature_blacklist": BLACKLIST, "prevalence_notice": NOTICE,
        "conditional_logistic_scope": "train_complete_pairs_only", "test_used_for_feature_selection": False,
        "matching_variables": "anchor phase, centre, UTC hour, speed, altitude, Stage04 density; not selected as interaction signals",
        "topsky_publication_timestamp_verified": False}, indent=2), encoding="utf-8")
    events = assignments.merge(windows, on="window_id", validate="one_to_one", suffixes=("", "_source"))
    assert np.array_equal(events.ATC_INTERVENTION.to_numpy(), events.ATC_INTERVENTION_source.to_numpy())
    assert np.allclose(events.event_epoch.to_numpy(), events.event_epoch_source.to_numpy())
    conditional = conditional_logit(pairs, windows)
    conditional.to_csv(OUT / "conditional_logistic_results.csv", index=False)
    print(f"Stage 07: cohort/split PASS; train/validation/test pairs "
          f"{[(assignments.split == s).sum()//2 for s in ('train','validation','test')]}; "
          f"conditional complete train pairs {int(conditional.n_complete_train_pairs.iloc[0])}.", flush=True)

    for kind in ("logistic", "tree", "ebm", "xgboost"):
        (OUT / kind).mkdir(exist_ok=True)
    optional = {}
    for kind, package in (("ebm", "interpret"), ("xgboost", "xgboost")):
        try:
            __import__(package)
            optional[kind] = True
        except (ImportError, OSError) as exc:
            optional[kind] = False
            (OUT / kind / "SKIPPED.md").write_text(f"Not run: {package} unavailable ({type(exc).__name__}).\n")
    kinds = ["logistic", "tree"] + [k for k in ("ebm", "xgboost") if optional[k]]
    results, predictions, ranks = [], [], []
    fitted_logistic = {}
    full_models = {}
    for family in ("SCAT_FULL", "OPENSKY_COMPATIBLE", "MINIMAL_PHYSICAL"):
        for kind in kinds:
            row, pred, rank, model = train_evaluate(events, family, families[family], kind)
            results.append(row)
            predictions.append(pred)
            ranks.append(rank)
            if kind == "logistic":
                fitted_logistic[family] = model
            if family == "SCAT_FULL":
                full_models[kind] = model
            print(f"Stage 07: {family} {kind} complete; test ROC-AUC {row['roc_auc']:.3f}.", flush=True)
    metrics_frame = pd.DataFrame(results)
    metrics_frame.to_csv(OUT / "model_metrics.csv", index=False)
    metrics_frame.to_csv(OUT / "model_family_comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / "test_predictions.parquet", index=False)
    rankings = pd.concat(ranks, ignore_index=True)
    consensus = consensus_table(conditional, rankings, fitted_logistic, families)
    consensus.to_csv(OUT / "feature_consensus.csv", index=False)

    # TopSky ablation preserves all other SCAT_FULL predictors and model settings.
    ablation_topsky = []
    for kind in kinds:
        baseline = metrics_frame.loc[(metrics_frame.feature_family == "SCAT_FULL") & (metrics_frame.model == kind)].iloc[0]
        reduced, reduced_pred, rank, _ = train_evaluate(events, "FULL_WITHOUT_TOPSKY", families["FULL_WITHOUT_TOPSKY"], kind)
        baseline_pred = next(pred for pred in predictions
                             if pred.feature_family.iloc[0] == "SCAT_FULL" and pred.model.iloc[0] == kind)
        ablation_topsky.append({"model": kind, "with_topsky_roc_auc": baseline.roc_auc,
                               "without_topsky_roc_auc": reduced["roc_auc"],
                               "delta_roc_auc": baseline.roc_auc - reduced["roc_auc"],
                               "with_topsky_pr_auc": baseline.pr_auc,
                               "without_topsky_pr_auc": reduced["pr_auc"],
                               "delta_pr_auc": baseline.pr_auc - reduced["pr_auc"],
                               "with_topsky_top5": ", ".join(rankings.loc[(rankings.model == kind) & (rankings.feature_family == "SCAT_FULL")].nsmallest(5, "rank").feature),
                               "without_topsky_top5": ", ".join(rank.nsmallest(5, "rank").feature),
                               **paired_auc_delta_intervals(baseline_pred, reduced_pred)})
    topsky = pd.DataFrame(ablation_topsky)
    topsky.to_csv(OUT / "topsky_ablation.csv", index=False)
    print("Stage 07: TopSky ablation complete.", flush=True)

    family_ablation = []
    base = metrics_frame.loc[(metrics_frame.feature_family == "OPENSKY_COMPATIBLE") & (metrics_frame.model == "logistic")].iloc[0]
    for removed in ("separation", "cpa_tcpa", "closing", "geometry", "density_neighbours"):
        columns = [f for f in families["OPENSKY_COMPATIBLE"] if FEATURES[f][0] != removed]
        result, _, _, _ = train_evaluate(events, f"WITHOUT_{removed.upper()}", columns, "logistic", save_detail=False)
        family_ablation.append({"model": "logistic", "removed_family": removed, "n_features": len(columns),
                                "base_roc_auc": base.roc_auc, "ablated_roc_auc": result["roc_auc"],
                                "delta_roc_auc": base.roc_auc - result["roc_auc"],
                                "base_pr_auc": base.pr_auc, "ablated_pr_auc": result["pr_auc"],
                                "delta_pr_auc": base.pr_auc - result["pr_auc"]})
    pd.DataFrame(family_ablation).to_csv(OUT / "feature_family_ablation.csv", index=False)
    print("Stage 07: five predeclared physical-family ablations complete.", flush=True)

    # Validation chooses the explanatory primary model. Test is held out for reporting only.
    primary = metrics_frame.loc[metrics_frame.feature_family == "SCAT_FULL"].sort_values(
        ["validation_roc_auc", "validation_pr_auc"], ascending=False).iloc[0]
    all_predictions = pd.concat(predictions, ignore_index=True)
    primary_pred = all_predictions.loc[(all_predictions.feature_family == "SCAT_FULL") &
                                       (all_predictions.model == primary.model)].copy()
    primary_pred["error_type"] = np.select(
        [(primary_pred.ATC_INTERVENTION == 0) & (primary_pred.predicted_label == 1),
         (primary_pred.ATC_INTERVENTION == 1) & (primary_pred.predicted_label == 0)],
        ["false_positive_ATC_intervention_classification", "false_negative_ATC_intervention_classification"],
        default="correct")
    examples = primary_pred.loc[primary_pred.error_type != "correct"].merge(
        windows[["window_id", "analysis_altitude_ft", "ground_speed_at_anchor_kt",
                 "min_horizontal_sep_within_30nm", "vertical_sep_at_event",
                 "min_future_cpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
                 "n_unique_neighbours_within_30nm", "min_topsky_predicted_sep_nm_proxy"]],
        on="window_id", validate="one_to_one")
    examples.sort_values("score").to_csv(OUT / "error_analysis.csv", index=False)
    examples.loc[examples.error_type.str.startswith("false_positive")].nlargest(20, "score").to_csv(
        OUT / "false_positive_examples.csv", index=False)
    examples.loc[examples.error_type.str.startswith("false_negative")].nsmallest(20, "score").to_csv(
        OUT / "false_negative_examples.csv", index=False)
    intervention = primary_pred.loc[primary_pred.ATC_INTERVENTION == 1].copy()
    clearance_meta = pd.read_parquet(ROOT / "03_clearances" / "scat_clearances.parquet",
                                     columns=["clearance_id", "archive_week", "clearance_type"])
    clearance_meta = clearance_meta.loc[clearance_meta.archive_week == PILOT_WEEK,
                                        ["clearance_id", "clearance_type"]]
    assert clearance_meta.clearance_id.is_unique
    intervention["clearance_type"] = intervention.clearance_id.map(
        clearance_meta.set_index("clearance_id").clearance_type)
    intervention["clearance_type"] = intervention.clearance_type.fillna("UNKNOWN")
    intervention.groupby("clearance_type", dropna=False).agg(
        n=("window_id", "size"), mean_score=("score", "mean"),
        median_score=("score", "median"), recall=("predicted_label", "mean")).reset_index().to_csv(
            OUT / "clearance_type_posthoc.csv", index=False)
    leakage_status = high_performance_audit(pairs, events, metrics_frame)

    full = metrics_frame.loc[(metrics_frame.feature_family == "SCAT_FULL") & (metrics_frame.model == primary.model)].iloc[0]
    without = metrics_frame.loc[(metrics_frame.feature_family == "OPENSKY_COMPATIBLE") &
                                (metrics_frame.model == primary.model)].iloc[0]
    opensky_best = metrics_frame.loc[metrics_frame.feature_family == "OPENSKY_COMPATIBLE"].sort_values(
        ["validation_roc_auc", "validation_pr_auc"], ascending=False).iloc[0]
    top_transfer = consensus.loc[consensus.feature.isin(families["OPENSKY_COMPATIBLE"])].nsmallest(5, "OPENSKY_COMPATIBLE_rank").feature.tolist()
    gap = full.roc_auc - without.roc_auc
    signal = "STRONG" if opensky_best.roc_auc_ci_lower > .65 else "MODERATE" if opensky_best.roc_auc_ci_lower > .5 else "WEAK" if opensky_best.roc_auc > .5 else "NONE"
    # Signal descriptors are descriptive; no a-priori acceptable Full-vs-OpenSky gap is imposed.
    transfer_decision = "YES" if opensky_best.roc_auc_ci_lower > .5 else "UNDETERMINED"
    next_gate = "READY_FOR_COUNTERFACTUAL_ANALYSIS" if full.roc_auc_ci_lower > .5 and opensky_best.roc_auc_ci_lower > .5 else "REVISE_STAGE07"
    if full.roc_auc <= .5 and opensky_best.roc_auc <= .5:
        next_gate = "INSUFFICIENT_PREDICTIVE_SIGNAL"
    if "triggered" in leakage_status:
        next_gate = "REVISE_STAGE07"
    write_report(pairs, assignments, conditional, metrics_frame, primary, topsky,
                 pd.DataFrame(family_ablation), consensus, examples, full, without,
                 opensky_best, top_transfer, gap, signal, transfer_decision, next_gate,
                 optional, leakage_status)
    terminal_summary(assignments, metrics_frame, primary, full, without, opensky_best,
                     top_transfer, gap, signal, transfer_decision, next_gate, consensus)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    selected = frame.loc[:, columns].head(max_rows).fillna("—")
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def write_report(pairs, assignments, conditional, metrics_frame, primary, topsky,
                 ablations, consensus, examples, full, without, opensky_best,
                 top_transfer, gap, signal, transfer_decision, next_gate, optional,
                 leakage_status) -> None:
    def metric_row(family, model):
        found = metrics_frame.loc[(metrics_frame.feature_family == family) & (metrics_frame.model == model)]
        return found.iloc[0] if len(found) else None
    full_table = metrics_frame[["feature_family", "model", "n_features", "roc_auc", "pr_auc", "f1",
                                "balanced_accuracy", "matched_cohort_brier", "roc_auc_ci_lower", "roc_auc_ci_upper"]]
    top_consensus = consensus.loc[consensus.robust_across_models]
    conditional_view = conditional[["feature", "coefficient", "odds_ratio", "ci_lower", "ci_upper",
                                    "direction", "p_value_exploratory"]]
    clearance = pd.read_csv(OUT / "clearance_type_posthoc.csv")
    split_lines = []
    for name in ("train", "validation", "test", "excluded_cut_or_embargo"):
        s = assignments.loc[assignments.split == name]
        split_lines.append(f"- {name}: {s.matched_pair_id.nunique()} pairs, {len(s)} windows, "
                           f"{s.focal_flight_key.nunique()} focal flights; "
                           f"{pd.to_datetime(s.event_epoch.min(), unit='s', utc=True)} to "
                           f"{pd.to_datetime(s.event_epoch.max(), unit='s', utc=True)}")
    sections = [
        "# SCAT Stage 07 — FL290–FL410 intervention explainability",
        "",
        "## 1. Scope", "",
        f"Only SCAT week `{PILOT_WEEK}`, band FL290–FL410 and `CRUISE_MATCHING_V2`. "
        "Stages 01–06 were not rerun; Stages 08–10 and other weeks were not executed. "
        "Input: [V2 pairs](04_windows/cruise_matching_v2.parquet) and "
        "[V2 features](05_features/window_features_cruise_matching_v2.parquet).", "",
        "## 2. Target semantics", "",
        "The sole target is `ATC_INTERVENTION`: a real recorded controller intervention versus a comparable non-intervention. "
        "No confirmed conflict, loss-of-separation or colour-state label was created. The models discriminate intervention, not conflict.", "",
        "## 3. CRUISE_MATCHING_V2 cohort", "",
        "Verified exactly **4,098** unique one-to-one pairs, **8,196** unique windows, and 4,098 observations in each target class. "
        "All have first-week provenance, cruise-band altitude and nonpositive predictor anchors. Exact pair IDs, flight keys and anchor epochs were reconciled.", "",
        "## 4. Matched-design implications", "",
        "The 50% intervention prevalence is artificial. ROC-AUC is interpretable as within-cohort discrimination; the PR-AUC reference is approximately 0.50. "
        "Brier and threshold metrics measure matched-cohort fit only. Odds ratios are associations, not causal effects. " + NOTICE, "",
        "## 5. Feature blacklist", "",
        "Predictors are selected by an explicit allowlist. Excluded: intervention target; window/pair/flight/event/archive IDs; "
        "matching metadata; all clearance information and post-clearance/future flight-plan state; colour states; "
        "Stage-06 clusters; raw linear bearing; unstable positive-TCPA and unbounded global extrema. "
        "[Full blacklist and configuration](07_explainability_cruise/run_config.json). "
        "TopSky source route points were attached to pre-anchor state rows and filtered by future ETO, but a separate route-publication timestamp is absent; this unresolved provenance caveat limits causal and operational interpretation.", "",
        "## 6. Feature sets", "",
        "`SCAT_FULL` uses 19 predeclared numeric pre-event features including TopSky; `OPENSKY_COMPATIBLE` removes SCAT-only TopSky and retains only fields mapped as available/derivable; "
        "`MINIMAL_PHYSICAL` uses eight fixed physical features. Derived future CPA and circular sine/cosine inherit the OpenSky class of their source primitives. "
        "[Exact membership](07_explainability_cruise/feature_sets.csv). Context speed/altitude are matched covariates, not evidence of a traffic-interaction signal. "
        "No feature was added/removed after inspecting test performance. Missing neighbour-within-30-NM quantities remain missing and are accompanied by train-fitted missing indicators.", "",
        "## 7. Temporal/group split methodology", "", *split_lines, "",
        "Connected components join any matched pairs sharing a focal flight. Each entire component is assigned or excluded using chronological cutpoints at the 70th/85th percentiles of anchor times. "
        "A 600-second exclusion on **each** side of both cutpoints enforces at least 1,200 seconds between retained partitions. The pair's two anchors must both lie in its partition. "
        "Exact 70/15/15 was not forced.", "",
        "## 8. Leakage validation", "",
        "[Split validation report](07_explainability_cruise/split_validation_report.md): **PASS** for pair, focal-flight and connected-group disjointness across train/validation/test, "
        "with temporal embargo. The `split_assignments.parquet` records every window, including exclusions. " + leakage_status + " "
        "Neighbour-flight identity overlap cannot be assessed from the aggregate window table.", "",
        "## 9. Conditional logistic results", "",
        "The installed environment lacks `statsmodels`; a transparent SciPy pair-conditioned likelihood was fitted on **train complete pairs only**, "
        "using case-minus-control standardized differences, weak L2 penalty (0.1), and flight-connected-group sandwich intervals. "
        "Each odds ratio is per one training-cohort SD increase within a pair; p-values are exploratory, unadjusted and not causal. "
        "This is the conditional likelihood for one case and one control per stratum, implemented locally rather than claiming a library fit.", "",
        markdown_table(conditional_view, list(conditional_view.columns)), "",
        "## 10. Logistic regression", "",
        f"SCAT_FULL: {format_metric(metric_row('SCAT_FULL','logistic'))}. "
        "L2 regularization, median imputation and standardization were fitted on train only. Missing indicators preserve no-neighbour absence. "
        "[Standardized coefficients](07_explainability_cruise/logistic/scat_full/coefficients.csv) and validation permutation importance are provided. "
        "Ordinary logistic coefficients are multivariable predictive associations, not matched conditional odds ratios.", "",
        "## 11. Decision tree", "",
        f"SCAT_FULL: {format_metric(metric_row('SCAT_FULL','tree'))}. "
        "Depth 3 and at least 80 samples per leaf keep the model readable. "
        "[Rules](07_explainability_cruise/tree/scat_full/rules.txt), "
        "[tree figure](07_explainability_cruise/tree/scat_full/tree.png) and validation permutation ranks are supplied.", "",
        "## 12. EBM", "",
        (f"SCAT_FULL: {format_metric(metric_row('SCAT_FULL','ebm'))}. "
         "Global term importance and top shape-function tables are in the EBM folder." if optional["ebm"]
         else "Skipped: `interpret` is not installed; no package was installed for this run."), "",
        "## 13. XGBoost and SHAP/permutation", "",
        (f"SCAT_FULL: {format_metric(metric_row('SCAT_FULL','xgboost'))}. "
         "SHAP is unavailable in this environment, so the actual validation-based method is labeled "
         "[permutation_feature_importance.csv](07_explainability_cruise/xgboost/scat_full/permutation_feature_importance.csv), not SHAP. "
         "Model complexity: 120 trees, depth 2, learning rate 0.05; no broad search." if optional["xgboost"]
         else "Skipped: XGBoost could not be imported; no package was installed."), "",
        "## 14. Model comparison", "",
        markdown_table(full_table, list(full_table.columns)), "",
        "All figures are on the held-out test. Threshold-dependent scores use a validation-selected balanced-accuracy threshold. "
        "ROC-AUC, PR-AUC, F1, balanced accuracy and matched-cohort Brier have 200-replicate flight-connected-group bootstrap intervals; "
        "[full metrics](07_explainability_cruise/model_metrics.csv) also contain accuracy, precision, recall, specificity and the confusion matrix. "
        "Only the validation set chose the primary explanatory model and threshold.", "",
        "## 15. Feature consensus", "",
        f"{len(top_consensus)} features meet the descriptive consensus flag (top-five validation permutation rank in at least two model types plus consistent available Stage-06 contrast/logistic/conditional direction). "
        "This is within-one-week method agreement, not independent replication. "
        "[Full consensus](07_explainability_cruise/feature_consensus.csv).", "",
        markdown_table(consensus[["feature", "stage06_descriptive_direction", "conditional_logistic_direction", "logistic_rank", "tree_rank",
                                  "xgboost_rank", "SCAT_FULL_rank", "OPENSKY_COMPATIBLE_rank",
                                  "direction_consistent", "robust_across_models"]],
                       ["feature", "stage06_descriptive_direction", "conditional_logistic_direction", "logistic_rank", "tree_rank",
                        "xgboost_rank", "SCAT_FULL_rank", "OPENSKY_COMPATIBLE_rank",
                        "direction_consistent", "robust_across_models"], 10), "",
        "Interpretation: the Stage-06 robust bounded horizontal separation remains negatively associated with intervention in the train-only pair-conditioned fit, "
        "but it is not a top-five feature in two model types. The Stage-06 robust future CPA does **not** retain a clear independent association "
        "after the compact paired adjustment (its odds-ratio interval crosses 1); a positive ordinary-logistic coefficient in SCAT_FULL underscores redundancy/correlation. "
        "Vertical separation, TopSky and some closing/geometry summaries contribute to discrimination; any apparent direction changes are reported, not hidden. "
        "The model-consensus flag is descriptive and does not erase Stage-06 physical-quality caveats or multiple-testing uncertainty.", "",
        "## 16. TopSky ablation", "",
        markdown_table(topsky, ["model", "with_topsky_roc_auc", "without_topsky_roc_auc", "delta_roc_auc",
                                "delta_roc_auc_ci_lower", "delta_roc_auc_ci_upper",
                                "with_topsky_pr_auc", "without_topsky_pr_auc", "delta_pr_auc"]), "",
        "The comparison holds every other feature and model setting fixed. The CSV includes 500-replicate **paired** focal-flight-connected-group bootstrap intervals for the AUC/PR-AUC differences. "
        "Positive delta means TopSky helped on this test, not that the effect is causal or that kinematic CPA can replace TopSky. "
        "[Feature-rank shifts](07_explainability_cruise/topsky_ablation.csv) are also recorded.", "",
        "## 17. OpenSky-compatible model", "",
        f"SCAT FULL ({primary.model}): {format_metric(full)}.  ",
        f"SCAT WITHOUT TOPSKY ({primary.model}): {format_metric(without)}.  ",
        f"OPENSKY-COMPATIBLE (validation-selected {opensky_best.model}): {format_metric(opensky_best)}.  ",
        f"Same-model Full minus OpenSky ROC-AUC gap: **{gap:+.3f}** "
        f"(paired connected-group 95% interval {topsky.loc[topsky.model == primary.model, 'delta_roc_auc_ci_lower'].iloc[0]:+.3f} to "
        f"{topsky.loc[topsky.model == primary.model, 'delta_roc_auc_ci_upper'].iloc[0]:+.3f}); "
        "no acceptability threshold was imposed in advance. "
        "Because the only SCAT-only predictor retained in this predeclared numeric set is TopSky, Full-without-TopSky equals the OpenSky-compatible feature set. "
        "This equality does not mean the underlying OpenSky extraction is already quality-valid.", "",
        "Top OpenSky-compatible validation-ranked features: " + ", ".join(f"`{f}`" for f in top_transfer) + ".", "",
        f"OpenSky-compatible discrimination signal: **{signal}** (descriptive descriptor). Proceed with a bounded FL290–FL410 OpenSky *feature-yield/quality* pilot: **{transfer_decision}**; do not infer transferable labels or population calibration.", "",
        "## 18. Feature-family ablation", "",
        markdown_table(ablations, ["removed_family", "n_features", "base_roc_auc", "ablated_roc_auc",
                                   "delta_roc_auc", "base_pr_auc", "ablated_pr_auc", "delta_pr_auc"]), "",
        "Five one-at-a-time logistic ablations were predeclared; changes are descriptive and correlated families may substitute for each other. "
        "No combinatorial search or test-guided feature selection was performed. Interactions were not added because the objective here is a compact, auditable explanation; "
        "tree/XGBoost can represent limited nonlinearities without creating hundreds of manual products.", "",
        "## 19. Clearance-type post-hoc analysis", "",
        "Clearance type was joined only **after** test predictions from the existing Stage-03 clearance table; it was never a predictor. "
        "Recall here is intervention-window classification sensitivity by recorded type, not conflict sensitivity.", "",
        markdown_table(clearance, ["clearance_type", "n", "mean_score", "median_score", "recall"]), "",
        "## 20. Error analysis", "",
        f"Validation-selected SCAT_FULL `{primary.model}` has {int(((examples.ATC_INTERVENTION == 0) & (examples.predicted_label == 1)).sum())} false-positive "
        f"and {int(((examples.ATC_INTERVENTION == 1) & (examples.predicted_label == 0)).sum())} false-negative ATC-intervention classifications in test. "
        "[All error rows](07_explainability_cruise/error_analysis.csv), [20 highest-scoring false positives](07_explainability_cruise/false_positive_examples.csv) "
        "and [20 lowest-scoring false negatives](07_explainability_cruise/false_negative_examples.csv) include IDs, UTC anchor, flight, physical context and score. "
        "These are classification errors relative to intervention, not false or missed conflicts.", "",
        "## 21. Limitations", "",
        "One week, one altitude stratum, a reused Stage-04 control pool and a matched 50/50 sample constrain generalization. "
        "Some physical features still have asymmetric missingness, neighbour-selection effects and Stage-05 alignment warnings. "
        "TopSky publication time is not separately stamped. Groups cover focal flights, not all shared neighbour flights. "
        "Group-bootstrap intervals capture dependence within retained connected groups, not independent-week uncertainty. "
        "Neither the propensity for a clearance in the population nor a causal intervention effect is estimated.", "",
        "## 22. Implications for OpenSky", "",
        f"A bounded OpenSky quality/yield pilot is **{transfer_decision}** on the basis of held-out matched-cohort signal, "
        "but OpenSky has no native intervention labels, controller clearance fields or TopSky predicted routes. "
        "Transfer requires quality-valid neutral reference times, contemporary traffic coverage, same-neighbour rules, barometric-altitude checks and separately derived kinematic CPA. "
        "The observed Full-versus-Compatible gap must be reported as measured, without assuming CPA replaces TopSky. No OpenSky processing occurred here.", "",
        "## 23. Recommendation for counterfactual Stage 09", "",
        f"**{next_gate}**. This is a Stage-07 recommendation only, not authorization to execute Stage 08, 09 or 10. "
        "The historical Stage 08 is a dependency screen, not a Bayesian Network. "
        "A later operational/counterfactual validation is required before asking whether any case is a confirmed flight conflict.", "",
    ]
    REPORT.write_text("\n".join(sections), encoding="utf-8")


def terminal_summary(assignments, metrics_frame, primary, full, without, opensky_best,
                     top_transfer, gap, signal, transfer_decision, next_gate, consensus) -> None:
    lines = ["=" * 60, "SCAT STAGE 07 — FL290–FL410 EXPLAINABILITY", "=" * 60,
             f"Week: {PILOT_WEEK}", "Cohort: CRUISE_MATCHING_V2", "Pairs: 4098", "Windows: 8196", "Split leakage: PASS"]
    for name in ("train", "validation", "test"):
        part = assignments.loc[assignments.split == name]
        lines += [f"{name.title()}: pairs = {part.matched_pair_id.nunique()}, flights = {part.focal_flight_key.nunique()}, "
                  f"time range = {pd.to_datetime(part.event_epoch.min(), unit='s', utc=True)} to "
                  f"{pd.to_datetime(part.event_epoch.max(), unit='s', utc=True)}"]
    lines += ["-" * 60, "MODEL PERFORMANCE", "SCAT FULL"]
    for kind in ("logistic", "tree", "ebm", "xgboost"):
        row = metrics_frame.loc[(metrics_frame.feature_family == "SCAT_FULL") & (metrics_frame.model == kind)]
        lines.append(f"{kind}: ROC-AUC = {row.iloc[0].roc_auc:.3f}, PR-AUC = {row.iloc[0].pr_auc:.3f}" if len(row) else f"{kind}: SKIPPED")
    lines += ["OPENSKY COMPATIBLE", f"Best model (validation): {opensky_best.model}",
              f"ROC-AUC: {opensky_best.roc_auc:.3f}", f"PR-AUC: {opensky_best.pr_auc:.3f}",
              "-" * 60, "TOP ROBUST FEATURES"]
    robust = consensus.loc[consensus.robust_across_models, "feature"].head(5).tolist()
    lines += [f"{i}. {feature}" for i, feature in enumerate(robust, 1)] or ["None met the cross-model consensus rule."]
    lines += ["-" * 60, "TOPSKY ABLATION", f"With TopSky ({primary.model}): ROC-AUC {full.roc_auc:.3f}",
              f"Without TopSky ({primary.model}): ROC-AUC {without.roc_auc:.3f}",
              f"Impact: {gap:+.3f} ROC-AUC", "-" * 60, "OPENSKY TRANSFER",
              f"OpenSky-compatible signal: {signal}",
              "Recommended OpenSky features: " + ", ".join(top_transfer),
              f"Proceed with bounded FL290–410 OpenSky quality pilot: {transfer_decision}",
              "-" * 60, "INTERPRETATION", "The model predicts: ATC_INTERVENTION in a matched analytical cohort",
              "NOT: confirmed conflict", "NOT: population intervention probability", NOTICE,
              "-" * 60, "NEXT", next_gate, "DO NOT RUN STAGE 08/09/10 AUTOMATICALLY.",
              "DO NOT PROCESS OTHER SCAT WEEKS.", "=" * 60]
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

