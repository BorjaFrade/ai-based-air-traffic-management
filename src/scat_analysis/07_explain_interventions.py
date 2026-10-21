#!/usr/bin/env python3
"""Explain and temporally evaluate prediction of ATC_INTERVENTION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from common import DEFAULT_OUTPUT_ROOT, update_read_me_first


FEATURE_COLUMNS = [
    "horizontal_separation_nm", "vertical_separation_ft", "relative_velocity_kt",
    "horizontal_closing_rate_nm_min", "vertical_closing_rate_fpm", "relative_heading_deg",
    "crossing_angle_deg", "relative_bearing_deg", "cpa_nm", "tcpa_min",
    "predicted_min_horizontal_sep_nm", "predicted_min_vertical_sep_ft",
    "time_to_predicted_minimum_s", "traffic_density", "nearby_aircraft_count_30nm",
    "shared_fpl_route_token_count", "topsky_predicted_min_sep_nm_proxy",
    "topsky_predicted_time_to_min_s_proxy", "separation_trend_30s", "separation_trend_60s",
    "separation_trend_120s",
]


def event_rows(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["relative_seconds"] = pd.to_numeric(features["relative_seconds"], errors="coerce")
    features["predictor_eligible"] = features["predictor_eligible"].astype(bool)
    selected = []
    for _, group in features.groupby("window_id", sort=False):
        eligible = group.loc[group["predictor_eligible"] & (group["relative_seconds"] <= 5)]
        if eligible.empty:
            eligible = group.loc[group["relative_seconds"] <= 0]
        if eligible.empty:
            continue
        selected.append(eligible.iloc[(eligible["relative_seconds"].abs()).argmin()])
    return pd.DataFrame(selected).reset_index(drop=True)


def metric_row(name: str, y_true, probability, prediction, split: str) -> dict:
    result = {"model": name, "split": split, "n": len(y_true)}
    if len(set(y_true)) > 1:
        result.update(
            {
                "roc_auc": roc_auc_score(y_true, probability),
                "pr_auc": average_precision_score(y_true, probability),
                "brier": brier_score_loss(y_true, probability),
            }
        )
    else:
        result.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
    result.update(
        {
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
        }
    )
    return result


WINDOW_SECONDS = 600


def temporal_split(events: pd.DataFrame, embargo_seconds: float = WINDOW_SECONDS * 2):
    """Split chronologically by archive week and enforce disjoint flights.

    The event table is one row per window.  A two-sided 10-minute window can
    overlap across a train/test boundary, so the test side starts after an
    explicit 1,200-second embargo.  Windows from a flight already present in
    training are removed from test as an additional leakage guard.
    """
    events = events.copy()
    events["event_epoch"] = pd.to_numeric(events["event_epoch"], errors="coerce")
    events = events.dropna(subset=["event_epoch"]).copy()
    flight_column = "focal_flight_key" if "focal_flight_key" in events else "flight_key"
    events["_split_flight_key"] = events[flight_column].astype(str)
    if "archive_week" in events.columns and events["archive_week"].notna().any():
        week_table = (
            events.groupby("archive_week", dropna=False)["event_epoch"]
            .min()
            .reset_index()
            .sort_values("event_epoch", kind="mergesort")
        )
    else:
        week_table = pd.DataFrame()
    if len(week_table) >= 2:
        cutoff_index = max(1, min(len(week_table) - 1, int(round(len(week_table) * 0.7))))
        test_weeks = set(week_table.iloc[cutoff_index:]["archive_week"])
        train = events.loc[~events["archive_week"].isin(test_weeks)].copy()
        test = events.loc[events["archive_week"].isin(test_weeks)].copy()
    else:
        windows = events[["window_id", "event_epoch"]].drop_duplicates().sort_values("event_epoch")
        if len(windows) < 2:
            return events.iloc[0:0].copy(), events.copy()
        cutoff_index = max(1, min(len(windows) - 1, int(round(len(windows) * 0.7))))
        cutoff = float(windows.iloc[cutoff_index]["event_epoch"])
        train = events.loc[events["event_epoch"] < cutoff].copy()
        test = events.loc[events["event_epoch"] >= cutoff].copy()
    if not train.empty:
        last_train_event = float(train["event_epoch"].max())
        test = test.loc[test["event_epoch"] >= last_train_event + embargo_seconds].copy()
        train_flights = set(train["_split_flight_key"])
        test = test.loc[~test["_split_flight_key"].isin(train_flights)].copy()
    assert set(train["_split_flight_key"]) & set(test["_split_flight_key"]) == set()
    train_windows = set(train["window_id"])
    test_windows = set(test["window_id"])
    assert train_windows.isdisjoint(test_windows)
    if not train.empty and not test.empty:
        assert float(test["event_epoch"].min()) >= float(train["event_epoch"].max()) + embargo_seconds
    return train.drop(columns=["_split_flight_key"]), test.drop(columns=["_split_flight_key"])


def save_optional_ebm(X_train, y_train, X_test, y_test, columns, output: Path) -> str:
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError:
        pd.DataFrame(
            [{"status": "not_run", "reason": "interpret is not installed", "feature": "", "importance": np.nan}]
        ).to_csv(output, index=False)
        return "not_installed"
    model = ExplainableBoostingClassifier(random_state=42, interactions=0)
    model.fit(X_train, y_train)
    importances = model.term_importances()
    pd.DataFrame({"feature": columns, "importance": importances, "status": "ebm"}).to_csv(output, index=False)
    return "ebm"


def make_calibration_figure(y_test, probabilities: dict[str, np.ndarray], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
    except ImportError:
        return
    directory = output / "figures" / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probability in probabilities.items():
        if len(set(y_test)) < 2:
            continue
        observed, predicted = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "calibration.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / "interaction_features.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "07_explainability")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "model_metrics.csv").exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_root}; use --overwrite")
    if not args.features_path.exists():
        raise FileNotFoundError("Run 05_build_interaction_features.py first")
    events = event_rows(pd.read_parquet(args.features_path))
    if events.empty:
        raise RuntimeError("No event-level rows available")
    if (events["relative_seconds"] > 0).any():
        raise RuntimeError("Leakage check failed: a post-clearance row entered the predictive event table")
    events["ATC_INTERVENTION"] = pd.to_numeric(events["ATC_INTERVENTION"], errors="coerce").fillna(0).astype(int)
    columns = [column for column in FEATURE_COLUMNS if column in events.columns]
    if not columns:
        raise RuntimeError("No continuous feature columns are available")
    train, test = temporal_split(events)
    train_ids = set(train["window_id"])
    test_ids = set(test["window_id"])
    assignments = events[[
        column for column in ("window_id", "event_epoch", "archive_week", "focal_flight_key", "ATC_INTERVENTION")
        if column in events.columns
    ]].drop_duplicates("window_id").copy()
    assignments["split"] = assignments["window_id"].map(
        lambda value: "train" if value in train_ids else "test" if value in test_ids else "excluded_embargo_or_flight_overlap"
    )
    assignments.to_csv(args.output_root / "split_assignments.csv", index=False)
    X_train = train[columns].apply(pd.to_numeric, errors="coerce")
    X_test = test[columns].apply(pd.to_numeric, errors="coerce")
    y_train = train["ATC_INTERVENTION"].to_numpy()
    y_test = test["ATC_INTERVENTION"].to_numpy()
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("Temporal split produced an empty train or test set")
    if len(set(y_train)) < 2:
        # A limited smoke test may contain only interventions.  Preserve a
        # machine-readable result and stop before a classifier fabricates a
        # meaningful probability model from one class.
        pd.DataFrame([{
            "model": "not_run",
            "split": "temporal_test",
            "n": len(test),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "status": "insufficient_training_classes",
        }]).to_csv(args.output_root / "model_metrics.csv", index=False)
        for filename in ("logistic_coefficients.csv", "xgboost_feature_importance.csv", "shap_feature_importance.csv", "ebm_feature_importance.csv"):
            pd.DataFrame([{"status": "not_run", "reason": "training split has one class"}]).to_csv(args.output_root / filename, index=False)
        (args.output_root / "decision_tree_rules.txt").write_text("Not run: training split has one class.\n", encoding="utf-8")
        config = {"features_path": str(args.features_path), "features": columns, "split": "archive_week_temporal_flight_disjoint_with_1200s_embargo", "target": "ATC_INTERVENTION", "status": "insufficient_training_classes", "conflict_label_created": False}
        (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (args.output_root / "explainability_report.md").write_text(
            "# SCAT intervention explainability\n\n"
            "The model stage was not run because the temporal training split contains only one target class. This is expected for a small smoke test; run it with enough intervention and control windows.\n",
            encoding="utf-8",
        )
        update_read_me_first(args.output_root.parent)
        print(f"Explainability skipped safely: one training class in {args.output_root}")
        return
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
        ]),
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=2,
        )
    except ImportError:
        pass

    metrics = []
    probabilities = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics.append(metric_row(name, y_test, probability, prediction, "temporal_test"))
        fitted[name] = model
        probabilities[name] = probability
    pd.DataFrame(metrics).to_csv(args.output_root / "model_metrics.csv", index=False)

    logistic = fitted.get("logistic_regression")
    if logistic is not None:
        coefficients = logistic.named_steps["model"].coef_[0]
        pd.DataFrame({"feature": columns, "coefficient_standardized": coefficients}).sort_values("coefficient_standardized", key=np.abs, ascending=False).to_csv(args.output_root / "logistic_coefficients.csv", index=False)
    tree = fitted.get("decision_tree")
    if tree is not None:
        rules = export_text(tree.named_steps["model"], feature_names=columns)
        (args.output_root / "decision_tree_rules.txt").write_text(rules, encoding="utf-8")

    xgb = fitted.get("xgboost")
    if xgb is not None:
        importance = xgb.feature_importances_
        pd.DataFrame({"feature": columns, "importance": importance}).sort_values("importance", ascending=False).to_csv(args.output_root / "xgboost_feature_importance.csv", index=False)
        if len(set(y_test)) > 1:
            permutation = permutation_importance(xgb, X_test_imputed, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
        else:
            permutation = permutation_importance(xgb, X_train_imputed, y_train, n_repeats=10, random_state=42, scoring="accuracy")
        pd.DataFrame({"feature": columns, "importance_mean": permutation.importances_mean, "importance_std": permutation.importances_std, "method": "permutation_fallback"}).sort_values("importance_mean", ascending=False).to_csv(args.output_root / "shap_feature_importance.csv", index=False)
    else:
        pd.DataFrame([{"feature": "", "importance_mean": np.nan, "importance_std": np.nan, "method": "xgboost_not_available"}]).to_csv(args.output_root / "shap_feature_importance.csv", index=False)

    ebm_status = save_optional_ebm(X_train_imputed, y_train, X_test_imputed, y_test, columns, args.output_root / "ebm_feature_importance.csv")
    make_calibration_figure(y_test, probabilities, args.output_root)
    config = {
        "features_path": str(args.features_path),
        "features": columns,
        "split": "chronological archive weeks; flight-disjoint; 1200-second window-overlap embargo; post-clearance rows excluded",
        "target": "ATC_INTERVENTION",
        "ebm_status": ebm_status,
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = [
        "# SCAT intervention explainability",
        "",
        f"- Training rows: **{len(train)}**",
        f"- Temporal test rows: **{len(test)}**",
        f"- Excluded split rows (flight overlap/embargo): **{len(events) - len(train) - len(test)}**",
        f"- Features: **{len(columns)}**",
        f"- EBM status: **{ebm_status}**",
        "",
        "The target is only `ATC_INTERVENTION`. Metrics do not measure conflict-detection performance. The split uses chronological archive weeks, disjoint flights and a 1,200-second embargo for the ten-minute windows; split_assignments.csv records every decision.",
        "If a metric is close to 1.0, audit leakage, repeated windows and flight/event overlap before interpreting it as a good model.",
    ]
    (args.output_root / "explainability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Intervention explainability complete: {args.output_root}")


if __name__ == "__main__":
    main()

