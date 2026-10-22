#!/usr/bin/env python3
"""Build a data-driven discrete dependency screen for ATC_INTERVENTION.

The full pgmpy structure learner is optional. When it is unavailable, this
stage still creates transparent quantile-based conditional tables and mutual
information edges, explicitly labelled as a dependency screen rather than a
causal Bayesian network. The filename is retained for compatibility; the
phase is conceptually named ``dependency_screen`` and does not produce a
Markov blanket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from common import DEFAULT_OUTPUT_ROOT, update_read_me_first
from pilot_scope import PILOT_WEEK, assert_pilot_frame, assert_pilot_parquet, scope_config


FEATURES = [
    "min_horizontal_separation", "min_vertical_separation", "min_predicted_horizontal_cpa",
    "tcpa_at_min_cpa", "max_horizontal_closing_rate", "max_vertical_closing_rate",
    "median_crossing_angle", "median_traffic_density", "flight_level", "ground_speed_kt",
]


def event_rows(features: pd.DataFrame) -> pd.DataFrame:
    if "anchor_relative_seconds" not in features or features.window_id.duplicated().any():
        raise SystemExit("Stage 08 requires one pre-event row per pilot window")
    return features.reset_index(drop=True)


def quantile_bins(series: pd.Series, bins: int = 4) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.nunique(dropna=True) < 2:
        return pd.Series("constant", index=series.index, dtype="string")
    result = pd.qcut(numeric, q=min(bins, numeric.nunique()), duplicates="drop")
    return result.astype("string").fillna("missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "08_dependency_screen")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "dependency_edges.csv").exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_root}; use --overwrite")
    if not args.features_path.exists():
        raise FileNotFoundError("Run 05_build_interaction_features.py first")
    if args.features_path.name != f"window_features_{PILOT_WEEK}.parquet":
        raise SystemExit("Stage 08 requires first-week primary window features")
    assert_pilot_parquet(args.features_path)
    events = event_rows(pd.read_parquet(args.features_path))
    assert_pilot_frame(events)
    if events.empty:
        raise RuntimeError("No event-level rows available")
    y = pd.to_numeric(events["ATC_INTERVENTION"], errors="coerce").fillna(0).astype(int)
    available = [feature for feature in FEATURES if feature in events.columns]
    numeric = events[available].apply(pd.to_numeric, errors="coerce")
    matrix = SimpleImputer(strategy="median").fit_transform(numeric)
    edges = []
    if len(set(y)) > 1 and available:
        mi = mutual_info_classif(matrix, y, random_state=42)
        for feature, score in zip(available, mi):
            edges.append({"source": feature, "target": "ATC_INTERVENTION", "mutual_information": score, "method": "quantile_dependency_screen"})
    edge_frame = pd.DataFrame(edges).sort_values("mutual_information", ascending=False) if edges else pd.DataFrame(columns=["source", "target", "mutual_information", "method"])
    edge_frame.to_csv(args.output_root / "dependency_edges.csv", index=False)

    discretised = pd.DataFrame(index=events.index)
    for feature in available:
        discretised[feature] = quantile_bins(events[feature])
    discretised["ATC_INTERVENTION"] = y.astype(str)
    conditional_rows = []
    for feature in available:
        table = pd.crosstab(discretised[feature], discretised["ATC_INTERVENTION"], normalize="index")
        for level, row in table.iterrows():
            for target, probability in row.items():
                conditional_rows.append({"feature": feature, "feature_bin": level, "ATC_INTERVENTION": target, "probability": probability, "discretisation": "data-driven quantile"})
    pd.DataFrame(conditional_rows).to_csv(args.output_root / "conditional_probabilities.csv", index=False)
    method = "mutual_information_dependency_screen"
    report = [
        "# SCAT dependency screen",
        "",
        "- analysis_scope: `FIRST_WEEK_PILOT`",
        f"- archive_week: `{PILOT_WEEK}`",
        f"- Event rows: **{len(events)}**",
        f"- Variables discretised using data-driven quantiles: **{len(available)}**",
        f"- Method: **{method}**",
        "",
        "This phase is a transparent dependency screen with conditional tables, not a learned Bayesian network and not a Markov-blanket analysis.",
        "Quantile bins are chosen from the data; no 5 NM, 1000 ft or GREEN/YELLOW/ORANGE/RED threshold is used.",
        "Mutual-information edges and conditional probabilities describe association with `ATC_INTERVENTION`, not causality.",
    ]
    (args.output_root / "dependency_screen_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.output_root / "run_config.json").write_text(json.dumps({**scope_config(), "features_path": str(args.features_path), "method": method, "target": "ATC_INTERVENTION", "conflict_label_created": False}, indent=2), encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Dependency screen complete: {args.output_root}")


if __name__ == "__main__":
    main()

