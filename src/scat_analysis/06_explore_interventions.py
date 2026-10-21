#!/usr/bin/env python3
"""Explore intervention/control differences and data-driven traffic families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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


def numeric_matrix(frame: pd.DataFrame, columns: list[str]):
    available = [column for column in columns if column in frame.columns]
    matrix = frame[available].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(matrix))
    return transformed, available


def make_figures(frame: pd.DataFrame, matrix: np.ndarray, output: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created = []
    distributions = output / "figures" / "distributions"
    clustering = output / "figures" / "clustering"
    pca_dir = output / "figures" / "pca"
    for directory in (distributions, clustering, pca_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if "horizontal_separation_nm" in frame:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, group in frame.groupby("ATC_INTERVENTION"):
            ax.hist(group["horizontal_separation_nm"].dropna(), bins=40, alpha=0.55, label=f"ATC_INTERVENTION={label}")
        ax.set_xlabel("Horizontal separation (NM)")
        ax.set_ylabel("Event count")
        ax.legend()
        fig.tight_layout()
        path = distributions / "horizontal_separation_by_intervention.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))
    if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
        projection = PCA(n_components=2, random_state=42).fit_transform(matrix)
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(projection[:, 0], projection[:, 1], c=frame["ATC_INTERVENTION"].to_numpy(), s=12, alpha=0.65, cmap="coolwarm")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Traffic-state projection; colour is ATC_INTERVENTION")
        fig.colorbar(scatter, ax=ax)
        fig.tight_layout()
        path = pca_dir / "pca_intervention.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / "interaction_features.parquet")
    parser.add_argument("--clearances-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "03_clearances" / "scat_clearances.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "06_exploration")
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "cluster_assignments.parquet").exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_root}; use --overwrite")
    if not args.features_path.exists():
        raise FileNotFoundError("Run 05_build_interaction_features.py first")
    raw = pd.read_parquet(args.features_path)
    events = event_rows(raw)
    if events.empty:
        raise RuntimeError("No pre-clearance event rows are available for exploration")
    if args.clearances_path.exists():
        clearances = pd.read_parquet(args.clearances_path, columns=["clearance_id", "clearance_type"])
        events = events.merge(clearances, on="clearance_id", how="left")
    else:
        events["clearance_type"] = ""

    available = [column for column in FEATURE_COLUMNS if column in events.columns]
    descriptive_rows = []
    for target, group in events.groupby("ATC_INTERVENTION", dropna=False):
        for feature in available:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            descriptive_rows.append(
                {
                    "ATC_INTERVENTION": target,
                    "feature": feature,
                    "count": len(values),
                    "mean": values.mean() if len(values) else np.nan,
                    "median": values.median() if len(values) else np.nan,
                    "std": values.std() if len(values) > 1 else np.nan,
                    "p05": values.quantile(0.05) if len(values) else np.nan,
                    "p95": values.quantile(0.95) if len(values) else np.nan,
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(args.output_root / "descriptive_statistics.csv", index=False)
    pivot = descriptive.pivot(index="feature", columns="ATC_INTERVENTION", values="median")
    if 0 in pivot.columns and 1 in pivot.columns:
        pivot["median_difference_1_minus_0"] = pivot[1] - pivot[0]
    pivot.reset_index().to_csv(args.output_root / "intervention_vs_control_statistics.csv", index=False)

    matrix, used_features = numeric_matrix(events, available)
    cluster_ids = np.full(len(events), -1, dtype=int)
    clustering_method = "none"
    silhouette_by_k = []
    if len(events) >= 4:
        upper = min(args.max_clusters, len(events) - 1)
        best_score = -np.inf
        for k in range(2, upper + 1):
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            candidate = model.fit_predict(matrix)
            score = silhouette_score(matrix, candidate) if len(set(candidate)) > 1 else -np.inf
            silhouette_by_k.append({"k": k, "silhouette": score})
            if score > best_score:
                best_score = score
                cluster_ids = candidate
        clustering_method = "agglomerative_ward_selected_by_silhouette"
    events["cluster_id"] = cluster_ids
    events["cluster_method"] = clustering_method
    events.to_parquet(args.output_root / "cluster_assignments.parquet", index=False, compression="zstd")
    pd.DataFrame(silhouette_by_k).to_csv(args.output_root / "cluster_selection.csv", index=False)
    summary_rows = []
    for cluster, group in events.groupby("cluster_id", dropna=False):
        row = {"cluster_id": cluster, "size": len(group), "intervention_rate": group["ATC_INTERVENTION"].mean()}
        counts = group["clearance_type"].fillna("").value_counts()
        row["dominant_clearance_type"] = counts.index[0] if len(counts) else ""
        for feature in available[:8]:
            row[f"median_{feature}"] = pd.to_numeric(group[feature], errors="coerce").median()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.output_root / "cluster_summary.csv", index=False)
    figure_paths = make_figures(events, matrix, args.output_root)
    config = {
        "features_path": str(args.features_path),
        "event_definition": "closest predictor-eligible row to each window event, never after t",
        "features_used": used_features,
        "max_clusters_considered": args.max_clusters,
        "clustering_method": clustering_method,
        "conflict_label_created": False,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = [
        "# SCAT intervention exploration",
        "",
        f"- Event windows analysed: **{len(events)}**",
        f"- `ATC_INTERVENTION=1`: **{int((events['ATC_INTERVENTION'] == 1).sum())}**",
        f"- `ATC_INTERVENTION=0`: **{int((events['ATC_INTERVENTION'] == 0).sum())}**",
        f"- Clustering: **{clustering_method}**",
        f"- Clusters selected: **{events['cluster_id'].nunique()}**",
        "",
        "The descriptive comparisons and clusters are exploratory. They do not mean conflict/no-conflict and do not validate any OpenSky threshold.",
        "",
        "## Outputs",
        "",
        "- `descriptive_statistics.csv` and `intervention_vs_control_statistics.csv` retain continuous variables.",
        "- `cluster_summary.csv` reports naturally selected families rather than a fixed number of safety states.",
        "- `figures/` contains optional PCA and distribution plots.",
    ]
    (args.output_root / "exploration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Intervention exploration complete: {args.output_root}")


if __name__ == "__main__":
    main()

