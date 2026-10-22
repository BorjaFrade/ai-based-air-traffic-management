#!/usr/bin/env python3
"""First-week, one-window-per-observation exploration of ATC interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_ROOT, update_read_me_first
from pilot_scope import PILOT_WEEK, assert_pilot_frame, assert_pilot_parquet, scope_config


# Only pre-event continuous traffic-state variables. No label, clearance command,
# matching field, post-event value, or clearance type can enter the model.
FEATURES = [
    "min_horizontal_separation", "p05_horizontal_separation", "horizontal_sep_at_event",
    "min_vertical_separation", "p05_vertical_separation", "vertical_sep_at_event",
    "min_predicted_horizontal_cpa", "vertical_sep_at_min_cpa", "tcpa_at_min_cpa",
    "max_horizontal_closing_rate", "max_vertical_closing_rate", "median_crossing_angle",
    "median_relative_heading", "median_relative_bearing", "median_traffic_density",
    "n_unique_neighbours", "median_relative_velocity_kt", "flight_level",
    "ground_speed_kt", "vertical_rate_fpm",
    "min_topsky_predicted_sep_nm_proxy", "horizontal_sep_trend_60s",
    "vertical_sep_trend_60s", "horizontal_sep_trend_120s", "vertical_sep_trend_120s",
]


def describe(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for target, group in frame.groupby("ATC_INTERVENTION"):
        for col in FEATURES:
            x = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"ATC_INTERVENTION": int(target), "feature": col, "count": len(x),
                         "missing": len(group) - len(x), "mean": x.mean(), "std": x.std(),
                         "median": x.median(), "p25": x.quantile(.25), "p75": x.quantile(.75),
                         "iqr": x.quantile(.75) - x.quantile(.25),
                         "p05": x.quantile(.05), "p95": x.quantile(.95)})
    descriptive = pd.DataFrame(rows)
    descriptive.to_csv(out / "descriptive_statistics.csv", index=False)
    a = descriptive.loc[descriptive.ATC_INTERVENTION == 1].set_index("feature")
    b = descriptive.loc[descriptive.ATC_INTERVENTION == 0].set_index("feature")
    pooled = np.sqrt((a["std"] ** 2 + b["std"] ** 2) / 2)
    comparison = pd.DataFrame({"feature": FEATURES, "intervention_median": a.loc[FEATURES, "median"].to_numpy(),
        "control_median": b.loc[FEATURES, "median"].to_numpy(),
        "intervention_iqr": a.loc[FEATURES, "iqr"].to_numpy(),
        "control_iqr": b.loc[FEATURES, "iqr"].to_numpy(),
        "median_difference_1_minus_0": (a.loc[FEATURES, "median"] - b.loc[FEATURES, "median"]).to_numpy(),
        "smd": ((a.loc[FEATURES, "mean"] - b.loc[FEATURES, "mean"]) / pooled.loc[FEATURES].replace(0, np.nan)).to_numpy()})
    comparison.to_csv(out / "intervention_vs_control_statistics.csv", index=False)
    return comparison


def score_clusters(x: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels >= 0
    count = len(set(labels[mask]))
    result = {"n_clusters": count, "noise_fraction": float((~mask).mean()),
              "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if 1 < count < mask.sum():
        result["silhouette"] = float(silhouette_score(x[mask], labels[mask], sample_size=min(2500, mask.sum()), random_state=42))
        result["davies_bouldin"] = float(davies_bouldin_score(x[mask], labels[mask]))
        result["calinski_harabasz"] = float(calinski_harabasz_score(x[mask], labels[mask]))
    return result


def plot_results(frame: pd.DataFrame, x: np.ndarray, labels: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for target, group in frame.groupby("ATC_INTERVENTION"):
        ax.hist(group["min_horizontal_separation"].dropna(), bins=50, range=(0, 50),
                alpha=.5, label=f"ATC_INTERVENTION={target}")
    ax.set(xlabel="Minimum horizontal separation (NM; clipped display 0–50)", ylabel="Windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "horizontal_separation_by_intervention.png", dpi=150)
    plt.close(fig)
    projection = PCA(n_components=2, random_state=42).fit_transform(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, s=5, alpha=.5, cmap="tab10")
    ax.set(xlabel="PCA 1", ylabel="PCA 2", title="Intervention traffic states; PCA is visualization only")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(fig_dir / "intervention_pca_clusters.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_{PILOT_WEEK}.parquet")
    parser.add_argument("--clearances-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "03_clearances" / "scat_clearances.parquet")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "06_exploration")
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.features_path.name != f"window_features_{PILOT_WEEK}.parquet":
        raise SystemExit("Stage 06 requires the first-week one-row-per-window feature file")
    assert_pilot_parquet(args.features_path)
    if (args.output_root / "cluster_assignments.parquet").exists() and not args.overwrite:
        raise SystemExit("Stage 06 output exists; pass --overwrite to replace it")
    frame = pd.read_parquet(args.features_path)
    assert_pilot_frame(frame)
    if frame.window_id.duplicated().any() or not set(frame.ATC_INTERVENTION.unique()) == {0, 1}:
        raise SystemExit("Stage 06 requires unique windows and both intervention/control groups")
    if (pd.to_numeric(frame.anchor_relative_seconds, errors="coerce") > 0).any():
        raise SystemExit("Post-event window anchor detected")
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing pre-event features: {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison = describe(frame, args.output_root)
    interventions = frame.loc[frame.ATC_INTERVENTION == 1].copy().reset_index(drop=True)
    numeric = interventions[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputer.fit_transform(numeric))
    reducer = PCA(n_components=.90, svd_solver="full", random_state=42)
    x = reducer.fit_transform(scaled)
    selections = []
    models = {}
    for k in range(2, args.max_clusters + 1):
        model = GaussianMixture(n_components=k, covariance_type="diag", n_init=2, random_state=42, reg_covar=1e-5)
        model.fit(x)
        selections.append({"k": k, "bic": model.bic(x), "aic": model.aic(x)})
        models[k] = model
        print(f"Stage 06 GMM: k={k}/{args.max_clusters}", flush=True)
    selection = pd.DataFrame(selections)
    selection.to_csv(args.output_root / "cluster_selection.csv", index=False)
    best_k = int(selection.loc[selection.bic.idxmin(), "k"])
    labels = models[best_k].predict(x)
    main_metrics = score_clusters(x, labels)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(x), size=min(3000, len(x)), replace=False)
    alternatives = []
    for name, method in (("HDBSCAN_sample", HDBSCAN(min_cluster_size=60, min_samples=15)),
                         ("Agglomerative_Ward_sample", AgglomerativeClustering(n_clusters=best_k, linkage="ward"))):
        try:
            alternative_labels = method.fit_predict(x[sample_idx])
            alternatives.append({"method": name, "sample_size": len(sample_idx), **score_clusters(x[sample_idx], alternative_labels)})
        except Exception as exc:
            alternatives.append({"method": name, "sample_size": len(sample_idx), "error": str(exc)})
    pd.DataFrame(alternatives).to_csv(args.output_root / "alternative_cluster_methods.csv", index=False)
    # Independent subsamples train two GMMs; compare assignments on a common holdout.
    order = rng.permutation(len(x))
    holdout = order[:min(2000, len(x) // 5)]
    pool = order[len(holdout):]
    fits = []
    for seed in (17, 29):
        train = rng.choice(pool, size=int(.7 * len(pool)), replace=False)
        model = GaussianMixture(n_components=best_k, covariance_type="diag", n_init=1,
                                random_state=seed, reg_covar=1e-5).fit(x[train])
        fits.append(model.predict(x[holdout]))
    stability_ari = float(adjusted_rand_score(fits[0], fits[1]))
    interventions["cluster_id"] = labels
    interventions["cluster_method"] = "GMM_diag_BIC"
    interventions.to_parquet(args.output_root / "cluster_assignments.parquet", index=False, compression="zstd")
    summary = interventions.groupby("cluster_id").agg(size=("window_id", "size"),
        median_min_horizontal_separation=("min_horizontal_separation", "median"),
        median_min_vertical_separation=("min_vertical_separation", "median"),
        median_min_cpa=("min_predicted_horizontal_cpa", "median"),
        median_density=("median_traffic_density", "median"),
        median_neighbours=("n_unique_neighbours", "median")).reset_index()
    summary.to_csv(args.output_root / "cluster_summary.csv", index=False)
    clearance_columns = ["archive_week", "clearance_id", "clearance_type"]
    clearances = pd.read_parquet(args.clearances_path, columns=clearance_columns,
                                 filters=[("archive_week", "==", PILOT_WEEK)])
    assert_pilot_frame(clearances)
    clearances = clearances.drop_duplicates("clearance_id")
    composition = interventions[["clearance_id", "cluster_id"]].merge(
        clearances[["clearance_id", "clearance_type"]], on="clearance_id", how="left", validate="many_to_one")
    composition.groupby(["cluster_id", "clearance_type"], dropna=False).size().rename("count").reset_index().to_csv(
        args.output_root / "clearance_type_composition.csv", index=False)
    plot_results(frame, x, labels, args.output_root)
    config = {**scope_config(), "features_path": str(args.features_path), "clearances_path": str(args.clearances_path),
              "unit_of_analysis": "one pre-event window", "clustering_population": "ATC_INTERVENTION=1 only",
              "model_features": FEATURES, "selection": "minimum GMM BIC among k=2..max_clusters",
              "max_clusters": args.max_clusters, "pca_explained_variance": float(reducer.explained_variance_ratio_.sum()),
              "pca_components": int(x.shape[1]), "random_seed": 42, "conflict_label_created": False}
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    top = comparison.reindex(comparison.smd.abs().sort_values(ascending=False).index).head(8)
    lines = ["# SCAT first-week intervention exploration", "", f"- analysis_scope: `FIRST_WEEK_PILOT`",
             f"- archive_week: `{PILOT_WEEK}`", f"- Windows: **{len(frame):,}**; interventions: **{len(interventions):,}**; controls: **{len(frame)-len(interventions):,}**",
             "- Unit: one pre-event window; clustering population: interventions only.",
             f"- GMM components selected by BIC: **{best_k}**; PCA components: **{x.shape[1]}** ({reducer.explained_variance_ratio_.sum():.1%} variance).",
             f"- Silhouette: **{main_metrics['silhouette']:.3f}**; Davies–Bouldin: **{main_metrics['davies_bouldin']:.3f}**; Calinski–Harabasz: **{main_metrics['calinski_harabasz']:.1f}**.",
             f"- Two-subsample common-holdout ARI: **{stability_ari:.3f}**.", "", "## Largest descriptive shifts by |SMD|", "",
             "| Feature | Intervention median | Control median | SMD |", "|---|---:|---:|---:|"]
    lines += [f"| {r.feature} | {r.intervention_median:.3f} | {r.control_median:.3f} | {r.smd:.3f} |" for r in top.itertuples()]
    lines += ["", "These are traffic-state clusters associated with ATC interventions, not confirmed conflict classes.",
              "Clearance type was joined only after clustering. No fixed separation threshold or OpenSky colour was used.",
              "Physical-tail warnings in `../05_features/window_feature_physical_report.md` limit operational interpretation."]
    (args.output_root / "exploration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_read_me_first(args.output_root.parent)
    print(f"Stage 06 complete: {len(interventions):,} intervention windows; {best_k} GMM clusters; stability ARI {stability_ari:.3f}", flush=True)


if __name__ == "__main__":
    main()
