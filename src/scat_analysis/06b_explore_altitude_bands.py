#!/usr/bin/env python3
"""First-week Stage 06 refinement: pair-preserving altitude-band exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_ROOT, update_read_me_first
from pilot_scope import PILOT_WEEK, assert_pilot_frame, assert_pilot_parquet, scope_config


BANDS = ["VERY_LOW", "LOW", "MID", "CRUISE_FL290_410", "ABOVE_FL410", "UNKNOWN_INVALID"]
ANALYSIS_BANDS = BANDS[:4]
FEATURES = [
    "min_horizontal_separation", "p05_horizontal_separation", "median_horizontal_separation",
    "horizontal_sep_at_event", "min_horizontal_sep_within_30nm",
    "min_vertical_separation", "p05_vertical_separation", "median_vertical_separation",
    "vertical_sep_at_event", "min_predicted_horizontal_cpa", "min_cpa_within_30nm",
    "vertical_sep_at_min_cpa", "tcpa_at_min_cpa", "min_positive_tcpa",
    "min_tcpa_ge_1s_within_30nm", "max_horizontal_closing_within_30nm",
    "max_vertical_closing_rate", "max_vertical_closing_within_30nm",
    "horizontal_closing_rate_at_min_cpa", "median_relative_heading",
    "median_crossing_angle", "median_relative_bearing", "median_traffic_density",
    "n_unique_neighbours_within_30nm", "median_nearby_aircraft_30nm", "far30_pair_fraction",
    "median_relative_velocity_kt", "ground_speed_at_anchor_kt", "vertical_rate_at_anchor_fpm",
    "analysis_altitude_ft", "min_topsky_predicted_sep_nm_proxy",
    "horizontal_sep_trend_60s", "vertical_sep_trend_60s",
    "horizontal_sep_trend_120s", "vertical_sep_trend_120s",
    "horizontal_sep_trend_300s", "vertical_sep_trend_300s",
    "horizontal_sep_trend_600s", "vertical_sep_trend_600s",
]
CLUSTER_FEATURES = [
    "min_horizontal_sep_within_30nm", "min_vertical_separation", "min_cpa_within_30nm",
    "vertical_sep_at_min_cpa", "tcpa_at_min_cpa", "max_horizontal_closing_within_30nm",
    "max_vertical_closing_within_30nm", "median_relative_heading", "median_crossing_angle",
    "median_relative_bearing", "median_traffic_density", "n_unique_neighbours_within_30nm", "median_nearby_aircraft_30nm",
    "ground_speed_at_anchor_kt", "vertical_rate_at_anchor_fpm", "analysis_altitude_ft",
    "min_topsky_predicted_sep_nm_proxy", "horizontal_sep_trend_60s", "vertical_sep_trend_60s",
    "horizontal_sep_trend_120s", "vertical_sep_trend_120s",
]
HORIZONS = (60, 120, 300, 600)


def smd(a, b) -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.to_numeric(pd.Series(b), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    scale = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    delta = x.mean() - y.mean()
    return float(delta / scale) if scale > 0 else (0.0 if abs(delta) < 1e-12 else np.inf)


def status(value) -> str:
    if not np.isfinite(value):
        return "UNDETERMINED"
    if abs(value) < .10:
        return "PASS"
    if abs(value) <= .20:
        return "WARNING"
    return "FAIL"


def score(x: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels >= 0
    count = len(set(labels[mask]))
    result = {"n_clusters": count, "noise_fraction": float((~mask).mean()),
              "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if 1 < count < mask.sum() and mask.sum() >= 30:
        result["silhouette"] = float(silhouette_score(x[mask], labels[mask], sample_size=min(2000, mask.sum()), random_state=42))
        result["davies_bouldin"] = float(davies_bouldin_score(x[mask], labels[mask]))
        result["calinski_harabasz"] = float(calinski_harabasz_score(x[mask], labels[mask]))
    return result


def describe(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for target, part in frame.groupby("ATC_INTERVENTION"):
        for feature in FEATURES:
            values = pd.to_numeric(part[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"ATC_INTERVENTION": int(target), "feature": feature, "count": len(values),
                         "missing": len(part) - len(values), "mean": values.mean(), "std": values.std(),
                         "p05": values.quantile(.05), "p25": values.quantile(.25),
                         "median": values.median(), "p75": values.quantile(.75),
                         "p95": values.quantile(.95), "iqr": values.quantile(.75) - values.quantile(.25)})
    desc = pd.DataFrame(rows)
    a = desc.loc[desc.ATC_INTERVENTION == 1].set_index("feature")
    b = desc.loc[desc.ATC_INTERVENTION == 0].set_index("feature")
    comp_rows = []
    for feature in FEATURES:
        r, s = a.loc[feature], b.loc[feature]
        comp_rows.append({"feature": feature, "intervention_count": int(r["count"]), "control_count": int(s["count"]),
                          "intervention_missing": int(r["missing"]), "control_missing": int(s["missing"]),
                          "intervention_median": r["median"], "control_median": s["median"],
                          "intervention_p05": r["p05"], "control_p05": s["p05"],
                          "intervention_p25": r["p25"], "control_p25": s["p25"],
                          "intervention_p75": r["p75"], "control_p75": s["p75"],
                          "intervention_p95": r["p95"], "control_p95": s["p95"],
                          "intervention_iqr": r["iqr"], "control_iqr": s["iqr"],
                          "median_difference": r["median"] - s["median"],
                          "smd": smd(frame.loc[frame.ATC_INTERVENTION == 1, feature], frame.loc[frame.ATC_INTERVENTION == 0, feature])})
    comp = pd.DataFrame(comp_rows)
    comp["abs_smd"] = comp.smd.abs()
    return desc, comp.sort_values("abs_smd", ascending=False, na_position="last")


def matching_balance(wide: pd.DataFrame, band: str) -> pd.DataFrame:
    rows = []
    numeric = [("local_traffic_density", "intervention_local_traffic_density", "control_local_traffic_density"),
               ("analysis_altitude_ft", "analysis_altitude_ft_I", "analysis_altitude_ft_C"),
               ("flight_level_original", "flight_level_original_I", "flight_level_original_C"),
               ("ground_speed_at_anchor_kt", "ground_speed_at_anchor_kt_I", "ground_speed_at_anchor_kt_C"),
               ("ground_speed_median_pre_event_kt", "ground_speed_kt_I", "ground_speed_kt_C"),
               ("hour_utc", "hour_utc_I", "hour_utc_C")]
    for name, left_col, right_col in numeric:
        left, right = wide[left_col], wide[right_col]
        value = smd(left, right)
        rows.append({"band": band, "feature": name, "intervention_mean": pd.to_numeric(left, errors="coerce").mean(),
                     "control_mean": pd.to_numeric(right, errors="coerce").mean(),
                     "intervention_median": pd.to_numeric(left, errors="coerce").median(),
                     "control_median": pd.to_numeric(right, errors="coerce").median(), "SMD": value,
                     "status": "DIAGNOSTIC" if name == "ground_speed_median_pre_event_kt" else status(value)})
    for name, field in (("phase", "flight_phase"), ("centre", "centre"), ("hour_utc_category", "hour_utc"),
                        ("stage04_phase", "intervention_phase"), ("stage04_centre", "intervention_centre")):
        if name == "stage04_phase":
            left = wide["intervention_phase"].fillna("UNKNOWN").astype(str)
            right = wide["control_phase"].fillna("UNKNOWN").astype(str)
        elif name == "stage04_centre":
            left = wide["intervention_centre"].fillna("UNKNOWN").astype(str)
            right = wide["control_centre"].fillna("UNKNOWN").astype(str)
        else:
            left = wide[f"{field}_I"].fillna("UNKNOWN").astype(str)
            right = wide[f"{field}_C"].fillna("UNKNOWN").astype(str)
        for category in sorted(set(left) | set(right)):
            a, b = left.eq(category).astype(int), right.eq(category).astype(int)
            value = smd(a, b)
            rows.append({"band": band, "feature": f"{name}:{category}", "intervention_mean": a.mean(),
                         "control_mean": b.mean(), "intervention_median": a.median(),
                         "control_median": b.median(), "SMD": value, "status": status(value)})
    distance = pd.to_numeric(wide.control_time_difference_s, errors="coerce")
    rows.append({"band": band, "feature": "matching_temporal_distance_s", "intervention_mean": np.nan,
                 "control_mean": distance.mean(), "intervention_median": np.nan, "control_median": distance.median(),
                 "SMD": np.nan, "status": "DIAGNOSTIC"})
    return pd.DataFrame(rows)


def cluster_band(interventions: pd.DataFrame, band: str, output: Path, types: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    n = len(interventions)
    if n < 300:
        return pd.DataFrame([{"band": band, "method": "NOT_RUN", "sample_size": n, "status": "INSUFFICIENT_FOR_CLUSTERING"}]), pd.DataFrame(), "INSUFFICIENT_DATA"
    matrix = interventions[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit_transform(matrix)
    x = PCA(n_components=.90, svd_solver="full", random_state=42).fit_transform(StandardScaler().fit_transform(imputed))
    upper = min(8, max(2, n // 75))
    models = {}
    selection = []
    for k in range(2, upper + 1):
        model = GaussianMixture(n_components=k, covariance_type="diag", n_init=2, reg_covar=1e-5, random_state=42).fit(x)
        models[k] = model
        selection.append({"band": band, "k": k, "bic": model.bic(x), "aic": model.aic(x)})
    pd.DataFrame(selection).to_csv(output / f"gmm_selection_{band}.csv", index=False)
    best = min(selection, key=lambda item: item["bic"])["k"]
    labels = models[best].predict(x)
    chosen_boundary = best == upper
    rng = np.random.default_rng(42)
    holdout_size = min(1000, n // 5)
    order = rng.permutation(n)
    holdout, pool = order[:holdout_size], order[holdout_size:]
    predictions = []
    for seed in (17, 29):
        train = rng.choice(pool, size=max(best * 5, int(.7 * len(pool))), replace=False)
        model = GaussianMixture(n_components=best, covariance_type="diag", n_init=1, reg_covar=1e-5, random_state=seed).fit(x[train])
        predictions.append(model.predict(x[holdout]))
    stability = float(adjusted_rand_score(predictions[0], predictions[1]))
    main = {"band": band, "method": "GMM_diag_BIC", "sample_size": n,
            "pca_components": x.shape[1], "selected_k_at_search_upper_bound": chosen_boundary,
            "stability_ari": stability, **score(x, labels)}
    validation = [main]
    sample_idx = rng.choice(n, size=min(2000, n), replace=False)
    for name, method in (("HDBSCAN_sample", HDBSCAN(min_cluster_size=max(20, int(.02 * len(sample_idx))), min_samples=10)),
                         ("Agglomerative_Ward_sample", AgglomerativeClustering(n_clusters=best, linkage="ward"))):
        try:
            alt_labels = method.fit_predict(x[sample_idx])
            validation.append({"band": band, "method": name, "sample_size": len(sample_idx), **score(x[sample_idx], alt_labels)})
        except Exception as exc:
            validation.append({"band": band, "method": name, "sample_size": len(sample_idx), "error": str(exc)})
    assignments = interventions[["window_id", "archive_week", "analysis_scope", "clearance_id", "analysis_altitude_ft", "altitude_band"]].copy()
    assignments["cluster_id"] = labels
    assignments.to_parquet(output / ("cruise_cluster_assignments.parquet" if band == "CRUISE_FL290_410" else f"cluster_assignments_{band}.parquet"), index=False, compression="zstd")
    characterized = assignments.merge(types, on="clearance_id", how="left", validate="many_to_one")
    characterized["clearance_type"] = characterized.clearance_type.fillna("OTHER")
    summaries = []
    source = interventions.assign(cluster_id=labels)
    for cluster_id, group in source.groupby("cluster_id"):
        type_counts = characterized.loc[characterized.cluster_id == cluster_id, "clearance_type"].value_counts()
        item = {"band": band, "cluster_id": int(cluster_id), "size": len(group),
                "median_horizontal_separation_within_30nm": group.min_horizontal_sep_within_30nm.median(),
                "median_vertical_separation": group.min_vertical_separation.median(),
                "median_cpa_within_30nm": group.min_cpa_within_30nm.median(),
                "median_density": group.median_traffic_density.median()}
        for typ in ("LEVEL", "HEADING", "SPEED", "OTHER"):
            item[f"clearance_{typ}"] = int(type_counts.get(typ, 0))
            item[f"percentage_{typ}"] = item[f"clearance_{typ}"] / len(group) * 100
        summaries.append(item)
    summary = pd.DataFrame(summaries)
    if band == "CRUISE_FL290_410":
        summary.to_csv(output / "cruise_cluster_summary.csv", index=False)
        pd.DataFrame(validation).to_csv(output / "cruise_cluster_validation.csv", index=False)
        characterized.groupby(["cluster_id", "clearance_type"]).size().rename("count").reset_index().to_csv(
            output / "cruise_cluster_clearance_composition.csv", index=False)
    hdb = next((row for row in validation if row["method"] == "HDBSCAN_sample"), {})
    if main["silhouette"] >= .25 and stability >= .75 and hdb.get("n_clusters", 0) >= 2 and hdb.get("noise_fraction", 1) < .5 and not chosen_boundary:
        evidence = "STRONG"
    elif main["silhouette"] >= .15 and stability >= .6 and hdb.get("n_clusters", 0) >= 2 and not chosen_boundary:
        evidence = "MODERATE"
    elif main["n_clusters"] > 1:
        evidence = "WEAK"
    else:
        evidence = "NONE"
    print(f"Band {band}: n={n:,}, GMM={best}, silhouette={main['silhouette']:.3f}, HDBSCAN={hdb.get('n_clusters', 'error')}, evidence={evidence}", flush=True)
    return pd.DataFrame(validation), summary, evidence


def horizon_table(wide: pd.DataFrame, band: str, seed: int = 42) -> pd.DataFrame:
    rows = []
    n = len(wide)
    rng = np.random.default_rng(seed)
    for horizon in HORIZONS:
        for variable in ("horizontal_sep", "vertical_sep"):
            feature = f"{variable}_trend_{horizon}s"
            a = pd.to_numeric(wide[f"{feature}_I"], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(wide[f"{feature}_C"], errors="coerce").to_numpy(dtype=float)
            effect = smd(a, b)
            bootstrap = []
            if n >= 30 and np.isfinite(effect):
                for _ in range(20):
                    idx = rng.integers(0, n, size=n)
                    bootstrap.append(smd(a[idx], b[idx]))
            valid_bootstrap = [x for x in bootstrap if np.isfinite(x)]
            stability = float(np.mean(np.sign(valid_bootstrap) == np.sign(effect))) if valid_bootstrap and effect != 0 else np.nan
            rows.append({"band": band, "horizon_s": horizon, "feature": feature, "pairs": n,
                         "missingness_intervention": float(np.isnan(a).mean()) if n else np.nan,
                         "missingness_control": float(np.isnan(b).mean()) if n else np.nan,
                         "SMD": effect, "bootstrap_sign_stability": stability,
                         "bootstrap_replicates": len(valid_bootstrap)})
    return pd.DataFrame(rows)


def recommended_horizon(table: pd.DataFrame) -> str:
    for horizon in HORIZONS:
        part = table.loc[table.horizon_s == horizon]
        good = part.loc[part.missingness_intervention.le(.2) & part.missingness_control.le(.2)
                        & part.SMD.abs().ge(.1) & part.bootstrap_sign_stability.ge(.8)]
        if len(good):
            return f"{horizon}s (pilot signal; retain adjacent horizon for sensitivity)"
    return "UNDETERMINED"


def figure_set(output: Path, inventory: pd.DataFrame, comparison: dict, composition: pd.DataFrame,
               validation: pd.DataFrame, rankings: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures" / "altitude_bands"
    figures.mkdir(parents=True, exist_ok=True)
    bands = ANALYSIS_BANDS
    inv = inventory.set_index("band").reindex(bands)
    x = np.arange(len(bands))

    def save(number: int, name: str, fig):
        fig.tight_layout()
        fig.savefig(figures / f"{number:02d}_{name}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, inv.valid_matched_pairs)
    ax.set(xticks=x, xticklabels=bands, ylabel="Valid matched pairs", title="Pilot sample sizes by altitude band")
    ax.tick_params(axis="x", rotation=20)
    save(1, "band_sample_sizes", fig)

    comp = composition.pivot_table(index="band", columns="clearance_type", values="count", aggfunc="sum", fill_value=0).reindex(bands).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(bands))
    for typ in ("LEVEL", "HEADING", "SPEED", "OTHER"):
        values = comp[typ].to_numpy() if typ in comp else np.zeros(len(bands))
        ax.bar(x, values, bottom=bottom, label=typ)
        bottom += values
    ax.set(xticks=x, xticklabels=bands, ylabel="Intervention windows", title="Clearance types by altitude band")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    save(2, "clearance_types_by_altitude_band", fig)

    top = rankings.loc[rankings["rank"] <= 3].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, band in zip(axes.flat, bands):
        part = top.loc[top.band == band].sort_values("effect_size")
        ax.barh(part.feature, part.effect_size)
        ax.set(title=band, xlabel="SMD (intervention minus control)")
    save(3, "top_effect_sizes_by_altitude_band", fig)

    for number, feature, name, title in ((4, "vertical_sep_at_event", "vertical_separation_by_band", "Vertical separation at event"),
                                          (5, "horizontal_sep_at_event", "horizontal_separation_by_band", "Horizontal separation at event")):
        fig, ax = plt.subplots(figsize=(9, 5))
        a = [comparison[b].set_index("feature").loc[feature, "intervention_median"] for b in bands]
        b = [comparison[bd].set_index("feature").loc[feature, "control_median"] for bd in bands]
        ax.bar(x-.18, a, width=.36, label="Intervention")
        ax.bar(x+.18, b, width=.36, label="Matched control")
        ax.set(xticks=x, xticklabels=bands, ylabel=feature, title=title)
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        save(number, name, fig)

    for number, details, name, title in (
        (6, [("min_cpa_within_30nm", "CPA (NM)"), ("tcpa_at_min_cpa", "TCPA (min)")],
         "cpa_tcpa_by_band", "CPA and TCPA by altitude band"),
        (7, [("max_horizontal_closing_within_30nm", "Horizontal closing within 30 NM (NM/min)"),
             ("max_vertical_closing_within_30nm", "Vertical closing within 30 NM (ft/min)")],
         "closing_rates_by_band", "Closing rates by altitude band")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (feature, label) in zip(axes, details):
            a = [comparison[b].set_index("feature").loc[feature, "intervention_median"] for b in bands]
            b = [comparison[bd].set_index("feature").loc[feature, "control_median"] for bd in bands]
            ax.bar(x-.18, a, width=.36, label="Intervention")
            ax.bar(x+.18, b, width=.36, label="Matched control")
            ax.set(xticks=x, xticklabels=bands, ylabel=label)
            ax.tick_params(axis="x", rotation=20)
        axes[0].legend()
        fig.suptitle(title)
        save(number, name, fig)

    cruise = comparison["CRUISE_FL290_410"].head(10).sort_values("smd")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(cruise.feature, cruise.smd)
    ax.set(xlabel="SMD (intervention minus control)", title="FL290–FL410: largest descriptive differences")
    save(8, "cruise_FL290_410_intervention_vs_control", fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cruise_val = validation.loc[(validation.band == "CRUISE_FL290_410") & (validation.method == "GMM_diag_BIC")]
    if len(cruise_val):
        assignments = pd.read_parquet(output / "cruise_cluster_assignments.parquet")
        sizes = assignments.cluster_id.value_counts().sort_index()
        ax.bar(sizes.index.astype(str), sizes.values)
        ax.set(xlabel="GMM component", ylabel="Intervention windows", title="FL290–FL410 component sizes (not conflict classes)")
    else:
        ax.text(.5, .5, "Insufficient data for clustering", ha="center", va="center", transform=ax.transAxes)
        ax.set(title="FL290–FL410 clustering")
    save(9, "cruise_clustering", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    global_sil = .059
    cruise_sil = float(cruise_val.silhouette.iloc[0]) if len(cruise_val) else np.nan
    ax.bar(["Global Stage 06", "FL290–FL410"], [global_sil, cruise_sil])
    ax.set(ylabel="Silhouette", title="Global versus cruise clustering quality")
    save(10, "global_vs_cruise_clustering_quality", fig)

    heat = rankings.pivot(index="feature", columns="band", values="effect_size").reindex(columns=bands)
    selected = rankings.loc[rankings["rank"] <= 5, "feature"].drop_duplicates().tolist()
    heat = heat.reindex(selected).fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(5, len(heat) * .35)))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-max(1, np.nanmax(abs(heat.to_numpy()))), vmax=max(1, np.nanmax(abs(heat.to_numpy()))))
    ax.set(xticks=np.arange(len(bands)), xticklabels=bands, yticks=np.arange(len(heat)), yticklabels=heat.index,
           title="Descriptive effect-size heatmap by altitude band")
    ax.tick_params(axis="x", rotation=20)
    fig.colorbar(image, ax=ax, label="SMD")
    save(11, "feature_importance_heatmap_by_band", fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    priority = recommendations.set_index("band").reindex(bands).recommended_processing_priority.fillna("UNDETERMINED")
    colors = {"HIGH": "#277da1", "MEDIUM": "#90be6d", "LOW": "#f9c74f", "UNDETERMINED": "#adb5bd"}
    ax.barh(bands[::-1], inv.valid_matched_pairs.reindex(bands[::-1]).to_numpy(), color=[colors.get(priority[b], "#adb5bd") for b in bands[::-1]])
    ax.set(xlabel="Valid SCAT pilot pairs", title="Pilot-supported OpenSky processing priority")
    save(12, "recommended_opensky_processing_by_band", fig)


def opensky_mapping(raw_path: Path, historical_path: Path, rankings: pd.DataFrame) -> pd.DataFrame:
    raw = set(raw_path.open(encoding="utf-8").readline().strip().split(","))
    historical = set(historical_path.open(encoding="utf-8").readline().strip().split(","))
    required = {"time", "icao24", "lat", "lon", "velocity", "heading", "vertrate", "baroaltitude"}
    if not required <= historical or not {"longitude", "latitude", "baro_altitude", "velocity", "true_track", "vertical_rate"} <= raw:
        raise SystemExit("Existing OpenSky schema lacks required historical/snapshot fields; do not invent a mapping")
    top = rankings.loc[rankings["rank"] <= 5].groupby("feature")["band"].agg(lambda x: ",".join(x)).to_dict()
    entries = [
        ("analysis_altitude_ft", "YES", "baroaltitude", "YES", "Convert historical barometric metres to feet; verify pressure reference", "YES", "Barometric OpenSky altitude is not identical to SCAT's measured FL"),
        ("horizontal_sep_at_event", "PARTIAL", "lat,lon,time,icao24", "YES", "Align concurrent aircraft and compute geodesic pair distance", "YES", "Needs complete, fresh states; no ATC event time in OpenSky"),
        ("median_horizontal_separation", "PARTIAL", "lat,lon,time,icao24", "YES", "Median pairwise distance over a defined lookback window", "TEST", "SCAT V1 included distant nearest-ten pairs; use a bounded-neighbour definition in OpenSky"),
        ("min_horizontal_sep_within_30nm", "PARTIAL", "lat,lon,time,icao24", "YES", "Pair distances over a pre-event window with an explicitly defined neighbourhood", "TEST", "OpenSky has no clearance anchor; use neutral reference times"),
        ("vertical_sep_at_event", "PARTIAL", "baroaltitude,time,icao24", "YES", "Absolute barometric-altitude difference in feet", "YES", "Missing altitude and pressure differences need quality controls"),
        ("median_vertical_separation", "PARTIAL", "baroaltitude,time,icao24", "YES", "Median pairwise altitude difference over the lookback window", "TEST", "Must use a consistent neighbourhood definition"),
        ("min_vertical_separation", "PARTIAL", "baroaltitude,time,icao24", "YES", "Minimum pairwise altitude difference over a window", "TEST", "Extrema depend on number of pair samples"),
        ("min_predicted_horizontal_cpa", "PARTIAL", "lat,lon,velocity,heading,time,icao24", "YES", "Minimum kinematic CPA over selected pairs", "TEST", "Unrestricted neighbour extrema can be dominated by distant aircraft"),
        ("min_cpa_within_30nm", "PARTIAL", "lat,lon,velocity,heading,time,icao24", "YES", "Local-plane constant-velocity CPA for contemporaneous states", "TEST", "Kinematic proxy, not TopSky prediction or conflict label"),
        ("tcpa_at_min_cpa", "PARTIAL", "lat,lon,velocity,heading,time,icao24", "YES", "Time to kinematic CPA from relative velocity", "TEST", "Near-zero relative velocity and sub-second extrema need filtering/diagnostics"),
        ("min_positive_tcpa", "PARTIAL", "lat,lon,velocity,heading,time,icao24", "YES", "Minimum positive kinematic TCPA over pairs/times", "NO", "Unstable extreme and often sub-second; do not use directly for model design"),
        ("min_tcpa_ge_1s_within_30nm", "PARTIAL", "lat,lon,velocity,heading,time,icao24", "YES", "Diagnostic TCPA among nearby pairs with at least one-second lead time", "TEST", "One-second floor and 30-NM radius are quality/selection choices, not conflict labels"),
        ("max_horizontal_closing_within_30nm", "PARTIAL", "lat,lon,velocity,heading,time,icao24", "YES", "Maximum line-of-sight closing among pairs currently within 30 NM", "TEST", "30-NM radius is a descriptive neighbourhood choice, not a conflict threshold"),
        ("max_vertical_closing_rate", "PARTIAL", "baroaltitude,vertrate,time,icao24", "YES", "Maximum pairwise vertical closing over all selected neighbours", "NO", "Unrestricted nearest-ten selection includes many pairs beyond 30 NM; use a bounded variant"),
        ("max_vertical_closing_within_30nm", "PARTIAL", "baroaltitude,vertrate,time,icao24", "YES", "Signed altitude-difference derivative from relative vertical rate", "TEST", "SCAT target metadata had 5-second bucket alignment discrepancies"),
        ("median_crossing_angle", "PARTIAL", "heading,time,icao24", "YES", "Angular difference between pair ground tracks", "TEST", "OpenSky heading is true track; SCAT crossing angle uses track"),
        ("median_relative_heading", "PARTIAL", "heading,time,icao24", "YES", "Ground-track angular difference as proxy", "TEST", "Not identical to SCAT magnetic-heading field"),
        ("median_relative_bearing", "PARTIAL", "lat,lon,time,icao24", "YES", "Bearing from focal to neighbour", "TEST", "Depends on focal-neighbour selection"),
        ("median_traffic_density", "PARTIAL", "time,icao24,lat,lon", "YES", "Count all quality-filtered contemporaneous states in spatial/temporal domain", "TEST", "Must retain full snapshots, not only selected pairs"),
        ("n_unique_neighbours_within_30nm", "PARTIAL", "time,icao24,lat,lon", "YES", "Count unique ICAO24 within 30 NM over the lookback window", "TEST", "Identity reuse and missing positions can bias counts; radius is descriptive"),
        ("ground_speed_at_anchor_kt", "YES", "velocity", "YES", "Convert historical m/s to knots at reference time", "YES", "OpenSky has no ATC event anchor; use a neutral reference time and check freshness"),
        ("vertical_rate_at_anchor_fpm", "YES", "vertrate", "YES", "Convert historical m/s to ft/min at reference time", "YES", "Some sources may have noisy climb/descent rate; OpenSky lacks ATC event time"),
        ("horizontal_sep_trend_60s", "PARTIAL", "time,lat,lon,icao24", "YES", "Current pair separation minus separation 60 seconds earlier", "TEST", "Requires same pair and adequate historical coverage"),
        ("horizontal_sep_trend_120s", "PARTIAL", "time,lat,lon,icao24", "YES", "Current pair separation minus separation 120 seconds earlier", "TEST", "Requires same pair and adequate historical coverage"),
        ("horizontal_sep_trend_300s", "PARTIAL", "time,lat,lon,icao24", "YES", "Current pair separation minus separation 300 seconds earlier", "TEST", "May be sparse near low-altitude boundaries"),
        ("horizontal_sep_trend_600s", "PARTIAL", "time,lat,lon,icao24", "YES", "Current pair separation minus separation 600 seconds earlier", "TEST", "Highest missingness in SCAT pilot"),
        ("vertical_sep_trend_600s", "PARTIAL", "time,baroaltitude,icao24", "YES", "Current pair altitude difference minus that 600 seconds earlier", "TEST", "High and asymmetric missingness in low-altitude SCAT windows"),
        ("min_topsky_predicted_sep_nm_proxy", "NO", "", "NO", "Not exactly derivable; test a separate constant-velocity or learned-trajectory proxy", "NO", "TopSky trajectory is SCAT-only; do not call a kinematic proxy equivalent"),
        ("clearance_type", "NO", "", "NO", "No ATC clearance field in inspected OpenSky state schemas", "NO", "Cannot characterize OpenSky cases by clearance type directly"),
        ("ATC_INTERVENTION", "NO", "", "NO", "No controller-intervention label in inspected OpenSky state schemas", "NO", "SCAT-supervised intervention model cannot be validated on OpenSky without external labels"),
        ("flight_phase", "PARTIAL", "baroaltitude,vertrate", "YES", "Approximate altitude/climb-state proxy", "TEST", "Not the same operational phase annotation as SCAT"),
        ("centre", "NO", "", "NO", "Would require external sector geometry and time-dependent operational context", "NO", "Not in inspected state-vector schemas"),
    ]
    mapping = pd.DataFrame([{"SCAT_feature": feature, "importance_by_band": top.get(feature, ""),
        "available_in_opensky": available, "opensky_source_column": source,
        "derivable": derivable, "derivation": derivation, "recommended_for_opensky": recommend,
        "limitations": limitation} for feature, available, source, derivable, derivation, recommend, limitation in entries])
    missing_top = set(top) - set(mapping.SCAT_feature)
    if missing_top:
        raise SystemExit(f"OpenSky mapping lacks pilot top features: {sorted(missing_top)}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_OUTPUT_ROOT / "05_features" / f"window_features_v2_{PILOT_WEEK}.parquet")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_OUTPUT_ROOT / "04_windows" / f"primary_matching_cohort_{PILOT_WEEK}.parquet")
    parser.add_argument("--clearances", type=Path, default=DEFAULT_OUTPUT_ROOT / "03_clearances" / "scat_clearances.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "06_exploration")
    args = parser.parse_args()
    if args.features.name != f"window_features_v2_{PILOT_WEEK}.parquet" or args.cohort.name != f"primary_matching_cohort_{PILOT_WEEK}.parquet":
        raise SystemExit("Altitude exploration requires validated first-week V2 and primary cohort")
    assert_pilot_parquet(args.features)
    windows = pd.read_parquet(args.features)
    cohort = pd.read_parquet(args.cohort)
    assert_pilot_frame(windows)
    assert_pilot_frame(cohort)
    if windows.window_id.duplicated().any() or cohort.intervention_window_id.duplicated().any() or cohort.control_window_id.duplicated().any():
        raise SystemExit("Window or pair IDs are not unique")
    if set(FEATURES + CLUSTER_FEATURES) - set(windows.columns):
        raise SystemExit("V2 is missing a required continuous feature")
    if set(windows.altitude_band) - set(BANDS):
        raise SystemExit("Unknown V2 altitude band")
    expected = set(cohort.intervention_window_id) | set(cohort.control_window_id)
    if expected != set(windows.window_id) or len(windows) != len(expected):
        raise SystemExit("V2 does not exactly cover the first-week primary cohort")
    if ((windows.window_id.str.startswith("I:")) != windows.ATC_INTERVENTION.eq(1)).any():
        raise SystemExit("Intervention/control labels do not match IDs")
    if windows.altitude_anchor_relative_seconds.gt(0).any():
        raise SystemExit("Post-event altitude anchor detected")
    source_types = pd.read_parquet(args.clearances, columns=["archive_week", "clearance_id", "clearance_type"],
                                   filters=[("archive_week", "==", PILOT_WEEK)])
    assert_pilot_frame(source_types)
    types = source_types[["clearance_id", "clearance_type"]].drop_duplicates("clearance_id")
    one = windows.add_suffix("_I")
    two = windows.add_suffix("_C")
    wide = cohort.merge(one, left_on="intervention_window_id", right_on="window_id_I", validate="one_to_one")
    wide = wide.merge(two, left_on="control_window_id", right_on="window_id_C", validate="one_to_one")
    if len(wide) != len(cohort):
        raise SystemExit("A primary matched pair lost a window")
    wide["same_band_match"] = wide.altitude_band_I.eq(wide.altitude_band_C) & wide.altitude_band_I.ne("UNKNOWN_INVALID")
    cross = wide.loc[wide.altitude_band_I.ne(wide.altitude_band_C)].copy()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pair_columns = ["clearance_id", "intervention_window_id", "control_window_id", "matching_level",
                    "analysis_altitude_ft_I", "analysis_altitude_ft_C", "altitude_band_I", "altitude_band_C",
                    "same_band_match", "control_time_difference_s"]
    cross[pair_columns].rename(columns={"analysis_altitude_ft_I": "intervention_altitude_ft",
        "analysis_altitude_ft_C": "control_altitude_ft", "altitude_band_I": "intervention_altitude_band",
        "altitude_band_C": "control_altitude_band"}).to_csv(output / "cross_band_pairs.csv", index=False)
    wide.loc[wide.altitude_band_I.eq("UNKNOWN_INVALID") & wide.altitude_band_C.eq("UNKNOWN_INVALID"), pair_columns].to_csv(
        output / "invalid_altitude_pairs.csv", index=False)
    valid = wide.loc[wide.same_band_match].copy()
    i_windows = windows.loc[windows.ATC_INTERVENTION == 1].copy()
    c_windows = windows.loc[windows.ATC_INTERVENTION == 0].copy()
    intervention_types = i_windows[["window_id", "clearance_id", "altitude_band"]].merge(types, on="clearance_id", how="left", validate="many_to_one")
    intervention_types["clearance_type"] = intervention_types.clearance_type.where(
        intervention_types.clearance_type.isin(["LEVEL", "HEADING", "SPEED"]), "OTHER")
    inventory = []
    comparison = {}
    rankings = []
    horizon_rows = []
    balance_rows = []
    validations = []
    summaries = []
    comp_types = []
    quality = {}
    evidence = {}
    for band in BANDS:
        band_pairs = valid.loc[valid.altitude_band_I == band]
        ids = set(band_pairs.intervention_window_id) | set(band_pairs.control_window_id)
        band_frame = windows.loc[windows.window_id.isin(ids)].copy()
        if len(band_frame) != 2 * len(band_pairs):
            raise SystemExit(f"Pair-preserving sample mismatch in {band}")
        intervention_ids = set(band_pairs.intervention_window_id)
        chosen_types = intervention_types.loc[intervention_types.window_id.isin(intervention_ids)]
        counts = chosen_types.clearance_type.value_counts()
        n = len(band_pairs)
        alt = band_frame.analysis_altitude_ft
        inventory.append({"band": band, "intervention_windows": int(i_windows.altitude_band.eq(band).sum()),
            "control_windows": int(c_windows.altitude_band.eq(band).sum()), "valid_matched_pairs": n,
            "cross_band_excluded": int(cross.altitude_band_I.eq(band).sum()),
            "unique_focal_flights": int(band_frame.focal_flight_key.nunique()),
            **{f"clearance_{typ}": int(counts.get(typ, 0)) for typ in ("LEVEL", "HEADING", "SPEED", "OTHER")},
            **{f"percentage_{typ}": float(counts.get(typ, 0) / n * 100) if n else np.nan for typ in ("LEVEL", "HEADING", "SPEED")},
            "median_altitude": alt.median(), "p05_altitude": alt.quantile(.05), "p95_altitude": alt.quantile(.95),
            "traffic_density_median": band_frame.median_traffic_density.median(),
            "unique_neighbours_median": band_frame.n_unique_neighbours_within_30nm.median(),
            "percentage_of_total_interventions": float(i_windows.altitude_band.eq(band).mean() * 100),
            "percentage_of_total_primary_pairs": float(n / len(cohort) * 100)})
        if n == 0:
            quality[band] = "UNDETERMINED"
            evidence[band] = "INSUFFICIENT_DATA"
            continue
        bal = matching_balance(band_pairs, band)
        balance_rows.append(bal)
        statuses = set(bal.loc[bal.status != "DIAGNOSTIC", "status"])
        quality[band] = "FAIL" if "FAIL" in statuses else "WARNING" if "WARNING" in statuses else "UNDETERMINED" if "UNDETERMINED" in statuses else "PASS"
        for typ, count in counts.items():
            comp_types.append({"band": band, "clearance_type": typ, "count": int(count), "percentage": float(count / n * 100)})
        if band not in ANALYSIS_BANDS:
            evidence[band] = "INSUFFICIENT_DATA"
            continue
        desc, comp = describe(band_frame)
        desc.to_csv(output / ("cruise_FL290_410_descriptive_statistics.csv" if band == "CRUISE_FL290_410" else f"descriptive_{band}.csv"), index=False)
        comp.to_csv(output / ("cruise_FL290_410_intervention_vs_control.csv" if band == "CRUISE_FL290_410" else f"intervention_vs_control_{band}.csv"), index=False)
        comparison[band] = comp
        ranked = comp.loc[comp.intervention_count.ge(30) & comp.control_count.ge(30) & comp.smd.notna()].copy()
        ranked["band"] = band
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["effect_size"] = ranked.smd
        ranked["direction"] = np.where(ranked.smd.gt(0), "HIGHER_WITH_INTERVENTION", "LOWER_WITH_INTERVENTION")
        rankings.append(ranked[["band", "rank", "feature", "effect_size", "direction", "intervention_count", "control_count"]])
        horizon_rows.append(horizon_table(band_pairs, band))
        val, summary, label = cluster_band(band_frame.loc[band_frame.ATC_INTERVENTION == 1].reset_index(drop=True), band, output, types)
        validations.append(val)
        if not summary.empty:
            summaries.append(summary)
        evidence[band] = label
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(output / "altitude_band_inventory.csv", index=False)
    balance_df = pd.concat(balance_rows, ignore_index=True)
    balance_df.to_csv(output / "altitude_band_matching_balance.csv", index=False)
    composition_df = pd.DataFrame(comp_types)
    composition_df.to_csv(output / "altitude_band_clearance_composition.csv", index=False)
    global_desc, global_comp = describe(windows)
    global_desc.to_csv(output / "global_v2_descriptive_statistics.csv", index=False)
    global_comp.to_csv(output / "global_v2_intervention_vs_control.csv", index=False)
    ranked_df = pd.concat(rankings, ignore_index=True)
    ranked_df.to_csv(output / "altitude_band_feature_rankings.csv", index=False)
    feature_comparison = pd.DataFrame({"feature": FEATURES})
    feature_comparison["GLOBAL_effect"] = feature_comparison.feature.map(global_comp.set_index("feature").smd)
    for band in ANALYSIS_BANDS:
        feature_comparison[f"{band}_effect"] = feature_comparison.feature.map(comparison[band].set_index("feature").smd)
    feature_comparison.to_csv(output / "altitude_band_feature_comparison.csv", index=False)
    horizon_df = pd.concat(horizon_rows, ignore_index=True)
    horizon_df.to_csv(output / "altitude_band_horizon_comparison.csv", index=False)
    validation_df = pd.concat(validations, ignore_index=True)
    validation_df.to_csv(output / "altitude_band_cluster_validation.csv", index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output / "altitude_band_cluster_summary.csv", index=False)
    key_features = ["horizontal_sep_at_event", "vertical_sep_at_event", "min_cpa_within_30nm",
        "tcpa_at_min_cpa", "max_horizontal_closing_within_30nm", "max_vertical_closing_within_30nm",
        "median_crossing_angle", "median_traffic_density", "n_unique_neighbours_within_30nm",
        "ground_speed_at_anchor_kt", "vertical_rate_at_anchor_fpm", "min_topsky_predicted_sep_nm_proxy"]
    summary_rows = []
    for band in ["GLOBAL", *ANALYSIS_BANDS]:
        part = global_comp if band == "GLOBAL" else comparison[band]
        indexed = part.set_index("feature")
        n = len(cohort) if band == "GLOBAL" else int(inventory_df.set_index("band").loc[band, "valid_matched_pairs"])
        type_counts = intervention_types.clearance_type.value_counts() if band == "GLOBAL" else composition_df.loc[composition_df.band == band].set_index("clearance_type")["count"]
        model_row = validation_df.loc[(validation_df.band == band) & (validation_df.method == "GMM_diag_BIC")]
        row = {"band": band, "analysis_scope": "FIRST_WEEK_PILOT", "matched_pairs": n,
               "intervention_windows": n, "control_windows": n,
               "matching_status": "PASS_GLOBAL_STAGE04_ONLY" if band == "GLOBAL" else quality[band],
               "dominant_clearance_type": type_counts.idxmax(),
               "clustering_evidence": "WEAK_V1_BASELINE" if band == "GLOBAL" else evidence[band],
               "gmm_silhouette": .059 if band == "GLOBAL" else float(model_row.silhouette.iloc[0]) if len(model_row) else np.nan,
               "cluster_score_comparability": "V1 global; not like-for-like with V2 bands" if band == "GLOBAL" else "V2 band"}
        for typ in ("LEVEL", "HEADING", "SPEED", "OTHER"):
            row[f"clearance_{typ}_pct"] = float(type_counts.get(typ, 0) / n * 100)
        for feature in key_features:
            row[f"{feature}_intervention_median"] = indexed.loc[feature, "intervention_median"]
            row[f"{feature}_control_median"] = indexed.loc[feature, "control_median"]
            row[f"{feature}_SMD"] = indexed.loc[feature, "smd"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / "global_vs_altitude_band_summary.csv", index=False)
    recommendations = []
    for band in ANALYSIS_BANDS:
        inv = inventory_df.set_index("band").loc[band]
        n = int(inv.valid_matched_pairs)
        q = quality[band]
        main_type = max(("LEVEL", "HEADING", "SPEED", "OTHER"), key=lambda typ: inv[f"clearance_{typ}"])
        raw_top = comparison[band].copy()
        raw_top["missing_i_share"] = raw_top.intervention_missing / n
        raw_top["missing_c_share"] = raw_top.control_missing / n
        candidate = raw_top.loc[raw_top.missing_i_share.le(.2) & raw_top.missing_c_share.le(.2)
            & (raw_top.missing_i_share - raw_top.missing_c_share).abs().le(.1)
            & ~raw_top.feature.isin(["min_positive_tcpa", "max_vertical_closing_rate",
                                     "min_predicted_horizontal_cpa", "far30_pair_fraction"])]
        top = candidate.feature.head(5).tolist()
        priority = "HIGH" if n >= 2000 and q in {"PASS", "WARNING"} else "MEDIUM" if n >= 500 and q in {"PASS", "WARNING"} else "LOW" if n < 300 else "UNDETERMINED"
        horizon = recommended_horizon(horizon_df.loc[horizon_df.band == band])
        recommendations.append({"band": band, "SCAT_sample_size": n, "SCAT_matching_quality": q,
            "dominant_clearance_type": main_type,
            **{f"top_feature_{i}": top[i-1] if len(top) >= i else "" for i in range(1, 6)},
            "recommended_time_horizon": horizon,
            "recommended_required_features": "time,icao24,lat,lon,baroaltitude,velocity,heading,vertrate; aligned full state snapshots",
            "recommended_optional_features": "ground-track geometry, kinematic CPA/TCPA and trend proxies, only after quality/sensitivity tests",
            "recommended_processing_priority": priority, "separate_model_recommended": "UNDETERMINED",
            "clustering_structure": evidence[band],
            "notes": "Top five are quality-screened descriptive SCAT candidates; raw |SMD| ranking is separate. OpenSky lacks ATC intervention labels and TopSky predictions; no model transfer without external labels"})
    recommendation_df = pd.DataFrame(recommendations)
    recommendation_df.to_csv(output / "opensky_processing_recommendations.csv", index=False)
    data_root = DEFAULT_OUTPUT_ROOT.parent.parent
    raw_opensky = data_root / "raw" / "opensky" / "opensky_snapshot.csv"
    historical = data_root / "processed" / "opensky_10s" / "LECMSAN" / "year=2019" / "month=01" / "lecmsan_2019-01_10s.csv"
    mapped_features = opensky_mapping(raw_opensky, historical, ranked_df)
    recommended_top = set(recommendation_df[[f"top_feature_{i}" for i in range(1, 6)]].stack().dropna())
    if recommended_top - set(mapped_features.SCAT_feature):
        raise SystemExit(f"OpenSky mapping lacks recommended features: {sorted(recommended_top - set(mapped_features.SCAT_feature))}")
    mapped_features.to_csv(output / "scat_to_opensky_feature_mapping.csv", index=False)
    figure_set(output, inventory_df, comparison, composition_df, validation_df, ranked_df, recommendation_df)
    summary = {**scope_config(), "input_v2": str(args.features), "primary_pairs": len(cohort),
        "same_band_pairs": len(valid), "cross_band_pairs": len(cross),
        "invalid_same_band_pairs": int((~wide.same_band_match & wide.altitude_band_I.eq(wide.altitude_band_C)).sum()),
        "band_matching_quality": quality, "band_clustering_evidence": evidence,
        "recommended_horizons": {band: recommended_horizon(horizon_df.loc[horizon_df.band == band]) for band in ANALYSIS_BANDS},
        "global_stage06_baseline_silhouette": .059, "clearance_not_conflict": True,
        "stage07_executed": False, "other_scat_weeks_used": False,
        "minimum_cluster_interventions": 300,
        "source_opensky_schemas": [str(raw_opensky), str(historical)]}
    (output / "altitude_stratified_run_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    update_read_me_first(output.parent)
    print(f"Altitude Stage 06 complete: same-band pairs={len(valid):,}, cross-band={len(cross):,}, quality={quality}", flush=True)


if __name__ == "__main__":
    main()
