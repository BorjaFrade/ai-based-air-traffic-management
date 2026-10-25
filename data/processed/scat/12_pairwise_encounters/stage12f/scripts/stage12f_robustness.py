"""Stage 12F robustness audit of the frozen Stage 12 physical outputs.

Reads only the Stage 12E structure samples and episode summaries.  It does not
rebuild candidates, pair tracks, or modify any upstream artifact.
"""
from __future__ import annotations

import json
import os
import resource
import time
import tracemalloc
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

OUT = Path(__file__).resolve().parents[2]
WEEKS = ("20161015_20161021", "20161112_20161118")
SEED = 12014
FULL = ["log_min_h", "log_simultaneous_v", "start_closing", "crossing_angle",
        "fraction_converging", "relative_speed", "track_change", "altitude_change"]
GEOMETRY = ["log_min_h", "log_simultaneous_v", "start_closing", "crossing_angle", "relative_speed"]
RATE_SET = GEOMETRY + ["track_rate_deg_min", "altitude_rate_ft_min"]
JOIN_COLS = ["episode_id", "a_flight_key", "b_flight_key", "a_phase", "b_phase",
             "n_samples", "max_state_age_s", "fully_bracketed_at_100nm",
             "temporally_censored", "quality_for_structure", "a_track_change_deg",
             "b_track_change_deg", "a_altitude_change_ft", "b_altitude_change_ft"]


def phase_pair(frame: pd.DataFrame) -> pd.Series:
    a = frame["a_phase"].fillna("UNKNOWN").astype(str)
    b = frame["b_phase"].fillna("UNKNOWN").astype(str)
    return pd.Series(np.where(a <= b, a + " | " + b, b + " | " + a), index=frame.index)


def flight_edge_split(frame: pd.DataFrame, test_fraction: float, seed: int):
    """Split vertices (flights), retaining only edges wholly within one side."""
    flights = pd.unique(pd.concat([frame.a_flight_key, frame.b_flight_key], ignore_index=True).dropna())
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(flights) * test_fraction)))
    test_flights = set(rng.choice(flights, size=min(n_test, len(flights)), replace=False).tolist())
    train_flights = set(flights) - test_flights
    train = frame.a_flight_key.isin(train_flights) & frame.b_flight_key.isin(train_flights)
    test = frame.a_flight_key.isin(test_flights) & frame.b_flight_key.isin(test_flights)
    dropped = ~(train | test)
    assert not (train_flights & test_flights)
    assert not set(frame.loc[train, "a_flight_key"]) & set(frame.loc[test, "a_flight_key"])
    assert not set(frame.loc[train, "b_flight_key"]) & set(frame.loc[test, "b_flight_key"])
    assert not set(pd.concat([frame.loc[train, "a_flight_key"], frame.loc[train, "b_flight_key"]])) & \
               set(pd.concat([frame.loc[test, "a_flight_key"], frame.loc[test, "b_flight_key"]]))
    return train.to_numpy(), test.to_numpy(), dropped.to_numpy(), train_flights, test_flights


def raw_matrix(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    x = frame[FULL].copy()
    if feature_set == "geometry_only":
        return x[GEOMETRY].copy()
    if feature_set == "geometry_plus_rates":
        duration_min = frame.observed_duration_s.clip(lower=30).div(60)
        x["track_rate_deg_min"] = frame.track_change.clip(lower=0).div(duration_min)
        x["altitude_rate_ft_min"] = frame.altitude_change.clip(lower=0).div(duration_min)
        return x[RATE_SET].copy()
    if feature_set == "full":
        return x[FULL].copy()
    raise ValueError(feature_set)


def fit_scale(train: pd.DataFrame, other: pd.DataFrame, scaling: str):
    scaler = RobustScaler(quantile_range=(10, 90)) if scaling == "robust" else StandardScaler()
    z_train = np.clip(scaler.fit_transform(train), -8, 8)
    z_other = np.clip(scaler.transform(other), -8, 8)
    return z_train, z_other


def ward(frame: pd.DataFrame, features: list[str], seed: int = SEED,
         pilot_reference: pd.DataFrame | None = None, scaling: str = "robust"):
    """Fixed 1,200-row Ward K=2 reproduction; labels are returned per row."""
    ix = np.random.default_rng(seed).choice(len(frame), min(1200, len(frame)), replace=False)
    use = frame.iloc[ix].reset_index(drop=True)
    x = raw_matrix(use, "geometry_only" if features == GEOMETRY else
                   "geometry_plus_rates" if features == RATE_SET else "full")
    ref = x if pilot_reference is None else pilot_reference[features]
    scaler = RobustScaler(quantile_range=(10, 90)) if scaling == "robust" else StandardScaler()
    scaler.fit(ref)
    z = np.clip(scaler.transform(x), -8, 8)
    labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(z)
    return use, x, z, labels


def summarize_episode_groups(frame: pd.DataFrame, labels: np.ndarray, week: str,
                             method: str, subset: str) -> list[dict]:
    counts = np.bincount(labels, minlength=2)
    minority = int(np.argmin(counts))
    rows = []
    frame = frame.reset_index(drop=True)
    for cluster in range(2):
        d = frame.loc[labels == cluster]
        row = dict(week=week, method=method, subset=subset, cluster=cluster,
                   role="minority" if cluster == minority else "majority", n=len(d),
                   share=len(d) / len(frame))
        for col in ["min_horizontal_nm", "vertical_at_min_horizontal_ft", "observed_duration_s",
                    "n_samples", "max_state_age_s"]:
            if col in d:
                row[f"median_{col}"] = float(d[col].median())
        for col, name in [("track_change", "track_change_deg"), ("altitude_change", "altitude_change_ft")]:
            if col in d:
                row[f"median_{name}"] = float(d[col].median())
        if "track_change" not in d:
            row["median_track_change_deg"] = float(np.maximum(d.a_track_change_deg.abs(), d.b_track_change_deg.abs()).median())
            row["median_altitude_change_ft"] = float(np.maximum(d.a_altitude_change_ft.abs(), d.b_altitude_change_ft.abs()).median())
        pp = phase_pair(d)
        row["modal_phase_pair"] = str(pp.value_counts().index[0]) if len(pp) else ""
        row["modal_phase_pair_share"] = float(pp.value_counts(normalize=True).iloc[0]) if len(pp) else np.nan
        rows.append(row)
    return rows


def duration_match(frame: pd.DataFrame, labels: np.ndarray, week: str) -> tuple[dict, list[dict]]:
    """Greedy 1:1 nearest matches within symmetric phase pair on exposure proxies."""
    d = frame.reset_index(drop=True).copy()
    d["_label"] = labels
    counts = np.bincount(labels, minlength=2)
    minor_label = int(np.argmin(counts))
    d["_phase_pair"] = phase_pair(d)
    d["_log_duration"] = np.log1p(d.observed_duration_s.clip(lower=0))
    d["_log_samples"] = np.log1p(d.n_samples.clip(lower=0))
    d["_age"] = d.max_state_age_s.fillna(d.max_state_age_s.median())
    cols = ["_log_duration", "_log_samples", "_age"]
    scale = d[cols].quantile(.75) - d[cols].quantile(.25)
    scale = scale.replace(0, 1).fillna(1)
    used_controls: set[int] = set()
    pairs = []
    for i in d.index[d._label.eq(minor_label)]:
        candidates = d.index[d._label.ne(minor_label) & d._phase_pair.eq(d.at[i, "_phase_pair"])].difference(list(used_controls))
        if not len(candidates):
            continue
        delta = (d.loc[candidates, cols].to_numpy() - d.loc[i, cols].to_numpy()) / scale.to_numpy()
        distances = np.sqrt((delta ** 2).mean(axis=1))
        j = int(candidates[int(np.argmin(distances))])
        distance = float(np.min(distances))
        used_controls.add(j)
        if distance <= 2.0:
            pairs.append((int(i), j, distance))
    rows = []
    raw_features = {
        "duration_s": "observed_duration_s", "n_samples": "n_samples", "state_age_s": "max_state_age_s",
        "min_horizontal_nm": "min_horizontal_nm", "vertical_ft": "vertical_at_min_horizontal_ft",
        "track_change_deg": None, "altitude_change_ft": None,
    }
    for pi, (i, j, dist) in enumerate(pairs):
        a, b = d.loc[i], d.loc[j]
        rows.append(dict(week=week, match=pi, minority_episode_id=a.episode_id,
                         control_episode_id=b.episode_id, phase_pair=a._phase_pair,
                         scaled_match_distance=dist,
                         duration_minority=a.observed_duration_s, duration_control=b.observed_duration_s,
                         n_samples_minority=a.n_samples, n_samples_control=b.n_samples,
                         age_minority=a.max_state_age_s, age_control=b.max_state_age_s,
                         track_change_minority=max(abs(a.a_track_change_deg), abs(a.b_track_change_deg)),
                         track_change_control=max(abs(b.a_track_change_deg), abs(b.b_track_change_deg)),
                         altitude_change_minority=max(abs(a.a_altitude_change_ft), abs(a.b_altitude_change_ft)),
                         altitude_change_control=max(abs(b.a_altitude_change_ft), abs(b.b_altitude_change_ft)),
                         min_h_minority=a.min_horizontal_nm, min_h_control=b.min_horizontal_nm,
                         vertical_minority=a.vertical_at_min_horizontal_ft,
                         vertical_control=b.vertical_at_min_horizontal_ft))
    out = pd.DataFrame(rows)
    if len(out):
        out["track_change_pair_difference"] = out.track_change_minority - out.track_change_control
        out["altitude_change_pair_difference"] = out.altitude_change_minority - out.altitude_change_control
    summary = dict(week=week, n_minority=int(counts.min()), n_majority=int(counts.max()),
                   n_matched=len(out), matched_share=float(len(out) / max(1, counts.min())),
                   caliper_scaled_distance=2.0, matching="greedy 1:1 without replacement; exact symmetric phase-pair stratum; log duration, log n_samples, max state age")
    if len(out):
        summary.update(median_track_change_difference=float(out.track_change_pair_difference.median()),
                       median_altitude_change_difference=float(out.altitude_change_pair_difference.median()),
                       median_duration_absolute_difference=float((out.duration_minority-out.duration_control).abs().median()),
                       median_samples_absolute_difference=float((out.n_samples_minority-out.n_samples_control).abs().median()))
    return summary, rows


def censor_audit(week: str, summary: pd.DataFrame):
    d = summary.copy()
    d["track_change_deg"] = np.maximum(d.a_track_change_deg.abs(), d.b_track_change_deg.abs())
    d["altitude_change_ft"] = np.maximum(d.a_altitude_change_ft.abs(), d.b_altitude_change_ft.abs())
    d["start_altitude_mean_ft"] = (d.start_a_alt_ft + d.start_b_alt_ft) / 2
    metrics = ["n_samples", "observed_duration_s", "min_horizontal_nm", "vertical_at_min_horizontal_ft",
               "track_change_deg", "altitude_change_ft", "start_altitude_mean_ft", "max_state_age_s"]
    rows = []
    for flag, label in [(True, "both_100nm_crossings_observed"), (False, "censored_or_unbracketed")]:
        g = d.loc[d.fully_bracketed_at_100nm.eq(flag)]
        for col in metrics:
            s = g[col].dropna()
            rows.append(dict(week=week, boundary_status=label, n=len(g), metric=col,
                             median=float(s.median()), q25=float(s.quantile(.25)), q75=float(s.quantile(.75)),
                             mean=float(s.mean()), sd=float(s.std(ddof=1))))
    wide = pd.DataFrame(rows)
    effect = []
    for metric in metrics:
        a = d.loc[d.fully_bracketed_at_100nm, metric].dropna()
        b = d.loc[~d.fully_bracketed_at_100nm, metric].dropna()
        pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(1, len(a)+len(b)-2))
        effect.append(dict(week=week, metric=metric, bracketed_n=len(a), censored_n=len(b),
                           bracketed_median=float(a.median()), censored_median=float(b.median()),
                           standardized_mean_difference=float((a.mean()-b.mean())/pooled) if pooled else np.nan))
    phases = d.assign(phase_pair=phase_pair(d)).groupby(["fully_bracketed_at_100nm", "phase_pair"], dropna=False).size().rename("n").reset_index()
    phases["week"] = week
    phases["boundary_status"] = np.where(phases.fully_bracketed_at_100nm,
                                         "both_100nm_crossings_observed", "censored_or_unbracketed")
    phases["share_within_status"] = phases.n / phases.groupby("fully_bracketed_at_100nm").n.transform("sum")
    phases = phases.drop(columns="fully_bracketed_at_100nm")
    return wide, pd.DataFrame(effect), phases


def load_inputs():
    datasets = {}
    full = {}
    for week in WEEKS:
        sample = pd.read_parquet(OUT / "tables" / f"{week}_structure_sample.parquet")
        summary = pd.read_parquet(OUT / "episodes" / f"{week}_episode_summary.parquet")
        meta = summary[JOIN_COLS]
        d = sample.merge(meta, on="episode_id", how="left", validate="one_to_one")
        if d["a_flight_key"].isna().any() or d["b_flight_key"].isna().any():
            raise ValueError(f"Flight identity join failed for {week}")
        # Explicitly check that the structure sample really represents the frozen complete cohort.
        assert d.fully_bracketed_at_100nm.all()
        assert d.quality_for_structure.all()
        d["track_change"] = np.maximum(d.a_track_change_deg.abs(), d.b_track_change_deg.abs()).clip(0, 180)
        d["altitude_change"] = np.maximum(d.a_altitude_change_ft.abs(), d.b_altitude_change_ft.abs()).clip(0, 20000)
        d["observed_duration_s"] = d.observed_duration_s
        datasets[week] = d.reset_index(drop=True)
        full[week] = summary
    return datasets, full


def main():
    out_tables = OUT / "stage12f" / "tables"
    out_diag = OUT / "stage12f" / "diagnostics"
    out_figs = OUT / "stage12f" / "figures"
    for p in (out_tables, out_diag, out_figs):
        p.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tracemalloc.start()
    datasets, full = load_inputs()
    # The pilot robust scale is fit on the same 10k frozen structure sample as Stage 12E.
    full_x = {w: raw_matrix(datasets[w], "full") for w in WEEKS}
    pilot_scaler = RobustScaler(quantile_range=(10, 90)).fit(full_x[WEEKS[0]])
    reproduced, profiles, sensitivity, correlations, phase_rows = [], [], [], [], []
    matched_summaries, matched_rows = [], []

    # Exact earlier K=2 pilot/replica reproduction (seed 12014, n=1,200).
    ward_frames = {}
    for week in WEEKS:
        d = datasets[week]
        x = raw_matrix(d, "full")
        idx = np.random.default_rng(SEED).choice(len(d), min(1200, len(d)), replace=False)
        part = d.iloc[idx].reset_index(drop=True)
        z = np.clip(pilot_scaler.transform(x.iloc[idx]), -8, 8)
        labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(z)
        counts = np.bincount(labels, minlength=2)
        reproduced.append(dict(week=week, cohort="Stage12E 10k fully bracketed quality sample; fixed 1,200-row subsample",
                              seed=SEED, n=len(part), silhouette=float(silhouette_score(z, labels)),
                              cluster_0_n=int(counts[0]), cluster_1_n=int(counts[1]),
                              minority_n=int(counts.min()), minority_share=float(counts.min()/len(part)),
                              scaler="RobustScaler q10-q90 fit on pilot 10k; clip [-8,8]", linkage="Ward", k=2))
        ward_frames[week] = (part, z, labels)
        profiles.extend(summarize_episode_groups(part, labels, week, "ward_full_robust_k2", "fixed_1200"))
        match_summary, match_rows = duration_match(part, labels, week)
        matched_summaries.append(match_summary)
        matched_rows.extend(match_rows)

        # Recomputed rank correlations and duration-normalized rates in original units.
        track = part.track_change
        altitude = part.altitude_change
        duration = part.observed_duration_s
        all_duration = d.observed_duration_s
        all_track = d.track_change
        all_altitude = d.altitude_change
        all_rates = pd.DataFrame({"track_rate_deg_min": all_track / (all_duration.clip(lower=30)/60),
                                  "altitude_rate_ft_min": all_altitude / (all_duration.clip(lower=30)/60)})
        for feature, values in [("track_change_deg", all_track), ("altitude_change_ft", all_altitude),
                                ("track_rate_deg_min", all_rates.track_rate_deg_min),
                                ("altitude_rate_ft_min", all_rates.altitude_rate_ft_min),
                                ("n_samples", d.n_samples), ("max_state_age_s", d.max_state_age_s)]:
            rho, pval = spearmanr(all_duration, values, nan_policy="omit")
            correlations.append(dict(week=week, sample_n=len(d), feature=feature,
                                    spearman_rho=float(rho), p_value=float(pval)))
        minority_label = int(np.argmin(np.bincount(labels, minlength=2)))
        part = part.assign(_cluster=labels, _phase_pair=phase_pair(part))
        for cluster, g in part.groupby("_cluster"):
            for pp, n_pp in g._phase_pair.value_counts().items():
                phase_rows.append(dict(week=week, cluster=int(cluster),
                                       role="minority" if int(cluster) == minority_label else "majority",
                                       phase_pair=pp, n=int(n_pp), share=float(n_pp/len(g))))

    # Ward sensitivity: feature groups and scaling fit on pilot; same sample rows and seed.
    for feature_set in ("full", "geometry_only", "geometry_plus_rates"):
        matrices = {w: raw_matrix(datasets[w], feature_set) for w in WEEKS}
        for scaling in ("robust", "standard"):
            scaler = RobustScaler(quantile_range=(10, 90)) if scaling == "robust" else StandardScaler()
            scaler.fit(matrices[WEEKS[0]])
            for week in WEEKS:
                d = datasets[week]
                idx = np.random.default_rng(SEED).choice(len(d), min(1200, len(d)), replace=False)
                x = matrices[week].iloc[idx]
                z = np.clip(scaler.transform(x), -8, 8)
                labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(z)
                counts = np.bincount(labels, minlength=2)
                sensitivity.append(dict(week=week, method="Ward", feature_set=feature_set, scaling=scaling,
                                        n=len(idx), silhouette=float(silhouette_score(z, labels)),
                                        minority_n=int(counts.min()), minority_share=float(counts.min()/len(idx))))

    # Flight-vertex split: all episodes belonging to any evaluation flight stay out of training.
    grouped_rows, grouped_profiles, gmm_rows = [], [], []
    for week in WEEKS:
        d = datasets[week]
        x = raw_matrix(d, "full")
        for split_seed in (SEED, SEED+1, SEED+2):
            tr_mask, te_mask, dropped_mask, train_flights, test_flights = flight_edge_split(d, .30, split_seed)
            train_d, test_d = d.loc[tr_mask].reset_index(drop=True), d.loc[te_mask].reset_index(drop=True)
            if len(train_d) < 100 or len(test_d) < 30:
                raise RuntimeError(f"Insufficient retained flight-disjoint edges in {week}, seed={split_seed}")
            xtr, xte = x.loc[tr_mask].reset_index(drop=True), x.loc[te_mask].reset_index(drop=True)
            ztr, zte = fit_scale(xtr, xte, "robust")
            # Bound Ward's quadratic memory while selecting train rows only.
            itrain = np.random.default_rng(split_seed).choice(len(ztr), min(1200, len(ztr)), replace=False)
            model = AgglomerativeClustering(n_clusters=2, linkage="ward")
            train_labels = model.fit_predict(ztr[itrain])
            counts = np.bincount(train_labels, minlength=2)
            train_centres = np.vstack([ztr[itrain][train_labels == c].mean(axis=0) for c in range(2)])
            test_labels = np.argmin(((zte[:, None, :] - train_centres[None, :, :]) ** 2).sum(axis=2), axis=1)
            grouped_rows.append(dict(week=week, split_seed=split_seed, train_flights=len(train_flights),
                                     evaluation_flights=len(test_flights), train_edges=len(train_d),
                                     evaluation_edges=len(test_d), discarded_cross_partition_edges=int(dropped_mask.sum()),
                                     retained_train_share=float(tr_mask.mean()), retained_evaluation_share=float(te_mask.mean()),
                                     train_ward_n=len(itrain), train_silhouette=float(silhouette_score(ztr[itrain], train_labels)),
                                     evaluation_nearest_centroid_silhouette=float(silhouette_score(zte, test_labels)) if len(np.unique(test_labels)) > 1 else np.nan,
                                     evaluation_minority_share=float(np.bincount(test_labels, minlength=2).min()/len(test_labels)),
                                     flight_disjoint=True, test_assignment="nearest training Ward-cluster centroid; descriptive, Ward has no native predict"))
            grouped_profiles.extend(summarize_episode_groups(test_d, test_labels, week, "flight_disjoint_ward_centroid_k2", f"seed_{split_seed}"))

        # One deterministic grouped holdout, fit scaling on train only, GMM K=1..12.
        split_seed = SEED
        tr_mask, te_mask, dropped_mask, train_flights, test_flights = flight_edge_split(d, .30, split_seed)
        xtr, xte = x.loc[tr_mask].reset_index(drop=True), x.loc[te_mask].reset_index(drop=True)
        ztr, zte = fit_scale(xtr, xte, "robust")
        eval_ix = np.random.default_rng(SEED+17).choice(len(zte), min(650, len(zte)), replace=False)
        for k in range(1, 13):
            gm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=.001,
                                 random_state=SEED, n_init=2, max_iter=200).fit(ztr)
            train_labels = gm.predict(ztr)
            if k > 1:
                ix = np.random.default_rng(SEED+31).choice(len(ztr), min(1000, len(ztr)), replace=False)
                sub_labels = gm.predict(ztr[ix])
                sil = float(silhouette_score(ztr[ix], sub_labels)) if len(np.unique(sub_labels)) > 1 else np.nan
            else:
                sil = np.nan
            gmm_rows.append(dict(week=week, split_seed=split_seed, k=k,
                                 train_n=len(ztr), evaluation_n=len(zte),
                                 train_bic=float(gm.bic(ztr)), evaluation_mean_log_likelihood=float(gm.score(zte[eval_ix])),
                                 train_silhouette_sample=sil,
                                 min_train_component_share=float(np.bincount(train_labels, minlength=k).min()/len(train_labels)),
                                 heldout_flights=len(test_flights), discarded_cross_partition_edges=int(dropped_mask.sum()),
                                 covariance_type="diag", feature_set="Stage12E full physical features", scaling="train-fit robust q10-q90"))

    # Censoring/selection comparison uses the complete frozen episode summary, not only structure rows.
    censor_tables, censor_effects, phase_tables = [], [], []
    for week in WEEKS:
        a, b, c = censor_audit(week, full[week])
        censor_tables.append(a); censor_effects.append(b); phase_tables.append(c)

    # HDBSCAN bounded sensitivity on a fixed 2,500-row cohort slice per week.
    hdb_rows = []
    for week in WEEKS:
        d = datasets[week]
        x = raw_matrix(d, "full")
        z = np.clip(pilot_scaler.transform(x), -8, 8)
        idx = np.random.default_rng(SEED).choice(len(z), min(2500, len(z)), replace=False)
        for min_samples in (5, 20):
            for min_cluster_size in (30, 120, 250):
                labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(z[idx])
                inlier = labels >= 0
                n_clusters = len(np.unique(labels[inlier]))
                sil = silhouette_score(z[idx][inlier], labels[inlier]) if n_clusters > 1 and inlier.sum() > n_clusters else np.nan
                hdb_rows.append(dict(week=week, n=len(idx), min_samples=min_samples,
                                     min_cluster_size=min_cluster_size, clusters=n_clusters,
                                     coverage=float(inlier.mean()), silhouette=float(sil) if np.isfinite(sil) else np.nan))

    # Tables
    pd.DataFrame(reproduced).to_csv(out_tables / "ward_k2_reproduction.csv", index=False)
    pd.DataFrame(profiles).to_csv(out_tables / "ward_k2_profiles_original_units.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(out_tables / "ward_k2_phase_composition.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out_tables / "ward_feature_scaling_sensitivity.csv", index=False)
    pd.DataFrame(correlations).to_csv(out_tables / "duration_sampling_correlations.csv", index=False)
    pd.DataFrame(grouped_rows).to_csv(out_tables / "flight_disjoint_ward_validation.csv", index=False)
    pd.DataFrame(grouped_profiles).to_csv(out_tables / "flight_disjoint_ward_test_profiles.csv", index=False)
    pd.DataFrame(gmm_rows).to_csv(out_tables / "flight_disjoint_gmm_k1_k12.csv", index=False)
    pd.DataFrame(hdb_rows).to_csv(out_tables / "hdbscan_sensitivity.csv", index=False)
    pd.concat(censor_tables, ignore_index=True).to_csv(out_tables / "censoring_profiles.csv", index=False)
    pd.concat(censor_effects, ignore_index=True).to_csv(out_tables / "censoring_effect_sizes.csv", index=False)
    pd.concat(phase_tables, ignore_index=True).to_csv(out_tables / "censoring_phase_composition.csv", index=False)
    pd.DataFrame(matched_summaries).to_csv(out_tables / "duration_sampling_matching_summary.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(out_tables / "duration_sampling_matched_pairs.csv", index=False)

    # Compact, auditable flight dependence summary on the Stage12E sample.
    dep_rows = []
    for week in WEEKS:
        d = datasets[week]
        counts = pd.concat([d.a_flight_key, d.b_flight_key], ignore_index=True).value_counts()
        any_rep = (d.a_flight_key.map(counts).gt(1) | d.b_flight_key.map(counts).gt(1))
        dep_rows.append(dict(week=week, episodes=len(d), unique_flights=int(counts.size),
                             unique_pairs=int(d.pair_id.nunique()),
                             flights_appearing_in_multiple_episodes=int(counts.gt(1).sum()),
                             endpoint_observation_share_from_repeated_flights=float(counts[counts.gt(1)].sum()/counts.sum()),
                             episodes_with_any_repeated_flight_share=float(any_rep.mean()),
                             episodes_with_both_flights_repeated_share=float((d.a_flight_key.map(counts).gt(1) & d.b_flight_key.map(counts).gt(1)).mean())))
    pd.DataFrame(dep_rows).to_csv(out_tables / "flight_dependence.csv", index=False)

    # Diagnostic figure: reproduction, scaling/feature sensitivity and censor selection.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#f5f8fb")
    repro = pd.DataFrame(reproduced)
    ax = axes[0]
    ax.bar(["Pilot", "Replica"], repro.silhouette, color=["#155e75", "#a21caf"])
    ax.set_ylim(0, .75); ax.set_title("Ward K=2: reproducción")
    for i, row in repro.iterrows():
        ax.text(i, row.silhouette+.02, f"{row.silhouette:.3f}\nminoría {row.minority_share:.1%}", ha="center", fontsize=9)
    ax = axes[1]
    sens = pd.DataFrame(sensitivity)
    for w, label, color in [(WEEKS[0], "Pilot", "#155e75"), (WEEKS[1], "Replica", "#a21caf")]:
        q = sens[(sens.week == w) & (sens.scaling == "robust")]
        ax.plot(["Full", "Geometría", "Geometría + tasas"], q.silhouette, "o-", label=label, color=color)
    ax.set_title("Sensibilidad de Ward por variables")
    ax.tick_params(axis="x", rotation=10); ax.legend()
    ax = axes[2]
    eff = pd.concat(censor_effects, ignore_index=True)
    q = eff[eff.metric.isin(["observed_duration_s", "n_samples", "track_change_deg", "altitude_change_ft"])]
    piv = q.pivot(index="metric", columns="week", values="standardized_mean_difference")
    piv.plot(kind="barh", ax=ax, color=["#155e75", "#a21caf"])
    ax.axvline(0, color="#526b7b", lw=.8); ax.set_title("Completo menos censurado\n(diferencia estandarizada)")
    ax.set_xlabel("SMD"); ax.legend(["Pilot", "Replica"], fontsize=8)
    fig.suptitle("STAGE 12F | Robustez, dependencia y selección", color="#153d59", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(out_figs / "stage12f_robustness_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    input_hashes = {}
    for week in WEEKS:
        for source in [OUT / "tables" / f"{week}_structure_sample.parquet",
                       OUT / "episodes" / f"{week}_episode_summary.parquet"]:
            digest = hashlib.sha256()
            with source.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            input_hashes[str(source.relative_to(OUT))] = digest.hexdigest()
    runtime = dict(started_utc=started_utc,
                   elapsed_seconds=round(time.perf_counter()-started, 3),
                   python_tracemalloc_peak_bytes=int(peak),
                   process_peak_rss_bytes=int(max_rss if os.uname().sysname == "Darwin" else max_rss * 1024),
                   source_scope="read-only Stage 12E episode summaries and structure samples",
                   source_sha256=input_hashes,
                   stages_executed=["Ward reproduction", "feature/scaling sensitivity", "flight-disjoint Ward", "flight-disjoint GMM K=1..12", "duration and sampling normalization", "duration/sample/phase/age matching", "full-cohort censoring comparison", "HDBSCAN sensitivity", "summary figure"],
                   stages_not_executed=["Stage 01-12E reconstruction or candidate generation", "ATC/color-based physical clustering", "detector validation", "risk classification or severity scoring"])
    (out_diag / "runtime_and_scope.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(runtime, indent=2, ensure_ascii=False))
    print(pd.DataFrame(reproduced).to_string(index=False))
    print(pd.DataFrame(matched_summaries).to_string(index=False))


if __name__ == "__main__":
    main()
