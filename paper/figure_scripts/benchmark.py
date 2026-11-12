#!/usr/bin/env python3
"""Build the frozen episode cohort and score M0–M5 once on held-out weeks.

Run from the repository root. All outputs are new files under results/benchmarks.
No source or Stage 1–21 result is modified. Protocol: docs/decisiones_protocolo.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
S20 = ROOT / "results/stage20"
STAGE03 = ROOT / "data/processed/scat/03_clearances/scat_clearances.parquet"
AUDIT = ROOT / "data/processed/scat/19_calib_selection/tables/audit.json"
COLORS = ("verde", "amarillo", "naranja", "rojo")
NS = 1_000_000_000
SEED = 20261112


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def entity(week, flight, member):
    return week.astype(str) + "\x1f" + flight.astype(str) + "\x1f" + member.astype(str)


def events(broad=False):
    wanted = None
    if not broad:
        attribution = pd.read_csv(S20 / "stage20_clearance_attribution.csv.gz",
                                  usecols=["clearance_id"])
        wanted = set(attribution.clearance_id.astype(str))
        check(len(wanted) == 51950, "Expected 51,950 eligible orders")
    columns = ["clearance_id", "archive_week", "flight_key", "source_member", "clearance_epoch"]
    if broad:
        columns.append("clearance_type")
    table = pq.read_table(STAGE03, columns=columns)
    df = table.to_pandas()
    if broad:
        df = df.loc[df.clearance_type.astype(str).str.contains("HEADING|LEVEL")].copy()
        check(len(df) == 908165, "Expected 908,165 heading/level records")
    else:
        df = df[df.clearance_id.astype(str).isin(wanted)].copy()
        check(len(df) == len(wanted) and df.clearance_id.is_unique,
              "Stage 03 eligible IDs do not match attribution")
    df["entity"] = entity(df.archive_week, df.flight_key, df.source_member)
    df["time_ns"] = np.rint(df.clearance_epoch.to_numpy(float) * NS).astype(np.int64)
    return {key: np.sort(group.time_ns.to_numpy(np.int64))
            for key, group in df.groupby("entity", sort=False)}


def any_order(keys_a, keys_b, times_ns, index, before, after):
    """Strict (T,T+after] or [T-before,T); one index per aircraft identity."""
    out = np.zeros(len(times_ns), dtype=bool)
    for keys in (keys_a, keys_b):
        frame = pd.DataFrame({"key": np.asarray(keys), "t": np.asarray(times_ns)})
        for key, rows in frame.groupby("key", sort=False).indices.items():
            if key not in index:
                continue
            t = index[key]
            q = frame.t.to_numpy(np.int64)[rows]
            if before:
                lo = np.searchsorted(t, q - before * NS, side="left")
                hi = np.searchsorted(t, q, side="left")
            else:
                lo = np.searchsorted(t, q, side="right")
                hi = np.searchsorted(t, q + after * NS, side="right")
            out[np.asarray(rows)] |= hi > lo
    return out


def green_episodes(week, gap, green_limit, non_green, index):
    path = S20 / "pairs" / f"pairs_{week}.parquet"
    cols = ["T", "flight_key_a", "flight_key_b", "source_member_a",
            "source_member_b", "m_min", "tau_sec", "color", "split"]
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=cols):
        frame = batch.to_pandas()
        frame = frame.loc[frame.color.eq("verde") & frame.m_min.lt(green_limit)].copy()
        if frame.empty:
            continue
        a = entity(pd.Series(week, index=frame.index), frame.flight_key_a,
                   frame.source_member_a)
        b = entity(pd.Series(week, index=frame.index), frame.flight_key_b,
                   frame.source_member_b)
        aa, bb = a.to_numpy(str), b.to_numpy(str)
        frame["entity1"] = np.where(aa < bb, aa, bb)
        frame["entity2"] = np.where(aa < bb, bb, aa)
        frame["time_ns"] = np.rint(frame["T"].to_numpy(float) * NS).astype(np.int64)
        parts.append(frame[["entity1", "entity2", "time_ns", "m_min", "tau_sec", "split"]])
    if not parts:
        return pd.DataFrame()
    green = pd.concat(parts, ignore_index=True)
    green.sort_values(["entity1", "entity2", "time_ns"], kind="mergesort", inplace=True)
    green.reset_index(drop=True, inplace=True)
    # Retain only green observations that cannot be attached to an existing
    # non-green run within the frozen gap. This preserves audited non-green IDs.
    ng = non_green.loc[non_green.week.eq(week), ["entity1", "entity2", "time_ns"]]
    if not ng.empty:
        ng = ng.sort_values("time_ns")
        green = green.sort_values("time_ns").reset_index(drop=True)
        group_cols = ["entity1", "entity2"]
        prior = pd.merge_asof(green, ng.rename(columns={"time_ns": "ng_time"}),
                              left_on="time_ns", right_on="ng_time", by=group_cols,
                              direction="backward")
        future = pd.merge_asof(green, ng.rename(columns={"time_ns": "ng_time"}),
                               left_on="time_ns", right_on="ng_time", by=group_cols,
                               direction="forward")
        near = ((green.time_ns - prior.ng_time).le(gap * NS).fillna(False) |
                (future.ng_time - green.time_ns).le(gap * NS).fillna(False))
        green = green.loc[~near.to_numpy()].copy()
    if green.empty:
        return pd.DataFrame()
    green.sort_values(["entity1", "entity2", "time_ns"], kind="mergesort", inplace=True)
    cut = (green.entity1.ne(green.entity1.shift()) |
           green.entity2.ne(green.entity2.shift()) |
           green.time_ns.diff().gt(gap * NS))
    green["run"] = cut.cumsum()
    green["y_exposure"] = any_order(green.entity1.to_numpy(str), green.entity2.to_numpy(str),
                                    green.time_ns.to_numpy(np.int64), index, 0, 300)
    grouped = green.groupby("run", sort=False)
    ep = grouped.agg(entity1=("entity1", "first"), entity2=("entity2", "first"),
                     split=("split", "first"), start_ns=("time_ns", "min"),
                     end_ns=("time_ns", "max"), observations=("time_ns", "size"),
                     m_min=("m_min", "min"), tau_sec=("tau_sec", "min"),
                     y_exposure=("y_exposure", "max")).reset_index(drop=True)
    ep["week"] = week
    ep["highest_color"] = "verde"
    return ep


def regroup_non_green(obs, gap):
    """Rebuild non-green runs for a prespecified alternative gap."""
    obs = obs.sort_values(["week", "entity1", "entity2", "time_ns"],
                          kind="mergesort").reset_index(drop=True).copy()
    cut = (obs.week.ne(obs.week.shift()) | obs.entity1.ne(obs.entity1.shift()) |
           obs.entity2.ne(obs.entity2.shift()) | obs.time_ns.diff().gt(gap * NS))
    obs["new_episode_id"] = cut.cumsum().astype(np.int64)
    obs["color_rank"] = obs.color.map({c: i for i, c in enumerate(COLORS)})
    g = obs.groupby("new_episode_id", sort=False)
    ep = g.agg(week=("week", "first"), entity1=("entity1", "first"),
               entity2=("entity2", "first"), start_ns=("time_ns", "min"),
               end_ns=("time_ns", "max"), observations=("time_ns", "size"),
               m_min=("mmin", "min"), tau_sec=("tau_min", lambda x: 60 * x.min()),
               color_rank=("color_rank", "max"),
               y_exposure=("with_next", "max")).reset_index(names="episode_id")
    ep["highest_color"] = ep.color_rank.map(dict(enumerate(COLORS)))
    return ep


def build_cohort(out, gap=360, green_limit=2.0):
    audit = json.loads(AUDIT.read_text())
    non_green = pd.read_csv(S20 / "stage20_non_green_observations.csv.gz",
                            usecols=["episode_id", "week", "entity1", "entity2", "time_ns",
                                     "mmin", "tau_min", "color", "with_next"])
    if gap == 360:
        old = pd.read_csv(S20 / "stage20_episodes.csv")
        counts = old.highest_color.value_counts()
        check([int(counts.get(c, 0)) for c in COLORS[1:]] == [39351, 9039, 6051],
              "Frozen non-green episode counts changed")
        check(int((old.split.eq("test") & old.highest_color.eq("rojo")).sum()) == 1863,
              "Frozen red test episode count changed")
        ng_features = non_green.groupby("episode_id", sort=False).agg(
            m_min=("mmin", "min"), tau_sec=("tau_min", lambda x: 60 * x.min()))
        old = old.join(ng_features, on="episode_id", validate="one_to_one")
        old["start_ns"] = pd.to_datetime(old.start_utc, utc=True).dt.as_unit("ns").astype("int64")
        old["end_ns"] = pd.to_datetime(old.end_utc, utc=True).dt.as_unit("ns").astype("int64")
        old["y_exposure"] = old.any_next_clearance
    else:
        old = regroup_non_green(non_green, gap)
        old["split"] = np.where(old.week.isin(audit["test"]), "test", "calibration")
    index = events()
    greens = []
    for week in audit["weeks"]:
        print(f"Building green episodes: {week}", flush=True)
        g = green_episodes(week, gap, green_limit, non_green, index)
        if not g.empty:
            greens.append(g)
    check(greens, "No green episodes constructed")
    green = pd.concat(greens, ignore_index=True)
    green["episode_id"] = np.arange(len(green), dtype=np.int64) + old.episode_id.max() + 1
    keep = ["episode_id", "week", "split", "entity1", "entity2", "start_ns",
            "end_ns", "observations", "m_min", "tau_sec", "highest_color", "y_exposure"]
    cohort = pd.concat([old[keep], green[keep]], ignore_index=True)
    check(cohort.episode_id.is_unique, "Duplicate episode ID")
    t = cohort.start_ns.to_numpy(np.int64)
    a, b = cohort.entity1.to_numpy(str), cohort.entity2.to_numpy(str)
    cohort["y_fixed"] = any_order(a, b, t, index, 0, 300)
    cohort["prior_600"] = any_order(a, b, t, index, 600, 0)
    cohort["duration_s"] = (cohort.end_ns - cohort.start_ns) / NS
    check(cohort.duration_s.max() > 60 and cohort.duration_s.max() < 7 * 86400,
          "Timestamp unit mismatch or implausible episode duration")
    check(cohort.loc[cohort.highest_color.eq("verde"), "start_ns"].min() > 10**18,
          "Green timestamps are not nanoseconds since epoch")
    cohort["day"] = pd.to_datetime(cohort.start_ns, utc=True).dt.strftime("%Y-%m-%d")
    # Causal rolling-hour density: count episode starts known at each T0.
    # The stratum edges are fitted only on calibration weeks.
    cohort["hour"] = pd.to_datetime(cohort.start_ns, utc=True).dt.floor("h")
    cohort["prior_hour_episodes"] = 0
    for _, rows in cohort.groupby("week", sort=False).indices.items():
        tt = cohort.start_ns.to_numpy(np.int64)[rows]
        ordered = np.sort(tt)
        count = (np.searchsorted(ordered, tt, side="right") -
                 np.searchsorted(ordered, tt - 3600 * NS, side="left"))
        cohort.loc[np.asarray(rows), "prior_hour_episodes"] = count
    edges = cohort.loc[cohort.split.eq("calibration"), "prior_hour_episodes"].quantile([1/3, 2/3]).to_numpy()
    cohort["traffic_stratum"] = np.searchsorted(edges, cohort.prior_hour_episodes.to_numpy(), side="right")
    check(cohort.split.eq("test").sum() > 0 and cohort.split.eq("calibration").sum() > 0,
          "Missing calibration or test split")
    check(set(cohort.loc[cohort.split.eq("test"), "week"]) == set(audit["test"]),
          "Test weeks differ from frozen manifest")
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out / "episodes.parquet", index=False)
    (out / "cohort_manifest.json").write_text(json.dumps({"gap_sec": gap,
        "green_m_limit": "unbounded" if np.isinf(green_limit) else green_limit,
        "traffic_quantile_edges": edges.tolist(),
        "counts": {c: int(cohort.highest_color.eq(c).sum()) for c in COLORS},
        "test_weeks": audit["test"]}, indent=2) + "\n")
    return cohort


def _features(frame, model, spline=None, scaler=None, fit=False):
    m = frame.m_min.to_numpy(float).reshape(-1, 1)
    tau = frame.tau_sec.to_numpy(float).reshape(-1, 1) / 60
    if model == "M2":
        raw = m
    else:
        if fit:
            spline = SplineTransformer(n_knots=4, degree=3, include_bias=False,
                                       extrapolation="constant").fit(np.column_stack([m, tau]))
        raw = spline.transform(np.column_stack([m, tau]))
        if model == "M4":
            traffic = np.eye(3)[frame.traffic_stratum.to_numpy(int)]
            raw = np.column_stack([raw, traffic[:, 1:], frame.prior_600.to_numpy(int),
                                   np.log1p(frame.duration_s.to_numpy(float))])
    if fit:
        scaler = StandardScaler().fit(raw)
    return scaler.transform(raw), spline, scaler


def _fit_predict(train, test, model, weight, c=1.0):
    y = train.y_fixed.to_numpy(int)
    w = train.observations.to_numpy(float) if weight == "pair" else None
    if model == "M0":
        rates = train.groupby("highest_color").apply(
            lambda g: np.average(g.y_fixed, weights=g.observations if weight == "pair" else None),
            include_groups=False).to_dict()
        return test.highest_color.map(rates).to_numpy(float)
    if model == "M1":
        key = train.m_min.lt(1)
        rates = {v: np.average(y[key.to_numpy() == v],
                               weights=None if w is None else w[key.to_numpy() == v])
                 for v in (False, True)}
        return test.m_min.lt(1).map(rates).to_numpy(float)
    if model.startswith("M5"):
        k = int(model[-1])
        # Increasing geometric danger = decreasing m. Quantile boundaries are
        # frozen on calibration and adjacent probabilities are pooled by PAV.
        edges = np.unique(np.quantile(train.m_min, np.linspace(0, 1, k + 1)[1:-1]))
        x = np.searchsorted(edges, train.m_min, side="right")
        xt = np.searchsorted(edges, test.m_min, side="right")
        grouped = pd.DataFrame({"bin": x, "y": y, "w": 1 if w is None else w})
        totals = grouped.groupby("bin").apply(
            lambda g: pd.Series({"rate": np.average(g.y, weights=g.w), "n": g.w.sum()}),
            include_groups=False)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            totals.index.to_numpy(float), totals.rate.to_numpy(float),
            sample_weight=totals.n.to_numpy(float))
        return iso.predict(xt.astype(float))
    x, spline, scaler = _features(train, model, fit=True)
    xt, _, _ = _features(test, model, spline=spline, scaler=scaler)
    lr = LogisticRegression(C=c, max_iter=500, solver="lbfgs")
    lr.fit(x, y, sample_weight=w)
    return lr.predict_proba(xt)[:, 1]


def _choose_c(train, model, weight):
    weeks = sorted(train.week.unique())
    scores = {}
    for c in (0.1, 1.0, 10.0):
        fold = []
        for week in weeks:
            tr = train.loc[train.week.ne(week)]
            va = train.loc[train.week.eq(week)]
            p = _fit_predict(tr, va, model, weight, c)
            sw = va.observations if weight == "pair" else None
            fold.append(brier_score_loss(va.y_fixed, p, sample_weight=sw))
        scores[c] = float(np.mean(fold))
    best = min(scores, key=scores.get)
    return best, scores[best]


def _calibration(y, p, w):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if len(np.unique(y)) < 2 or np.std(z) < 1e-10:
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6, max_iter=500).fit(z.reshape(-1, 1), y, sample_weight=w)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def score(cohort, out):
    check(cohort.y_fixed.notna().all(), "Missing fixed outcome")
    train = cohort.loc[cohort.split.eq("calibration")].copy()
    test = cohort.loc[cohort.split.eq("test")].copy()
    check(len(train.week.unique()) == 9 and len(test.week.unique()) == 4,
          "Expected 9 calibration and 4 test weeks")
    rng = np.random.default_rng(SEED)
    metrics, predictions, reliability, weekly, rates = [], [], [], [], []
    models = ["M0", "M1", "M2", "M3", "M4", "M5_3", "M5_4", "M5_5"]
    for weight in ("episode", "pair"):
        test_w = test.observations.to_numpy(float) if weight == "pair" else np.ones(len(test))
        y = test.y_fixed.to_numpy(int)
        pred = {}
        for model in models:
            c, cv = (_choose_c(train, model, weight) if model in ("M2", "M3", "M4")
                     else (1.0, np.nan))
            p = np.clip(_fit_predict(train, test, model, weight, c), 1e-6, 1 - 1e-6)
            check(np.isfinite(p).all(), f"Invalid probabilities for {model}")
            pred[model] = p
            intercept, slope = _calibration(y, p, test_w)
            row = {"weight": weight, "model": model, "C_calibration": c,
                   "calibration_lowo_brier": cv,
                   "n_test_episodes": len(test), "brier": brier_score_loss(y, p, sample_weight=test_w),
                   "log_loss": log_loss(y, p, sample_weight=test_w),
                   "calibration_intercept": intercept, "calibration_slope": slope,
                   "auc": roc_auc_score(y, p, sample_weight=test_w)}
            metrics.append(row)
            predictions.append(pd.DataFrame({"episode_id": test.episode_id, "weight": weight,
                                              "model": model, "y": y, "p": p,
                                              "week": test.week, "day": test.day,
                                              "observations": test.observations}))
            bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
            rel = pd.DataFrame({"bin": bins, "y": y, "p": p, "w": test_w}).groupby("bin", observed=True)
            for i, (_, g) in enumerate(rel):
                reliability.append({"weight": weight, "model": model, "decile": i + 1,
                                    "n": len(g), "mean_predicted": np.average(g.p, weights=g.w),
                                    "observed_rate": np.average(g.y, weights=g.w)})
            for week, g in test.groupby("week"):
                ix = test.week.eq(week).to_numpy()
                weekly.append({"weight": weight, "model": model, "week": week,
                               "brier": np.average((y[ix] - p[ix]) ** 2, weights=test_w[ix])})
        days = np.array(sorted(test.day.unique()))
        day_index = pd.Categorical(test.day, categories=days).codes
        day_w = np.bincount(day_index, weights=test_w, minlength=len(days))
        draws = rng.integers(0, len(days), size=(400, len(days)))
        base = pred["M0"]
        for row in metrics:
            if row["weight"] != weight:
                continue
            p = pred[row["model"]]
            loss = (y - p) ** 2 - (y - base) ** 2
            day_loss = np.bincount(day_index, weights=test_w * loss, minlength=len(days))
            boot = day_loss[draws].sum(axis=1) / day_w[draws].sum(axis=1)
            row["delta_brier_vs_M0"] = float(day_loss.sum() / day_w.sum())
            row["delta_brier_ci_low"], row["delta_brier_ci_high"] = np.quantile(boot, [.025, .975])
            row["delta_brier_se"] = float(np.std(boot, ddof=1))
        for color in COLORS:
            for label, df in (("calibration", train), ("test", test)):
                g = df.loc[df.highest_color.eq(color)]
                if not g.empty:
                    rates.append({"weight": weight, "split": label, "color": color,
                                  "n": len(g), "y_fixed_rate": np.average(g.y_fixed, weights=g.observations if weight == "pair" else None),
                                  "y_exposure_rate": np.average(g.y_exposure, weights=g.observations if weight == "pair" else None),
                                  "duration_y_corr": g.duration_s.corr(g.y_fixed.astype(float))})
    met = pd.DataFrame(metrics)
    selection = {}
    for weight in ("episode", "pair"):
        continuous = met.loc[met.weight.eq(weight) & met.model.isin(["M2", "M3", "M4"])]
        chosen = continuous.loc[continuous.calibration_lowo_brier.idxmin()]
        selection[weight] = {"selected_continuous_model": chosen.model,
                             "selection_rule": "minimum mean leave-one-week-out calibration Brier",
                             "calibration_lowo_brier": float(chosen.calibration_lowo_brier),
                             "test_delta_brier_vs_M0": float(chosen.delta_brier_vs_M0),
                             "test_delta_brier_se": float(chosen.delta_brier_se),
                             "continuous_preferred": bool(-chosen.delta_brier_vs_M0 > chosen.delta_brier_se)}
    (out / "decision.json").write_text(json.dumps(selection, indent=2) + "\n")
    met.to_csv(out / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(reliability).to_csv(out / "reliability.csv", index=False)
    pd.DataFrame(weekly).to_csv(out / "weekly_brier.csv", index=False)
    pd.DataFrame(rates).to_csv(out / "outcome_windows.csv", index=False)
    wdf = pd.DataFrame(weekly)
    ref = wdf.loc[wdf.model.eq("M0"), ["weight", "week", "brier"]].rename(columns={"brier": "M0_brier"})
    wdf = wdf.merge(ref, on=["weight", "week"], validate="many_to_one")
    wdf["paired_delta_vs_M0"] = wdf.brier - wdf.M0_brier
    wdf.to_csv(out / "weekly_paired_brier.csv", index=False)
    burden = test.loc[test.highest_color.ne("verde")].groupby(["week", "hour"]).size().rename("non_green_episodes").reset_index()
    burden.to_csv(out / "alert_burden_per_hour.csv", index=False)
    # The protocol's comparison picks the best continuous model on calibration
    # only; test Brier is then used once to apply the frozen one-SE rule.
    # LOWO complexity choices above are retained in the metrics file.
    print(met[["weight", "model", "brier", "delta_brier_vs_M0", "delta_brier_se"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results/benchmarks")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--use-prepared", action="store_true")
    p.add_argument("--green-limit", type=float, default=2.0,
                   help="Green m_min inclusion limit; use inf for no limit")
    p.add_argument("--gap", type=int, default=360, choices=(240, 360, 480))
    args = p.parse_args()
    check((ROOT / "docs/decisiones_protocolo.md").exists(), "Missing approved protocol")
    cohort = (pd.read_parquet(args.out / "episodes.parquet") if args.use_prepared else
              build_cohort(args.out, gap=args.gap, green_limit=args.green_limit))
    if args.prepare_only:
        return
    score(cohort, args.out)


if __name__ == "__main__":
    main()
