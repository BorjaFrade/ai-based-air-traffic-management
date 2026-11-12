#!/usr/bin/env python3
"""Episode-weighted M3/M4 term ablation on the frozen benchmark cohort.

Run: python3 paper/figure_scripts/ablation.py
Only calibration weeks choose regularisation. Held-out predictions are scored
once per fitted variant; no feature set is selected using test outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import SplineTransformer, StandardScaler

import benchmark as base

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "results/benchmarks/episodes.parquet"
PUBLISHED = ROOT / "results/benchmarks/metrics.csv"
DEFAULT_OUT = ROOT / "results/benchmarks/ablation"
VARIANTS = {
    "M3": (),
    "M3+density": ("density",),
    "M3+prior": ("prior",),
    "M3+duration": ("duration",),
    "M3+density+prior": ("density", "prior"),
    "M4": ("density", "prior", "duration"),
}
CS = (0.1, 1.0, 10.0)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def design(train: pd.DataFrame, target: pd.DataFrame, terms: tuple[str, ...]):
    """Match benchmark._features: fit spline and scaler on this training fold."""
    tr_geo = np.column_stack([train.m_min.to_numpy(float), train.tau_sec.to_numpy(float) / 60])
    te_geo = np.column_stack([target.m_min.to_numpy(float), target.tau_sec.to_numpy(float) / 60])
    spline = SplineTransformer(n_knots=4, degree=3, include_bias=False,
                               extrapolation="constant").fit(tr_geo)
    tr = spline.transform(tr_geo)
    te = spline.transform(te_geo)
    for term in terms:
        if term == "density":
            tr = np.column_stack([tr, np.eye(3)[train.traffic_stratum.to_numpy(int)][:, 1:]])
            te = np.column_stack([te, np.eye(3)[target.traffic_stratum.to_numpy(int)][:, 1:]])
        elif term == "prior":
            tr = np.column_stack([tr, train.prior_600.to_numpy(int)])
            te = np.column_stack([te, target.prior_600.to_numpy(int)])
        elif term == "duration":
            tr = np.column_stack([tr, np.log1p(train.duration_s.to_numpy(float))])
            te = np.column_stack([te, np.log1p(target.duration_s.to_numpy(float))])
        else:
            raise ValueError(term)
    scaler = StandardScaler().fit(tr)
    return scaler.transform(tr), scaler.transform(te)


def predict(train: pd.DataFrame, target: pd.DataFrame, terms: tuple[str, ...], c: float):
    x, z = design(train, target, terms)
    lr = LogisticRegression(C=c, max_iter=500, solver="lbfgs")
    lr.fit(x, train.y_fixed.to_numpy(int))
    return np.clip(lr.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)


def choose_c(train: pd.DataFrame, terms: tuple[str, ...]):
    folds = []
    weeks = sorted(train.week.unique())
    check(len(weeks) == 9, "Expected nine calibration weeks")
    for c in CS:
        score = []
        for week in weeks:
            tr = train.loc[train.week.ne(week)]
            va = train.loc[train.week.eq(week)]
            p = predict(tr, va, terms, c)
            score.append(brier_score_loss(va.y_fixed.to_numpy(int), p))
        folds.append((float(np.mean(score)), c))
    return min(folds)[1], min(folds)[0]


def paired_bootstrap(loss: np.ndarray, day_codes: np.ndarray, draws: np.ndarray, day_n: np.ndarray):
    day_loss = np.bincount(day_codes, weights=loss, minlength=len(day_n))
    boot = day_loss[draws].sum(axis=1) / day_n[draws].sum(axis=1)
    return (float(np.mean(loss)), float(np.quantile(boot, .025)),
            float(np.quantile(boot, .975)), float(np.std(boot, ddof=1)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = cli.parse_args()
    cohort = pd.read_parquet(COHORT)
    check(len(cohort) == 67367 and cohort.episode_id.is_unique, "Cohort differs from 67,367 episodes")
    train = cohort.loc[cohort.split.eq("calibration")].copy()
    test = cohort.loc[cohort.split.eq("test")].copy()
    check(len(test) == 21281 and len(test.week.unique()) == 4, "Test cohort differs from 21,281 episodes")
    check(len(train.week.unique()) == 9, "Calibration week count differs")
    y = test.y_fixed.to_numpy(int)
    check(set(np.unique(y)) == {0, 1}, "Outcome differs from fixed binary target")
    published = pd.read_csv(PUBLISHED)
    published = published.loc[published.weight.eq("episode")].set_index("model")
    p0 = np.clip(base._fit_predict(train, test, "M0", "episode"), 1e-6, 1 - 1e-6)
    check(abs(brier_score_loss(y, p0) - published.loc["M0", "brier"]) < 1e-12,
          "M0 fails exact published-score reproduction")
    pred = {"M0": p0}
    chosen = {}
    for name, terms in VARIANTS.items():
        c, cv = choose_c(train, terms)
        chosen[name] = (c, cv)
        pred[name] = predict(train, test, terms, c)
    # Assert both published baselines to numerical precision, not merely to
    # their four-decimal manuscript display.
    for name in ("M3", "M4"):
        score = brier_score_loss(y, pred[name])
        check(abs(score - published.loc[name, "brier"]) < 1e-12,
              f"{name} Brier fails exact published-score reproduction")
        check(abs((score - brier_score_loss(y, p0)) -
                  published.loc[name, "delta_brier_vs_M0"]) < 1e-12,
              f"{name} ΔBrier fails exact published-score reproduction")
        check(round(score, 4) == {"M3": 0.1696, "M4": 0.1690}[name],
              f"{name} fails manuscript rounding")
    saved = pd.read_csv(ROOT / "results/benchmarks/predictions.csv.gz",
                        usecols=["episode_id", "weight", "model", "p"])
    saved = saved.loc[saved.weight.eq("episode") & saved.model.isin(["M3", "M4"])]
    for name in ("M3", "M4"):
        prior = saved.loc[saved.model.eq(name)].set_index("episode_id").loc[test.episode_id, "p"]
        check(np.allclose(pred[name], prior.to_numpy(float), rtol=0, atol=1e-12),
              f"{name} predictions differ from the published benchmark CSV")
    days = np.array(sorted(test.day.unique()))
    day_codes = pd.Categorical(test.day, categories=days).codes
    day_n = np.bincount(day_codes, minlength=len(days))
    draws = np.random.default_rng(base.SEED).integers(0, len(days), size=(400, len(days)))
    rows, weekly = [], []
    for name in VARIANTS:
        p = pred[name]
        log_loss_value = log_loss(y, p)
        _, slope = base._calibration(y, p, np.ones(len(test)))
        vs0 = paired_bootstrap((y - p) ** 2 - (y - p0) ** 2, day_codes, draws, day_n)
        vs3 = paired_bootstrap((y - p) ** 2 - (y - pred["M3"]) ** 2, day_codes, draws, day_n)
        row = {"model": name, "terms": "+".join(VARIANTS[name]) or "geometry_only",
               "C_calibration": chosen[name][0], "calibration_lowo_brier": chosen[name][1],
               "n_calibration": len(train), "n_test": len(test),
               "brier": brier_score_loss(y, p), "log_loss": log_loss_value,
               "calibration_slope": slope}
        for prefix, values in (("delta_vs_M0", vs0), ("delta_vs_M3", vs3)):
            row[prefix] = values[0]
            row[prefix + "_ci_low"] = values[1]
            row[prefix + "_ci_high"] = values[2]
            row[prefix + "_se"] = values[3]
        rows.append(row)
        for week in sorted(test.week.unique()):
            ix = test.week.eq(week).to_numpy()
            weekly.append({"model": name, "week": week,
                           "delta_vs_M0": float(np.mean((y[ix] - p[ix])**2 - (y[ix] - p0[ix])**2)),
                           "delta_vs_M3": float(np.mean((y[ix] - p[ix])**2 - (y[ix] - pred["M3"][ix])**2))})
    metrics = pd.DataFrame(rows)
    for name in ("M3", "M4"):
        row = metrics.set_index("model").loc[name]
        for current, prior in (("delta_vs_M0_ci_low", "delta_brier_ci_low"),
                               ("delta_vs_M0_ci_high", "delta_brier_ci_high")):
            check(abs(row[current] - published.loc[name, prior]) < 1e-12,
                  f"{name} bootstrap interval differs from original benchmark")
    week_df = pd.DataFrame(weekly)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "metrics.csv", index=False, float_format="%.15g")
    week_df.to_csv(args.out / "weekly_differences.csv", index=False, float_format="%.15g")
    full = metrics.set_index("model")
    extra = full.loc["M4", "brier"] - full.loc["M3", "brier"]
    density_prior = full.loc["M3+density+prior", "brier"] - full.loc["M3", "brier"]
    duration_after = full.loc["M4", "brier"] - full.loc["M3+density+prior", "brier"]
    check(abs(extra - density_prior - duration_after) < 1e-12, "Ablation decomposition does not sum")
    lines = ["# Ablación de M4 (T11)", "",
             "Cohorte congelada: 67.367 episodios; 21.281 de test. Resultado: orden elegible en "
             "`(T0,T0+300 s]`. Un episodio, un voto. Nueve semanas de calibración "
             "para selección de C por leave-one-week-out; cuatro semanas de test para puntuación.", "",
             "M3 y M4 completos reproducen exactamente los Brier y ΔBrier guardados en "
             "`results/benchmarks/metrics.csv` (asserts en `ablation.py`). Los IC son "
             "percentiles de 400 réplicas bootstrap por días con semilla 20261112, "
             "y usan los mismos índices de días en todas las diferencias pareadas.", "",
             "| Modelo | Brier | Log score | Pendiente | Δ vs M0 [IC 95 %] | Δ vs M3 [IC 95 %] |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        fmt = lambda x: f"{x:+.6f}"
        lines.append(f"| {r['model']} | {r['brier']:.6f} | {r['log_loss']:.6f} | "
                     f"{r['calibration_slope']:.4f} | {fmt(r['delta_vs_M0'])} "
                     f"[{fmt(r['delta_vs_M0_ci_low'])}, {fmt(r['delta_vs_M0_ci_high'])}] | "
                     f"{fmt(r['delta_vs_M3'])} "
                     f"[{fmt(r['delta_vs_M3_ci_low'])}, {fmt(r['delta_vs_M3_ci_high'])}] |")
    lines += ["", f"Incremento M4 frente a M3: **{extra:+.6f}** de Brier.",
              f"Densidad + orden previa sin duración: **{density_prior:+.6f}**.",
              f"Duración añadida después de esas dos variables: **{duration_after:+.6f}**.",
              "Estas son diferencias condicionales entre modelos reentrenados, no una "
              "atribución causal ni una descomposición aditiva de efectos individuales.", "",
              "## Diferencias por semana de test", "",
              "| Modelo | Semana | Δ vs M0 | Δ vs M3 |", "| --- | --- | ---: | ---: |"]
    for r in weekly:
        lines.append(f"| {r['model']} | {r['week']} | {r['delta_vs_M0']:+.6f} | "
                     f"{r['delta_vs_M3']:+.6f} |")
    lines += ["", "## Límite de interpretación", "",
              "La duración y el máximo geométrico se conocen al terminar el episodio. "
              "Esta ablación compara asociaciones retrospectivas, no predicción "
              "disponible en tiempo real en T0.", ""]
    (args.out / "RESUMEN.md").write_text("\n".join(lines), encoding="utf-8")
    print(metrics[["model", "brier", "delta_vs_M0", "delta_vs_M3"]].to_string(index=False))


if __name__ == "__main__":
    main()
