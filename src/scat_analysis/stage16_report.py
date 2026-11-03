#!/usr/bin/env python3
"""Stage 16 report: does predicted pair geometry anticipate ATC orders? (density-standardised).

Reads the per-week shards written by ``stage16_pair_hazard.py`` and produces

* ``tables/*.csv``      lift tables (cell and cumulative), colour-state lifts, validation, cutpoints
* ``figures/*.png``
* ``reports/STAGE16_REPORT.pdf`` and ``reports/STAGE16_SUMMARY.md`` and ``reports/cutpoints.json``

Definitions
-----------
* A *pair snapshot* is a pair of aircraft (FL290-410, < 350 NM apart) at a snapshot time T.
* m_min = minimum, over the next 20 min, of the normalised separation max(H/5 NM, V/1000 ft)
  along the predicted trajectories (predictor R = follow route, C = constant velocity).
  tau = time at which that minimum is reached.
* Label W = an eligible LEVEL/HEADING order is issued to either aircraft within (T, T+W].
* Baseline = pairs whose m_min >= 12 (or never below 12) in the same density stratum.
* Lift = observed labelled pairs / labelled pairs expected if the pairs had the baseline order
  rate of their density stratum (Mantel-Haenszel style standardisation).
* Uncertainty: bootstrap over calendar days (cluster), baseline recomputed in each replicate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

M_EDGES = np.array([0, 1, 1.2, 1.5, 2, 3, 4, 6, 8, 12, np.inf])
M_LABELS = ["<1", "1-1.2", "1.2-1.5", "1.5-2", "2-3", "3-4", "4-6", "6-8", "8-12"]
TAU_LABELS = ["0-2 min", "2-5 min", "5-10 min", "10-20 min"]
TAU_MAX_S = [120, 300, 600, 1200]
M_UPPER = [1, 1.2, 1.5, 2, 3, 4, 6, 8, 12]
STATE_H = (60, 120, 300, 600, 1200)
STATE_NAMES = ["verde", "amarillo", "naranja", "rojo"]
LABEL_W = (120, 300, 600)
NM, NT, ND = 9, 4, 4  # used cells (m 0..8, tau 0..3), density strata
PRED = ("R", "C")
RNG_SEED = 1616


# ----------------------------------------------------------------------------- loading
def nominal_days_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop border days: snapshots whose UTC day falls outside the nominal 7-day SCAT week of the shard.
    (16 such days, 803 of 21.6 M pair-snapshots; flights crossing midnight.)"""
    t0 = pd.Timestamp("1970-01-01")
    d0 = (pd.to_datetime(df.week.str.slice(0, 8), format="%Y%m%d") - t0).dt.days
    d1 = (pd.to_datetime(df.week.str.slice(9, 17), format="%Y%m%d") - t0).dt.days
    return df[(df.day >= d0) & (df.day <= d1)].reset_index(drop=True)


def load_tables(tab_dir: Path, nominal_only: bool = True):
    cells = pd.concat([pd.read_parquet(f) for f in sorted(tab_dir.glob("cells_*.parquet"))], ignore_index=True)
    states = pd.concat([pd.read_parquet(f) for f in sorted(tab_dir.glob("colour_states_*.parquet"))], ignore_index=True)
    val = pd.concat([pd.read_parquet(f) for f in sorted(tab_dir.glob("validation_*.parquet"))], ignore_index=True)
    if nominal_only:
        cells, states = nominal_days_only(cells), nominal_days_only(states)
    for name, df in (("cells", cells), ("colour_states", states)):
        if "dens_bin" not in df.columns:
            raise SystemExit(f"{name} shards lack dens_bin: re-run stage16_pair_hazard.py --force")
    return cells, states, val


def cell_tensor(cells: pd.DataFrame, weeks: list[str]):
    """Return days list and tensor [day, pred, dens, m(10), tau(5), 4] (n_pairs, lab120, lab300, lab600)."""
    c = cells[cells.week.isin(weeks)]
    days = sorted(c.day.unique())
    di = {d: i for i, d in enumerate(days)}
    T = np.zeros((len(days), 2, ND, 10, 5, 4))
    cols = ["n_pairs"] + [f"n_lab_{w}" for w in LABEL_W]
    pi = c.predictor.map({"R": 0, "C": 1}).to_numpy()
    idx = (c.day.map(di).to_numpy(), pi, c.dens_bin.to_numpy(), c.m_bin.to_numpy(), c.tau_bin.to_numpy())
    for j, col in enumerate(cols):
        np.add.at(T[..., j], idx, c[col].to_numpy())
    return days, T


def state_tensor(states: pd.DataFrame, weeks: list[str]):
    s = states[states.week.isin(weeks)]
    days = sorted(s.day.unique())
    di = {d: i for i, d in enumerate(days)}
    T = np.zeros((len(days), 2, ND, len(STATE_H), 4, 4))
    hi = s.horizon_s.map({h: i for i, h in enumerate(STATE_H)}).to_numpy()
    pi = s.predictor.map({"R": 0, "C": 1}).to_numpy()
    idx = (s.day.map(di).to_numpy(), pi, s.dens_bin.to_numpy(), hi, s.state.to_numpy())
    for j, col in enumerate(["n_pairs"] + [f"n_lab_{w}" for w in LABEL_W]):
        np.add.at(T[..., j], idx, s[col].to_numpy())
    return days, T


# ----------------------------------------------------------------------------- lift maths
def lift_from_sums(n_set: np.ndarray, lab_set: np.ndarray, n_base: np.ndarray, lab_base: np.ndarray):
    """n_set, lab_set, n_base, lab_base: arrays over density strata. Returns (lift, obs_rate, exp_rate)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r_base = np.where(n_base > 0, lab_base / n_base, 0.0)
    expected = float((n_set * r_base).sum())
    obs = float(lab_set.sum())
    n = float(n_set.sum())
    return (obs / expected if expected > 0 else np.nan), (obs / n if n else np.nan), (expected / n if n else np.nan)


def _band(lo, hi):
    mk = np.zeros((10, 5), dtype=bool)
    mk[lo:hi + 1, :4] = True
    return mk


def cum_mask(a_idx: int, t_idx: int):
    mk = np.zeros((10, 5), dtype=bool)
    mk[: a_idx + 1, : t_idx + 1] = True
    return mk


def cell_mask(m_idx: int, t_idx: int):
    mk = np.zeros((10, 5), dtype=bool)
    mk[m_idx, t_idx] = True
    return mk


def boot_weights(n_days: int, B: int, rng):
    return rng.multinomial(n_days, np.ones(n_days) / n_days, size=B).astype(float)


def lift_ci(T: np.ndarray, p: int, j: int, mask: np.ndarray, wts: np.ndarray, base_cell=(9, 4)):
    """T [day,pred,dens,m,tau,4]. j: 1..3 label index. Returns point, lo, hi, n_pairs, obs_rate."""
    X = T[:, p]  # day,dens,m,tau,4
    n_set_d = (X[..., 0] * mask).sum((2, 3))  # day,dens
    lab_set_d = (X[..., j] * mask).sum((2, 3))
    n_base_d = X[:, :, base_cell[0], base_cell[1], 0]
    lab_base_d = X[:, :, base_cell[0], base_cell[1], j]
    pt, rate, _ = lift_from_sums(n_set_d.sum(0), lab_set_d.sum(0), n_base_d.sum(0), lab_base_d.sum(0))
    ns, ls = wts @ n_set_d, wts @ lab_set_d  # B,dens
    nb, lb = wts @ n_base_d, wts @ lab_base_d
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(nb > 0, lb / nb, 0.0)
        exp = (ns * rb).sum(1)
        bl = ls.sum(1) / exp
    bl = bl[np.isfinite(bl)]
    lo, hi = (np.percentile(bl, [2.5, 97.5]) if len(bl) > 20 else (np.nan, np.nan))
    return pt, lo, hi, float(n_set_d.sum()), rate


# ----------------------------------------------------------------------------- analyses
def table_cells(T, wts, W=300):
    j = 1 + LABEL_W.index(W)
    rows = []
    for p, pn in enumerate(PRED):
        for m in range(NM):
            for t in range(NT):
                pt, lo, hi, n, rate = lift_ci(T, p, j, cell_mask(m, t), wts)
                rows.append(dict(predictor=pn, W=W, m_bin=M_LABELS[m], m_idx=m, tau_bin=TAU_LABELS[t], tau_idx=t,
                                 n_pairs=n, order_rate=rate, lift=pt, lift_lo=lo, lift_hi=hi))
    return pd.DataFrame(rows)


def table_cumulative(T, wts, W=300):
    j = 1 + LABEL_W.index(W)
    rows = []
    for p, pn in enumerate(PRED):
        for a in range(NM):
            for t in range(NT):
                pt, lo, hi, n, rate = lift_ci(T, p, j, cum_mask(a, t), wts)
                rows.append(dict(predictor=pn, W=W, m_max=M_UPPER[a], a_idx=a, tau_max_s=TAU_MAX_S[t], t_idx=t,
                                 n_pairs=n, order_rate=rate, lift=pt, lift_lo=lo, lift_hi=hi))
    return pd.DataFrame(rows)


def table_states(S, wts, W=300):
    j = 1 + LABEL_W.index(W)
    rows = []
    for p, pn in enumerate(PRED):
        for h, hs in enumerate(STATE_H):
            X = S[:, p, :, h]  # day,dens,state,4
            n0, l0 = X[:, :, 0, 0], X[:, :, 0, j]
            for st in range(4):
                ns_d, ls_d = X[:, :, st, 0], X[:, :, st, j]
                if ns_d.sum() == 0:
                    continue
                pt, rate, _ = lift_from_sums(ns_d.sum(0), ls_d.sum(0), n0.sum(0), l0.sum(0))
                ns, ls, nb, lb = wts @ ns_d, wts @ ls_d, wts @ n0, wts @ l0
                with np.errstate(divide="ignore", invalid="ignore"):
                    bl = ls.sum(1) / (ns * np.where(nb > 0, lb / nb, 0)).sum(1)
                bl = bl[np.isfinite(bl)]
                lo, hi = np.percentile(bl, [2.5, 97.5]) if len(bl) > 20 else (np.nan, np.nan)
                rows.append(dict(predictor=pn, W=W, horizon_s=hs, state=STATE_NAMES[st], n_pairs=float(ns_d.sum()),
                                 order_rate=rate, lift_vs_green=pt, lift_lo=lo, lift_hi=hi))
    return pd.DataFrame(rows)


def table_density(T, W=300):
    j = 1 + LABEL_W.index(W)
    rows = []
    for p, pn in enumerate(PRED):
        X = T[:, p].sum(0)
        for d in range(ND):
            base = X[d, 9, 4]
            near = X[d, :1, :4].sum((0, 1))  # m<1
            mid = X[d, :3, :4].sum((0, 1))  # m<1.5
            rows.append(dict(predictor=pn, dens_bin=d, base_rate=base[j] / max(base[0], 1),
                             rate_m_lt_1=near[j] / max(near[0], 1), rate_m_lt_1_5=mid[j] / max(mid[0], 1),
                             n_base=base[0], n_m_lt_1=near[0], n_m_lt_1_5=mid[0]))
    return pd.DataFrame(rows)


def table_validation(val: pd.DataFrame):
    g = []
    for h, d in val.groupby("horizon_s"):
        for pn in PRED:
            e, v = d[f"err_{pn}"], d[f"verr_{pn}"]
            g.append(dict(horizon_s=h, predictor=pn, n=len(d), h_median_nm=e.median(), h_p90_nm=e.quantile(.9),
                          h_p95_nm=e.quantile(.95), v_median_ft=v.median(), v_p90_ft=v.quantile(.9),
                          v_p95_ft=v.quantile(.95)))
    return pd.DataFrame(g)


def choose_cutpoints(cum_train: pd.DataFrame, pred="R", tiers=(("yellow", 1.5), ("orange", 2.0), ("red", 2.75)), m_cap=3):
    """Broadest (a, T) rule whose bootstrap lower bound on train lift reaches the tier threshold.

    Rules are capped at m_max <= m_cap: pairs with m in 4-12 already show a background lift of ~1.3
    (they share busy flows/sectors), so wide rules would only re-describe that floor.
    """
    d = cum_train[(cum_train.predictor == pred) & (cum_train.m_max <= m_cap)]
    out = {}
    for name, thr in tiers:
        ok = d[(d.lift_lo >= thr)]
        if ok.empty:
            out[name] = None
            continue
        best = ok.sort_values("n_pairs", ascending=False).iloc[0]
        out[name] = dict(m_max=float(best.m_max), tau_max_s=int(best.tau_max_s), lift_train=float(best.lift),
                         lift_train_lo=float(best.lift_lo), lift_train_hi=float(best.lift_hi),
                         order_rate_train=float(best.order_rate), threshold=thr)
    return out


def evaluate_rules(T, rules: dict, wts, W=300, pred=0):
    j = 1 + LABEL_W.index(W)
    rows = []
    for name, r in rules.items():
        if r is None:
            continue
        a = M_UPPER.index(r["m_max"])
        t = TAU_MAX_S.index(r["tau_max_s"])
        pt, lo, hi, n, rate = lift_ci(T, pred, j, cum_mask(a, t), wts)
        rows.append(dict(tier=name, m_max=r["m_max"], tau_max_s=r["tau_max_s"], n_pairs=n, order_rate=rate,
                         lift=pt, lift_lo=lo, lift_hi=hi))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def make_figures(fig_dir: Path, cellt, cum, st, val, dens, evals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # 1 heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, pn in zip(axs, PRED):
        d = cellt[cellt.predictor == pn]
        M = d.pivot(index="m_idx", columns="tau_idx", values="lift").to_numpy()
        im = ax.imshow(M, origin="lower", cmap="RdYlBu_r", vmin=0.6, vmax=2.0, aspect="auto")
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(NT)); ax.set_xticklabels(TAU_LABELS, fontsize=8)
        ax.set_yticks(range(NM)); ax.set_yticklabels(M_LABELS, fontsize=8)
        ax.set_title("Prediccion: " + ("ruta (R)" if pn == "R" else "vel. constante (C)"), fontsize=9)
        ax.set_xlabel("tiempo hasta la separacion minima")
    axs[0].set_ylabel("separacion normalizada minima prevista m")
    fig.colorbar(im, ax=axs, label="lift (orden en 5 min)")
    p = fig_dir / "fig1_lift_heatmap.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths["heat"] = p
    # 2 cumulative curves
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for ax, pn in zip(axs, PRED):
        d = cum[cum.predictor == pn]
        for t in range(NT):
            e = d[d.t_idx == t]
            ax.plot(range(NM), e.lift, "-o", ms=3, color=cols[t], label=f"tau <= {TAU_MAX_S[t] // 60} min")
            ax.fill_between(range(NM), e.lift_lo, e.lift_hi, color=cols[t], alpha=.12)
        ax.axhline(1, color="k", lw=.6)
        ax.set_xticks(range(NM)); ax.set_xticklabels([f"<{u}" for u in M_UPPER], fontsize=7)
        ax.set_xlabel("m minimo previsto menor que"); ax.set_title(f"Predictor {pn}", fontsize=9)
    axs[0].set_ylabel("lift acumulado (estandarizado por densidad)"); axs[0].legend(fontsize=7)
    p = fig_dir / "fig2_cumulative_lift.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths["cum"] = p
    # 3 colour state lifts
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    colr = {"amarillo": "#e6c200", "naranja": "#ff6f00", "rojo": "#d7191c"}
    for ax, pn in zip(axs, PRED):
        d = st[st.predictor == pn]
        for s, c in colr.items():
            e = d[d.state == s]
            if e.empty:
                continue
            ax.errorbar(range(len(e)), e.lift_vs_green, yerr=[e.lift_vs_green - e.lift_lo, e.lift_hi - e.lift_vs_green],
                        fmt="-o", ms=3, color=c, label=s, capsize=2)
            ax.set_xticks(range(len(STATE_H)))
        ax.set_xticklabels([f"{h // 60} min" for h in STATE_H]); ax.axhline(1, color="k", lw=.6)
        ax.set_xlabel("horizonte del color previsto"); ax.set_title(f"Predictor {pn}", fontsize=9)
    axs[0].set_ylabel("lift respecto a verde"); axs[0].legend(fontsize=7)
    p = fig_dir / "fig3_colour_lift.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths["col"] = p
    # 4 validation
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for pn, ls in zip(PRED, ("-o", "--s")):
        d = val[val.predictor == pn]
        axs[0].plot(d.horizon_s / 60, d.h_p90_nm, ls, label=f"{pn} P90"); axs[0].plot(d.horizon_s / 60, d.h_median_nm, ls, alpha=.5, label=f"{pn} mediana")
        axs[1].plot(d.horizon_s / 60, d.v_p90_ft, ls, label=f"{pn} P90"); axs[1].plot(d.horizon_s / 60, d.v_median_ft, ls, alpha=.5, label=f"{pn} mediana")
    axs[0].set_ylabel("error horizontal (NM)"); axs[1].set_ylabel("error vertical (ft)"); axs[1].set_yscale("symlog")
    for ax in axs:
        ax.set_xlabel("horizonte (min)"); ax.legend(fontsize=7)
    p = fig_dir / "fig4_validation.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths["val"] = p
    # 5 train vs test
    if evals is not None and len(evals):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        for k, (sp, d) in enumerate(evals.groupby("split")):
            xs = np.arange(len(d)) + (k - .5) * .3
            ax.errorbar(xs, d.lift, yerr=[d.lift - d.lift_lo, d.lift_hi - d.lift], fmt="o", capsize=3, label=sp)
        names = [f"{t}\nm<{m:g}, {s // 60} min" for t, m, s in zip(d.tier, d.m_max, d.tau_max_s)]
        ax.set_xticks(range(len(d))); ax.set_xticklabels(names, fontsize=7); ax.axhline(1, color="k", lw=.6)
        ax.set_ylabel("lift"); ax.legend(fontsize=7)
        p = fig_dir / "fig5_train_test.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths["tt"] = p
    return paths


# ----------------------------------------------------------------------------- PDF
def build_pdf(pdf_path: Path, facts: dict, tabs: dict, figs: dict, interp: str | None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    h1, h2 = ss["Heading1"], ss["Heading2"]

    def tbl(df, widths=None, fs=7):
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    def img(p, w=17):
        from PIL import Image as PI
        iw, ih = PI.open(p).size
        return Image(str(p), width=w * cm, height=w * cm * ih / iw)

    def fmt(df, cols_f):
        d = df.copy()
        for c, f in cols_f.items():
            d[c] = d[c].map(lambda x: "-" if pd.isna(x) else f.format(x))
        return d

    st = []
    st += [Paragraph("Fase 16 - La geometria prevista entre pares y las ordenes del ATC", h1),
           Paragraph(facts["intro"], body), Spacer(1, 6)]
    st += [Paragraph("1. Resultado principal", h2), Paragraph(facts["headline"], body), Spacer(1, 4)]
    st += [Paragraph("2. Lift por celda (m prevista x tiempo hasta la minima)", h2),
           Paragraph("Lift = tasa de ordenes en 5 min en la celda / tasa esperada para pares lejanos con la misma densidad de trafico. "
                     "Intervalos por bootstrap de dias. 1.0 = sin senal.", body), img(figs["heat"], 16)]
    st += [PageBreak(), Paragraph("3. Lift acumulado: umbral de m y horizonte", h2),
           Paragraph(facts["cum_text"], body), img(figs["cum"], 16)]
    c = tabs["cum"]
    c = c[(c.predictor == "R") & (c.t_idx.isin([1, 2, 3]))][["m_max", "tau_max_s", "n_pairs", "order_rate", "lift", "lift_lo", "lift_hi"]]
    c = fmt(c, {"n_pairs": "{:,.0f}", "order_rate": "{:.3f}", "lift": "{:.2f}", "lift_lo": "{:.2f}", "lift_hi": "{:.2f}"})
    st += [Paragraph("Predictor R, lift acumulado (m < m_max y tau < tau_max):", small), tbl(c)]
    st += [PageBreak(), Paragraph("4. Los colores previstos a distintos horizontes", h2), Paragraph(facts["col_text"], body), img(figs["col"], 16)]
    s = fmt(tabs["states"][tabs["states"].predictor == "R"].drop(columns="W"),
            {"n_pairs": "{:,.0f}", "order_rate": "{:.3f}", "lift_vs_green": "{:.2f}", "lift_lo": "{:.2f}", "lift_hi": "{:.2f}"})
    st += [tbl(s)]
    st += [PageBreak(), Paragraph("5. Calidad de la prediccion: ruta frente a velocidad constante", h2), Paragraph(facts["val_text"], body),
           img(figs["val"], 16)]
    v = fmt(tabs["val"], {k: "{:.1f}" for k in ["h_median_nm", "h_p90_nm", "h_p95_nm", "v_median_ft", "v_p90_ft", "v_p95_ft"]})
    st += [tbl(v)]
    st += [PageBreak(), Paragraph("6. Umbrales propuestos (calibrados en 9 semanas, evaluados en 4)", h2), Paragraph(facts["cut_text"], body)]
    if "tt" in figs:
        st += [img(figs["tt"], 13)]
    e = fmt(tabs["evals"], {"n_pairs": "{:,.0f}", "order_rate": "{:.3f}", "lift": "{:.2f}", "lift_lo": "{:.2f}", "lift_hi": "{:.2f}"})
    st += [tbl(e)]
    st += [Spacer(1, 8), Paragraph("7. Efecto de la densidad de trafico", h2), Paragraph(facts["dens_text"], body)]
    dd = fmt(tabs["dens"][tabs["dens"].predictor == "R"].drop(columns=["predictor"]),
             {"base_rate": "{:.3f}", "rate_m_lt_1": "{:.3f}", "rate_m_lt_1_5": "{:.3f}", "n_base": "{:,.0f}", "n_m_lt_1": "{:,.0f}", "n_m_lt_1_5": "{:,.0f}"})
    st += [tbl(dd)]
    st += [PageBreak(), Paragraph("8. Limitaciones", h2)]
    for t in facts["limits"]:
        st += [Paragraph("- " + t, body)]
    if interp:
        st += [PageBreak(), Paragraph("9. Interpretacion", h2)]
        for para in interp.split("\n\n"):
            st += [Paragraph(para.replace("\n", " "), body), Spacer(1, 4)]
    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Fase 16").build(st)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default="data/processed/scat/16_pair_hazard")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    base = Path(a.base_dir)
    cells, states, val = load_tables(base / "tables")
    weeks = sorted(cells.week.unique())
    test_weeks = weeks[3::3]
    train_weeks = [w for w in weeks if w not in test_weeks]
    rng = np.random.default_rng(RNG_SEED)

    days, T = cell_tensor(cells, weeks)
    wts = boot_weights(len(days), a.boot, rng)
    cellt = table_cells(T, wts, 300)
    cum = table_cumulative(T, wts, 300)
    cum600 = table_cumulative(T, wts, 600)
    cum120 = table_cumulative(T, wts, 120)
    sdays, S = state_tensor(states, weeks)
    st = table_states(S, boot_weights(len(sdays), a.boot, rng), 300)
    dens = table_density(T, 300)
    valt = table_validation(val)

    dtr, Ttr = cell_tensor(cells, train_weeks)
    dte, Tte = cell_tensor(cells, test_weeks)
    cum_tr = table_cumulative(Ttr, boot_weights(len(dtr), a.boot, rng), 300)
    rules = choose_cutpoints(cum_tr, "R")
    ev_tr = evaluate_rules(Ttr, rules, boot_weights(len(dtr), a.boot, rng)); ev_tr["split"] = "train"
    ev_te = evaluate_rules(Tte, rules, boot_weights(len(dte), a.boot, rng)); ev_te["split"] = "test"
    evals = pd.concat([ev_tr, ev_te], ignore_index=True)
    # same rules under predictor C on the test set (does the route help out of sample?)
    evC = evaluate_rules(Tte, rules, boot_weights(len(dte), a.boot, rng), pred=1); evC["split"] = "test (C)"
    evals_all = pd.concat([evals, evC], ignore_index=True)

    out_t = base / "tables"
    for name, df in (("report_cells_lift", cellt), ("report_cumulative_lift_300", cum), ("report_cumulative_lift_600", cum600),
                     ("report_cumulative_lift_120", cum120), ("report_colour_state_lift", st), ("report_density", dens),
                     ("report_validation", valt), ("report_cutpoint_eval", evals_all)):
        df.to_csv(out_t / f"{name}.csv", index=False)
    figs = make_figures(base / "figures", cellt, cum, st, valt, dens,
                        pd.concat([ev_tr, ev_te], ignore_index=True) if len(ev_tr) else None)
    rep = Path(a.reports_dir); rep.mkdir(parents=True, exist_ok=True)
    (rep / "cutpoints.json").write_text(json.dumps(dict(train_weeks=train_weeks, test_weeks=test_weeks, label_window_s=300,
                                                       predictor="R", rules=rules), indent=2) + "\n")

    # ---- facts for text
    def g(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0]
    r_near = g(cum, predictor="R", a_idx=0, t_idx=3)
    c_near = g(cum, predictor="C", a_idx=0, t_idx=3)
    r_15 = g(cum, predictor="R", a_idx=2, t_idx=3)
    r_15s = g(cum, predictor="R", a_idx=2, t_idx=1)
    plateau = lift_ci(T, 0, 2, _band(6, 8), np.ones((1, len(days))))[0]
    dip = g(cellt, predictor="R", m_idx=1, tau_idx=3).lift
    base_rates = dens[dens.predictor == "R"].base_rate.round(3).tolist()
    n_pairs_all = float(T[:, 0].sum(axis=(0, 1, 2, 3))[0])
    vR = valt[(valt.predictor == "R") & (valt.horizon_s == 600)].iloc[0]
    vC = valt[(valt.predictor == "C") & (valt.horizon_s == 600)].iloc[0]
    facts = dict(
        intro=(f"Se analizan {len(weeks)} semanas SCAT ({len(days)} dias) en ENROUTE FL290-410. Cada 120 s se forman todos los pares de "
               f"aeronaves a menos de 350 NM ({n_pairs_all:,.0f} pares-instante) y se predicen sus trayectorias 20 min con dos predictores: "
               "R (sigue la ruta prevista) y C (velocidad constante). Se mide, para cada par, la separacion normalizada minima prevista "
               "m = max(H/5 NM, V/1000 ft) y cuando ocurre, y se compara con la tasa de ordenes LEVEL/HEADING emitidas a cualquiera de las "
               "dos aeronaves en los 5 min siguientes. La referencia son los pares lejanos (m >= 12) con la misma densidad de trafico."),
        headline=(f"Con prediccion por ruta, los pares con m < 1 (perdida de separacion prevista) en los proximos 20 min tienen un lift de "
                  f"{r_near.lift:.2f} (IC95% {r_near.lift_lo:.2f}-{r_near.lift_hi:.2f}); con velocidad constante, {c_near.lift:.2f} "
                  f"({c_near.lift_lo:.2f}-{c_near.lift_hi:.2f}). Para m < 1.5 el lift es {r_15.lift:.2f} ({r_15.lift_lo:.2f}-{r_15.lift_hi:.2f}) "
                  f"y si ademas el minimo ocurre en menos de 5 min, {r_15s.lift:.2f} ({r_15s.lift_lo:.2f}-{r_15s.lift_hi:.2f}). "
                  f"Fondo: los pares con m entre 4 y 12 ya tienen lift {plateau:.2f} (comparten flujos y sectores densos), asi que ese es el suelo contra el que hay que juzgar cada umbral. "
                  f"La banda m 1-1.2 (separados ~1000-1200 ft con cruce horizontal) tiene lift {dip:.2f} a 10-20 min, por debajo de la banda 1.2-1.5: el cruce con 1000 ft vertical es rutina, no conflicto."),
        cum_text=("Cada curva acumula los pares con m previsto menor que el valor del eje y minimo antes de tau. Si los umbrales 6/1.2 NM-1200 ft "
                  "(naranja, m = 1.2) y 7.5 NM-1500 ft (amarillo, m = 1.5) fueran fronteras naturales, se veria un salto del lift en esos puntos."),
        col_text=("Aqui se usa el color previsto con la regla dinamica de separation_profiles.json (amarillo/naranja requieren deterioro del separador activo) "
                  "en cada horizonte; el lift es respecto a los pares verdes con la misma densidad. Un color util deberia tener lift creciente "
                  "verde < amarillo < naranja < rojo."),
        val_text=(f"A 10 min el error horizontal P90 es {vR.h_p90_nm:.1f} NM (R) frente a {vC.h_p90_nm:.1f} NM (C); el vertical P90 es "
                  f"{vR.v_p90_ft:.0f} ft (R) frente a {vC.v_p90_ft:.0f} ft (C). Ojo: solo se evaluan aeronaves que siguen observadas en t+h "
                  "(sesgo de supervivencia)."),
        cut_text=("Regla de calibracion: sobre 9 semanas se elige, para cada nivel, la regla acumulada (m < m_max y tau < tau_max, con m_max <= 3) mas amplia cuyo "
                  "limite inferior del IC95% del lift supere 1.5 (amarillo), 2.0 (naranja) y 2.75 (rojo). Despues se evalua en las otras 4 semanas "
                  f"({', '.join(test_weeks)}). Si un nivel aparece como inexistente, ninguna regla alcanza ese lift de forma fiable. "
                  "Reglas elegidas (predictor R, orden en 5 min): " + "; ".join(
                      f"{k}: m < {v['m_max']:g} con tau < {v['tau_max_s'] // 60} min (lift train {v['lift_train']:.2f})" if v else f"{k}: ninguna"
                      for k, v in rules.items()) + "."),
        dens_text=(f"La tasa base de ordenes en 5 min (pares lejanos) crece con la densidad: {base_rates} para los cuatro estratos "
                   "(<15, 15-25, 25-35, >=35 aeronaves). Sin estandarizar, los pares cercanos se concentran en periodos densos y su lift "
                   "se inflaria."),
        limits=["Estudio observacional: una orden puede deberse a otros motivos (ruta directa, meteorologia, sectorizacion) y diluye el lift.",
                "La orden puede ir a cualquiera de las dos aeronaves y no se exige que sea por ese par; el lift mide asociacion, no causalidad.",
                "La rejilla de 10 s limita la resolucion de m cerca de 1.0; las cifras para m < 1 son aproximadas.",
                "La validacion de la prediccion solo incluye aeronaves observadas en t+h (sesgo de supervivencia).",
                "La densidad se estratifica en 4 niveles; puede quedar confusion residual (sector, hora, direccion de flujo)."],
    )
    summary = ["# Fase 16 - resumen\n", facts["intro"] + "\n", "## Resultado principal\n", facts["headline"] + "\n",
               "## Umbrales propuestos\n", "```json\n" + json.dumps(rules, indent=2) + "\n```\n", "## Evaluacion train/test\n",
               evals_all.round(3).to_markdown(index=False) if hasattr(evals_all, "to_markdown") else evals_all.round(3).to_string(index=False)]
    try:
        (rep / "STAGE16_SUMMARY.md").write_text("\n".join(summary) + "\n")
    except Exception:
        (rep / "STAGE16_SUMMARY.md").write_text("\n".join(summary[:-1]) + evals_all.round(3).to_string(index=False) + "\n")
    interp_path = rep / "stage16_interpretation.md"
    interp = interp_path.read_text() if interp_path.exists() else None
    if not a.no_pdf:
        build_pdf(rep / "STAGE16_REPORT.pdf", facts, dict(cum=cum, states=st, val=valt, evals=evals_all, dens=dens), figs, interp)
    print(json.dumps(dict(rules=rules, headline=facts["headline"]), indent=1))


if __name__ == "__main__":
    main()
