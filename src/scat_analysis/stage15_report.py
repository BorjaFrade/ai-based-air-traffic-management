#!/usr/bin/env python3
"""Stage 15 analysis and PDF: predicted-separation severity at ATC orders vs no-order controls.

Reads ``tables/predicted_separation_queries.parquet`` (written by
``stage15_predicted_separation.py``), writes CSV tables, PNG figures and
``reports/STAGE15_REPORT.pdf``. No SCAT source file is read here.

Statistical design
------------------
Every order (event) has up to K controls from the same flight. Rates are computed
per event as (event indicator) vs (mean indicator of its controls), so each event
carries the same weight and the comparison is paired. Confidence intervals are
percentile bootstrap intervals resampling *flights* (clusters), because a flight
can have several orders. Nothing here is a causal estimate: the controls share
aircraft, route and vertical phase but not the traffic situation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data/processed/scat/15_predicted_separation"
HORIZONS = ["0", "1", "2", "5", "10"]
HLABEL = {"0": "t = 0", "1": "<= 1 min", "2": "<= 2 min", "5": "<= 5 min", "10": "<= 10 min"}
SEVERITY = {"RED": 3, ">=ORANGE": 2, ">=YELLOW": 1}
THRESHOLDS = [1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
ORDER_TYPES = ["LEVEL", "HEADING"]
EVENT_COLOR, CONTROL_COLOR = "#b45309", "#155e75"
BOOT = 1000


# --------------------------------------------------------------------------- statistics
def paired(ev: pd.DataFrame, ct: pd.DataFrame, fn, rng: np.random.Generator, boot: int = BOOT) -> dict:
    """Event rate vs mean control rate with a flight-cluster bootstrap."""
    e = pd.Series(fn(ev).astype(float).to_numpy(), index=ev.event_id.to_numpy())
    c = pd.Series(fn(ct).astype(float).to_numpy()).groupby(ct.parent_event_id.to_numpy()).mean()
    frame = pd.DataFrame({"e": e}).join(c.rename("c"), how="inner")
    if frame.empty:
        return dict(n_events=0, event_rate=np.nan, control_rate=np.nan, diff=np.nan, diff_lo=np.nan,
                    diff_hi=np.nan, ratio=np.nan, ratio_lo=np.nan, ratio_hi=np.nan)
    frame["cluster"] = ev.set_index("event_id").loc[frame.index, "flight_key"].to_numpy()
    g = frame.groupby("cluster").agg(e=("e", "sum"), c=("c", "sum"), n=("e", "size"))
    idx = rng.integers(0, len(g), (boot, len(g)))
    E, C, N = g.e.to_numpy()[idx].sum(1), g.c.to_numpy()[idx].sum(1), g.n.to_numpy()[idx].sum(1)
    diff = (E - C) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(C > 0, E / C, np.nan)
    total_e, total_c, total_n = frame.e.sum(), frame.c.sum(), len(frame)
    return dict(n_events=int(total_n), event_rate=total_e / total_n, control_rate=total_c / total_n,
                diff=(total_e - total_c) / total_n, diff_lo=float(np.percentile(diff, 2.5)),
                diff_hi=float(np.percentile(diff, 97.5)),
                ratio=total_e / total_c if total_c > 0 else np.nan,
                ratio_lo=float(np.nanpercentile(ratio, 2.5)) if np.isfinite(ratio).any() else np.nan,
                ratio_hi=float(np.nanpercentile(ratio, 97.5)) if np.isfinite(ratio).any() else np.nan)


def auc_higher_is_event(event_values: pd.Series, control_values: pd.Series) -> float:
    """P(score_event > score_control) via ranks; ties count one half."""
    pooled = pd.concat([event_values, control_values], ignore_index=True)
    mask = pooled.notna().to_numpy()
    pooled = pooled[mask]
    labels = np.r_[np.ones(len(event_values)), np.zeros(len(control_values))][mask]
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pooled.rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def split(queries: pd.DataFrame, order_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = queries.loc[queries.valid_query & queries.order_type.eq(order_type)]
    return q.loc[q.query_kind.eq("event")], q.loc[q.query_kind.eq("control")]


# --------------------------------------------------------------------------- tables
def severity_table(queries, rng) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        for horizon in HORIZONS:
            for label, level in SEVERITY.items():
                res = paired(ev, ct, lambda d, h=horizon, l=level: d[f"state_c{h}"] >= l, rng)
                rows.append(dict(order_type=otype, horizon_min=horizon, severity=label, **res))
    return pd.DataFrame(rows)


def incidence_table(queries, rng) -> pd.DataFrame:
    rows = []
    grid = np.arange(0, 601, 15)
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        for label, col in (("RED", "t_first_red_s"), ("NON_NORMAL", "t_first_nonnormal_s")):
            for t in grid:
                res = paired(ev, ct, lambda d, c=col, tt=t: d[c].le(tt), rng, boot=200)
                rows.append(dict(order_type=otype, severity=label, t_s=int(t), **res))
    return pd.DataFrame(rows)


def enrichment_table(queries, rng) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        for horizon in HORIZONS:
            for x in THRESHOLDS:
                res = paired(ev, ct, lambda d, h=horizon, xx=x: d[f"minnorm_c{h}"].lt(xx), rng, boot=300)
                rows.append(dict(order_type=otype, horizon_min=horizon, threshold=x, **res))
    return pd.DataFrame(rows)


def auc_table(queries) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        for horizon in HORIZONS:
            col = f"minnorm_c{horizon}"
            rows.append(dict(order_type=otype, feature=col, auc_event_more_severe=auc_higher_is_event(
                -ev[col], -ct[col])))
        for col in ("nearest_h_nm", "n_nb_50", "n_nb_100"):
            sign = -1 if col == "nearest_h_nm" else 1
            rows.append(dict(order_type=otype, feature=col,
                             auc_event_more_severe=auc_higher_is_event(sign * ev[col], sign * ct[col])))
    return pd.DataFrame(rows)


def density_table(queries) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        for name, d in (("event", ev), ("control", ct)):
            for col in ("n_nb_50", "n_nb_100", "n_nb_used", "nearest_h_nm"):
                q = d[col].quantile([.1, .25, .5, .75, .9])
                rows.append(dict(order_type=otype, group=name, metric=col, n=int(d[col].notna().sum()),
                                 mean=d[col].mean(), p10=q[.1], p25=q[.25], median=q[.5], p75=q[.75], p90=q[.9]))
    return pd.DataFrame(rows)


def stratified_table(queries, rng) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        if ev.empty:
            continue
        edges = np.unique(ev.n_nb_100.quantile([0, .25, .5, .75, 1]).to_numpy())
        if len(edges) < 3:
            continue
        stratum = pd.cut(ev.n_nb_100, edges, include_lowest=True)
        for interval in stratum.cat.categories:
            sub = ev.loc[stratum.eq(interval)]
            sub_ct = ct.loc[ct.parent_event_id.isin(sub.event_id)]
            for label, level in (("RED", 3), (">=YELLOW", 1)):
                res = paired(sub, sub_ct, lambda d, l=level: d["state_c10"] >= l, rng, boot=300)
                rows.append(dict(order_type=otype, density_stratum=str(interval), severity=label,
                                 horizon_min="10", **res))
    return pd.DataFrame(rows)


def post_order_table(queries) -> pd.DataFrame:
    ev = queries.loc[queries.valid_query & queries.query_kind.eq("event") & queries.order_type.eq("LEVEL")]
    rows = []
    if "post_state_c10" not in ev:
        return pd.DataFrame(rows)
    for horizon in ("2", "5", "10"):
        a, c = ev[f"state_c{horizon}"] >= 3, ev[f"post_state_c{horizon}"] >= 3
        for name, mask in (("RED predicted without order, absent if order followed", a & ~c),
                           ("RED predicted without order and still present after order", a & c),
                           ("RED absent without order, created by order", ~a & c),
                           ("No RED either way", ~a & ~c)):
            rows.append(dict(horizon_min=horizon, outcome=name, n=int(mask.sum()), share=float(mask.mean()),
                             n_events=len(ev)))
    return pd.DataFrame(rows)


def coverage_table(queries) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        q = queries.loc[queries.order_type.eq(otype)]
        ev, ct = q.loc[q.query_kind.eq("event")], q.loc[q.query_kind.eq("control")]
        with_control = ev.event_id.isin(set(ct.parent_event_id)).sum()
        rows.append(dict(order_type=otype, events=len(ev), valid_events=int(ev.valid_query.sum()),
                         events_with_control=int(with_control),
                         share_with_control=with_control / max(len(ev), 1), controls=len(ct),
                         controls_per_event=len(ct) / max(with_control, 1)))
    return pd.DataFrame(rows)


def sensitivity_table(main: pd.DataFrame, alt: pd.DataFrame, rng, label: str) -> pd.DataFrame:
    rows = []
    for name, queries in (("tasa vertical observada (principal)", main), (label, alt)):
        for otype in ORDER_TYPES:
            ev, ct = split(queries, otype)
            for horizon in ("2", "5", "10"):
                for sev, level in (("RED", 3), (">=YELLOW", 1)):
                    res = paired(ev, ct, lambda d, h=horizon, l=level: d[f"state_c{h}"] >= l, rng, boot=300)
                    rows.append(dict(model=name, order_type=otype, horizon_min=horizon, severity=sev, **res))
    return pd.DataFrame(rows)


def lookahead_table(queries) -> pd.DataFrame:
    rows = []
    for otype in ORDER_TYPES:
        ev, ct = split(queries, otype)
        for name, d in (("event", ev), ("control", ct)):
            t = d.t_first_red_s.dropna()
            rows.append(dict(order_type=otype, group=name, n_queries=len(d), n_red_within_10min=len(t),
                             share_red_within_10min=len(t) / max(len(d), 1),
                             median_s=t.median(), p25_s=t.quantile(.25), p75_s=t.quantile(.75),
                             p90_s=t.quantile(.90), share_within_60s=(t <= 60).mean() if len(t) else np.nan,
                             share_within_120s=(t <= 120).mean() if len(t) else np.nan,
                             share_within_300s=(t <= 300).mean() if len(t) else np.nan,
                             far_start_share=(d.loc[d.t_first_red_s.notna(), "rel_initial_distance_nm"] > 100).mean()
                             if len(t) else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor("white")
    ax.grid(alpha=.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_severity(sev, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for j, otype in enumerate(ORDER_TYPES):
        for i, label in enumerate(("RED", ">=YELLOW")):
            ax = axes[i, j]
            d = sev.loc[sev.order_type.eq(otype) & sev.severity.eq(label)].set_index("horizon_min").loc[HORIZONS]
            x = np.arange(len(HORIZONS))
            ax.plot(x, 100 * d.event_rate, "o-", color=EVENT_COLOR, label="Orden ATC")
            ax.plot(x, 100 * d.control_rate, "s--", color=CONTROL_COLOR, label="Control sin orden")
            for xi, (e, c, lo, hi) in enumerate(zip(d.event_rate, d.control_rate, d.diff_lo, d.diff_hi)):
                if np.isfinite(lo):
                    ax.annotate(f"{100*(e-c):+.1f} pp", (xi, 100 * max(e, c)), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=8, color="#334155")
            ax.set_xticks(x, [HLABEL[h] for h in HORIZONS])
            ax.set_title(f"{otype} - estado {label} predicho (peor caso acumulado)", fontsize=10)
            ax.set_ylabel("% de consultas")
            ax.set_ylim(bottom=0)
            style(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Severidad predicha por horizonte: órdenes vs controles (diferencia pareada en pp)", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_incidence(inc, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, otype in enumerate(ORDER_TYPES):
        for i, sev in enumerate(("RED", "NON_NORMAL")):
            ax = axes[i, j]
            d = inc.loc[inc.order_type.eq(otype) & inc.severity.eq(sev)]
            ax.plot(d.t_s / 60, 100 * d.event_rate, color=EVENT_COLOR, label="Orden ATC")
            ax.plot(d.t_s / 60, 100 * d.control_rate, color=CONTROL_COLOR, ls="--", label="Control sin orden")
            ax.fill_between(d.t_s / 60, 100 * (d.control_rate + d.diff_lo), 100 * (d.control_rate + d.diff_hi),
                            color=EVENT_COLOR, alpha=.12)
            ax.set_title(f"{otype} - primera aparición de {'RED' if sev == 'RED' else 'estado no normal'}", fontsize=10)
            ax.set_ylabel("% acumulado de consultas")
            style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Tiempo de predicción (min)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Incidencia acumulada del primer estado predicho (banda: IC95 % de la diferencia pareada)", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_enrichment(enr, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, otype in zip(axes, ORDER_TYPES):
        for horizon, color in zip(HORIZONS, plt.cm.viridis(np.linspace(0, .9, len(HORIZONS)))):
            d = enr.loc[enr.order_type.eq(otype) & enr.horizon_min.eq(horizon)]
            ax.plot(d.threshold, d.ratio, "o-", color=color, label=HLABEL[horizon])
        ax.axhline(1, color="#64748b", lw=1)
        for x, name in ((1.0, "RED"), (1.2, "ORANGE"), (1.5, "YELLOW")):
            ax.axvline(x, color="#cbd5e1", lw=1)
            ax.text(x, ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7, color="#64748b")
        ax.set_xscale("log")
        ax.set_xticks(THRESHOLDS, [f"{x:g}" for x in THRESHOLDS])
        ax.set_xlabel("Umbral de separación mínima normalizada  max(H/5 NM, V/1000 ft)")
        ax.set_title(otype)
        style(ax)
    axes[0].set_ylabel("Cociente orden / control")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("¿Dónde se concentran las órdenes? Enriquecimiento frente a controles", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_density(queries, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ev, ct = split(queries, "LEVEL")
    top = float(np.nanpercentile(pd.concat([ev.n_nb_100, ct.n_nb_100]), 99))
    bins = np.linspace(0, max(top, 5), 40)
    axes[0].hist(ct.n_nb_100, bins, density=True, alpha=.6, color=CONTROL_COLOR, label="Control")
    axes[0].hist(ev.n_nb_100, bins, density=True, alpha=.6, color=EVENT_COLOR, label="Orden ATC")
    axes[0].set(xlabel="Aeronaves FL290-410 a menos de 100 NM", ylabel="Densidad", title="Densidad de tráfico (LEVEL)")
    bins = np.linspace(0, 100, 41)
    axes[1].hist(ct.nearest_h_nm.clip(upper=100), bins, density=True, alpha=.6, color=CONTROL_COLOR, label="Control")
    axes[1].hist(ev.nearest_h_nm.clip(upper=100), bins, density=True, alpha=.6, color=EVENT_COLOR, label="Orden ATC")
    axes[1].set(xlabel="Distancia horizontal al vecino más próximo (NM)", title="Vecino más próximo (LEVEL)")
    axes[0].legend(frameon=False)
    for ax in axes:
        style(ax)
    fig.suptitle("Línea base: densidad de tráfico y proximidad en órdenes vs controles", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_time_to_red(queries, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, otype in zip(axes, ORDER_TYPES):
        ev, ct = split(queries, otype)
        bins = np.arange(0, 601, 30)
        for d, color, label in ((ev, EVENT_COLOR, "Orden ATC"), (ct, CONTROL_COLOR, "Control")):
            t = d.t_first_red_s.dropna()
            ax.hist(t, bins, weights=np.full(len(t), 100.0 / max(len(d), 1)), alpha=.6, color=color, label=label)
        ax.set(xlabel="Tiempo hasta el primer RED predicho (s)", ylabel="% de consultas por intervalo de 30 s", title=otype)
        style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("¿A cuántos minutos vista aparece el primer RED predicho?", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_post(post, path):
    if post.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.4))
    horizons = sorted(post.horizon_min.unique(), key=int)
    outcomes = [o for o in post.outcome.unique() if not o.startswith("No RED")]
    colors = ["#16a34a", "#d97706", "#dc2626"]
    width = .26
    for i, (outcome, color) in enumerate(zip(outcomes, colors)):
        vals = [100 * post.loc[post.horizon_min.eq(h) & post.outcome.eq(outcome), "share"].iloc[0] for h in horizons]
        counts = [int(post.loc[post.horizon_min.eq(h) & post.outcome.eq(outcome), "n"].iloc[0]) for h in horizons]
        bars = ax.bar(np.arange(len(horizons)) + (i - 1) * width, vals, width, color=color, label=outcome)
        for bar, n in zip(bars, counts):
            ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center",
                        va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(np.arange(len(horizons)), [f"<= {h} min" for h in horizons])
    ax.set_ylabel("% de órdenes de nivel")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=1)
    ax.set_title("Órdenes de nivel: ¿evita el nuevo nivel el RED predicho? (supuesto de cumplimiento)", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- narrative and PDF
def pct(x, d=1):
    return "n/d" if not np.isfinite(x) else f"{100 * x:.{d}f} %"


def pp(x, lo, hi):
    return "n/d" if not np.isfinite(x) else f"{100 * x:+.2f} pp (IC95 {100 * lo:+.2f} a {100 * hi:+.2f})"


def verdict(diff, lo, hi):
    if not np.isfinite(diff):
        return "sin datos"
    if lo > 0:
        return "mayor en las órdenes (el IC95 excluye 0)"
    if hi < 0:
        return "menor en las órdenes (el IC95 excluye 0)"
    return "sin diferencia distinguible de 0 (el IC95 incluye 0)"


def summary_bullets(sev, look, auc, dens, post, run) -> list[str]:
    out = []
    for otype in ORDER_TYPES:
        d = sev.loc[sev.order_type.eq(otype)]
        if d.empty:
            continue
        for h in ("0", "5", "10"):
            r = d.loc[d.horizon_min.eq(h) & d.severity.eq("RED")].iloc[0]
            y = d.loc[d.horizon_min.eq(h) & d.severity.eq(">=YELLOW")].iloc[0]
            out.append(f"{otype}, {HLABEL[h]}: RED en {pct(r.event_rate, 2)} de las órdenes frente a "
                       f"{pct(r.control_rate, 2)} de los controles ({pp(r['diff'], r.diff_lo, r.diff_hi)}; "
                       f"{verdict(r['diff'], r.diff_lo, r.diff_hi)}). Algún estado no normal: "
                       f"{pct(y.event_rate)} vs {pct(y.control_rate)}.")
    for otype in ORDER_TYPES:
        row = look.loc[look.order_type.eq(otype) & look.group.eq("event")]
        if len(row) and row.iloc[0].n_red_within_10min:
            r = row.iloc[0]
            out.append(f"{otype}: entre las órdenes con RED predicho en 10 min (n={int(r.n_red_within_10min)}), "
                       f"el primer RED aparece a una mediana de {r.median_s / 60:.1f} min "
                       f"(P25 {r.p25_s / 60:.1f}, P75 {r.p75_s / 60:.1f}, P90 {r.p90_s / 60:.1f}); "
                       f"{pct(r.share_within_120s, 0)} dentro de 2 min y {pct(r.share_within_300s, 0)} dentro de 5 min.")
    a = auc.loc[auc.feature.eq("minnorm_c10")]
    for _, r in a.iterrows():
        out.append(f"{r.order_type}: AUC de 'separación mínima predicha a 10 min' para distinguir orden de control = "
                   f"{r.auc_event_more_severe:.3f} (0,5 = sin información).")
    nb = dens.loc[dens.metric.eq("n_nb_100") & dens.order_type.eq("LEVEL")]
    if len(nb) == 2:
        e, c = nb.set_index("group").loc["event"], nb.set_index("group").loc["control"]
        out.append(f"Densidad (LEVEL): mediana de {e['median']:.0f} aeronaves a <100 NM en órdenes vs {c['median']:.0f} en controles.")
    if not post.empty:
        h = post.loc[post.horizon_min.eq("10")].set_index("outcome")
        resolved, persists = h.iloc[0], h.iloc[1]
        denom = resolved.n + persists.n
        if denom:
            out.append(f"LEVEL: de {int(denom)} órdenes con RED predicho en 10 min sin intervención, {pct(resolved.n / denom, 0)} "
                       f"dejan de tenerlo si la aeronave cumple el nuevo nivel (supuesto: retardo "
                       f"{run['config']['delay_s']:.0f} s, {run['config']['nominal_fpm']:.0f} ft/min).")
    return out


def build_pdf(path: Path, figs: dict, tables: dict, bullets: list[str], run: dict, notes: str | None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.2, leading=12.4, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor("#475569"))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, textColor=colors.HexColor("#153d59"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#153d59"), spaceBefore=8)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#456172"))
        canvas.drawString(1.5 * cm, 1.0 * cm, "SCAT Stage 15 | Severidad predicha por horizonte | órdenes vs controles")
        canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, str(doc.page))
        canvas.restoreState()

    def table(df: pd.DataFrame, widths=None, font=7.2):
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2eaf0")),
                               ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#cbd5e1")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    def image(file, width=17.5 * cm):
        from PIL import Image as PILImage
        w, h = PILImage.open(file).size
        return Image(str(file), width=width, height=width * h / w)

    story = [Paragraph("Stage 15 - Severidad predicha por horizonte en órdenes ATC (SCAT)", h1),
             Paragraph(f"Generado el {datetime.now():%d/%m/%Y %H:%M} | {run['n_events']:,} órdenes y {run['n_controls']:,} "
                       f"controles | {len(run['weeks'])} semanas | reglas: {run['rules'].get('logic_version')}", small),
             Paragraph("Resumen", h2)]
    story += [Paragraph(b, bullet, bulletText="-") for b in bullets]
    cov = tables["coverage"].copy()
    cov["share_with_control"] = cov.share_with_control.map(lambda x: f"{100 * x:.1f} %")
    cov["controls_per_event"] = cov.controls_per_event.map(lambda x: f"{x:.2f}")
    story += [Paragraph("Cobertura", h2),
              table(cov.rename(columns={"order_type": "Tipo", "events": "Órdenes", "valid_events": "Con cinemática válida",
                                        "events_with_control": "Con >=1 control", "share_with_control": "% con control",
                                        "controls": "Controles", "controls_per_event": "Controles/orden"}), font=7.4),
              Paragraph("Las comparaciones pareadas solo usan las órdenes con al menos un control. Las órdenes de rumbo quedan "
                        "menos cubiertas porque suelen darse en ráfagas y no dejan un intervalo de 120 s sin órdenes de esa aeronave.", small),
              Paragraph("Qué se ha medido", h2),
              Paragraph("Para cada orden de cambio de nivel (Stage 13) o de rumbo (Stage 14) en FL290-410, y para hasta "
                        f"{run['config']['controls']} instantes de control de la misma aeronave sin ninguna orden en "
                        f"{run['config']['guard_s']:.0f} s y con la misma fase vertical, se propagan a velocidad constante "
                        f"todas las aeronaves a menos de {run['config']['radius_nm']:.0f} NM (último plot de 10 s o menos, "
                        "alineado al instante de la consulta) en una rejilla de 5 s hasta 10 min. El estado de cada pareja "
                        "(NORMAL, YELLOW, ORANGE, RED) usa la regla dinámica ENROUTE de separation_profiles.json "
                        "(RED: H < 5 NM y V < 1000 ft; amarillo y naranja exigen que el separador activo se deteriore). "
                        "Se guarda el peor estado en t=0 y en los horizontes acumulados 1, 2, 5 y 10 min, el tiempo hasta el "
                        "primer RED y la separación mínima normalizada max(H/5, V/1000), que no depende de ningún umbral.", body),
              Paragraph("Modelo vertical A (principal): tasa vertical observada constante con zona muerta de 100 ft/min. "
                        "Modelo C (solo órdenes de nivel): la aeronave cumple el nivel autorizado tras un retardo; es un supuesto "
                        "de trabajo, no una observación.", body)]
    story.append(PageBreak())
    story += [Paragraph("1. Severidad predicha por horizonte", h2), image(figs["severity"], 15.5 * cm),
              Paragraph("Cada punto es el porcentaje de consultas cuyo peor estado predicho hasta ese horizonte alcanza la "
                        "severidad indicada. La anotación es la diferencia pareada orden - control en puntos porcentuales.", small)]
    sev = tables["severity"]
    show = sev.loc[sev.severity.isin(["RED", ">=YELLOW"])].copy()
    show["Orden"] = show.event_rate.map(pct)
    show["Control"] = show.control_rate.map(pct)
    show["Diferencia (IC95)"] = [pp(a, b, c) for a, b, c in zip(show["diff"], show.diff_lo, show.diff_hi)]
    show["Cociente"] = show.ratio.map(lambda x: "n/d" if not np.isfinite(x) else f"{x:.2f}")
    story.append(table(show[["order_type", "horizon_min", "severity", "n_events", "Orden", "Control", "Diferencia (IC95)", "Cociente"]]
                       .rename(columns={"order_type": "Tipo", "horizon_min": "Horizonte (min)", "severity": "Severidad", "n_events": "N"}),
                       widths=[1.6 * cm, 2.2 * cm, 2.2 * cm, 1.5 * cm, 1.9 * cm, 1.9 * cm, 5.6 * cm, 1.5 * cm]))
    story.append(PageBreak())
    story += [Paragraph("2. Cuándo aparece el primer RED predicho", h2), image(figs["incidence"]), image(figs["time_to_red"], 16.5 * cm)]
    look = tables["lookahead"].copy()
    for c in ("share_red_within_10min", "share_within_60s", "share_within_120s", "share_within_300s", "far_start_share"):
        look[c] = look[c].map(lambda x: "n/d" if not np.isfinite(x) else f"{100 * x:.1f} %")
    for c in ("median_s", "p25_s", "p75_s", "p90_s"):
        look[c] = look[c].map(lambda x: "n/d" if not np.isfinite(x) else f"{x:.0f}")
    story.append(table(look.rename(columns={"order_type": "Tipo", "group": "Grupo", "n_queries": "N", "n_red_within_10min": "N RED<=10",
                                            "share_red_within_10min": "% RED<=10", "median_s": "Med (s)", "p25_s": "P25", "p75_s": "P75",
                                            "p90_s": "P90", "share_within_60s": "<=60 s", "share_within_120s": "<=120 s",
                                            "share_within_300s": "<=300 s", "far_start_share": "inicio >100 NM"}), font=6.6))
    story.append(Paragraph("'inicio >100 NM': porcentaje de los casos con RED cuyo vecino relevante estaba a más de 100 NM al emitir la orden; "
                           "mide cuánto se perdería si el radio de preselección fuese 100 NM en vez de 150.", small))
    story.append(PageBreak())
    story += [Paragraph("3. Calibración de umbrales: dónde se concentran las órdenes", h2), image(figs["enrichment"], 17 * cm),
              Paragraph("Cociente entre el porcentaje de órdenes y de controles cuya separación mínima normalizada predicha cae por debajo "
                        "del umbral. 1,0 = RED, 1,2 = envolvente naranja, 1,5 = envolvente amarilla del perfil actual (sin la condición "
                        "de deterioro). Un máximo en otro valor sugiere un umbral distinto al actual.", small)]
    auc = tables["auc"].copy()
    auc["auc_event_more_severe"] = auc.auc_event_more_severe.map(lambda x: f"{x:.3f}")
    story.append(table(auc.rename(columns={"order_type": "Tipo", "feature": "Variable", "auc_event_more_severe": "AUC (orden vs control)"}),
                       widths=[2.5 * cm, 4 * cm, 4 * cm]))
    story.append(Paragraph("AUC > 0,5: la orden tiene valores más severos (menor separación, más tráfico) que el control.", small))
    story.append(PageBreak())
    story += [Paragraph("4. Línea base de tráfico", h2), image(figs["density"], 17 * cm)]
    strat = tables["stratified"]
    if not strat.empty:
        s = strat.loc[strat.severity.eq(">=YELLOW")].copy()
        s["Orden"] = s.event_rate.map(pct)
        s["Control"] = s.control_rate.map(pct)
        s["Diferencia (IC95)"] = [pp(a, b, c) for a, b, c in zip(s["diff"], s.diff_lo, s.diff_hi)]
        story += [Paragraph("Diferencia orden - control (estado no normal a 10 min) dentro de cuartiles de densidad:", body),
                  table(s[["order_type", "density_stratum", "n_events", "Orden", "Control", "Diferencia (IC95)"]]
                        .rename(columns={"order_type": "Tipo", "density_stratum": "Aeronaves <100 NM", "n_events": "N"}),
                        widths=[2 * cm, 3.5 * cm, 1.6 * cm, 2 * cm, 2 * cm, 6 * cm])]
    if figs.get("post"):
        story += [Spacer(1, 8), Paragraph("5. ¿Evita la orden de nivel el RED predicho?", h2), image(figs["post"], 14 * cm),
                  Paragraph("Compara el modelo A (sin la orden) con el modelo C (la aeronave sigue el nivel autorizado). No demuestra "
                            "que el controlador pretendiera evitar esa pareja; solo cuantifica cuántas órdenes tienen ese efecto "
                            "geométrico si se cumplen.", small)]
    sens = tables.get("sensitivity")
    if sens is not None and not sens.empty:
        s = sens.copy()
        s["Orden"] = s.event_rate.map(pct)
        s["Control"] = s.control_rate.map(pct)
        s["Diferencia (IC95)"] = [pp(a, b, c) for a, b, c in zip(s["diff"], s.diff_lo, s.diff_hi)]
        story += [PageBreak(), Paragraph("6. Sensibilidad al modelo vertical", h2),
                  Paragraph("El modelo principal mantiene la tasa vertical observada de todas las aeronaves durante 10 min, lo que "
                            "puede generar cruces espurios con aeronaves que en realidad se nivelarían. La sensibilidad supone que "
                            "nadie cambia de altitud.", body),
                  table(s[["model", "order_type", "horizon_min", "severity", "Orden", "Control", "Diferencia (IC95)"]]
                        .rename(columns={"model": "Modelo", "order_type": "Tipo", "horizon_min": "Horizonte (min)", "severity": "Severidad"}),
                        font=6.6)]
    if notes:
        story += [PageBreak(), Paragraph("Interpretación", h2)]
        story += [Paragraph(p.strip(), body) for p in notes.split("\n\n") if p.strip()]
    story += [Paragraph("Límites", h2)]
    for t in ("La predicción es lineal y no modela maniobras futuras ni la propia orden (salvo el modelo C para el nivel).",
              "Un control comparte aeronave, ruta y fase vertical con la orden pero no la situación de tráfico; la diferencia no es causal.",
              "Una orden puede responder a varias aeronaves, a restricciones de ruta, a coordinación o a eficiencia; el vecino relevante es "
              "el de menor separación predicha, no necesariamente el motivo de la orden.",
              "El RED predicho es una condición geométrica, no un conflicto confirmado ni una etiqueta experta.",
              "Los plots de cada aeronave no son simultáneos; se alinean al instante de la orden por extrapolación lineal."):
        story.append(Paragraph(t, bullet, bulletText="-"))
    story += [Paragraph("Reproducibilidad", h2),
              Paragraph(f"python3 src/scat_analysis/stage15_predicted_separation.py | python3 src/scat_analysis/stage15_report.py. "
                        f"Cálculo: {run['elapsed_seconds']:.0f} s, pico de memoria {run['peak_rss_mib']:.0f} MiB. "
                        f"Semilla {run['config']['seed']}, radio {run['config']['radius_nm']:.0f} NM, retardo {run['config']['delay_s']:.0f} s.", small)]
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm,
                      bottomMargin=1.8 * cm, title="SCAT Stage 15").build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--compare-dir", type=Path, default=None,
                        help="Output folder of a --vertical-model level run, for the sensitivity table")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    queries = pd.read_parquet(out / "tables" / "predicted_separation_queries.parquet")
    run = json.loads((out / "diagnostics" / "run.json").read_text())
    rng = np.random.default_rng(args.seed)
    tables = {"severity": severity_table(queries, rng), "incidence": incidence_table(queries, rng),
              "enrichment": enrichment_table(queries, rng), "auc": auc_table(queries),
              "density": density_table(queries), "stratified": stratified_table(queries, rng),
              "post": post_order_table(queries), "lookahead": lookahead_table(queries),
              "coverage": coverage_table(queries)}
    compare = args.compare_dir or (out / "sensitivity_level_hold")
    alt_path = compare / "tables" / "predicted_separation_queries.parquet"
    if alt_path.exists():
        tables["sensitivity"] = sensitivity_table(queries, pd.read_parquet(alt_path), rng,
                                                  "altitud constante (sensibilidad)")
    names = {"severity": "severity_by_horizon", "incidence": "cumulative_incidence_first_state",
             "enrichment": "minnorm_enrichment", "auc": "auc_event_vs_control", "density": "density_baseline",
             "stratified": "stratified_by_density", "post": "post_order_outcomes", "lookahead": "lookahead_time_to_red",
             "coverage": "coverage", "sensitivity": "sensitivity_vertical_model"}
    for key, frame in tables.items():
        frame.to_csv(out / "tables" / f"{names[key]}.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    figs = {"severity": fig_dir / "stage15_severity_by_horizon.png", "incidence": fig_dir / "stage15_cumulative_incidence.png",
            "enrichment": fig_dir / "stage15_enrichment.png", "density": fig_dir / "stage15_density_baseline.png",
            "time_to_red": fig_dir / "stage15_time_to_red.png", "post": fig_dir / "stage15_post_order.png"}
    fig_severity(tables["severity"], figs["severity"])
    fig_incidence(tables["incidence"], figs["incidence"])
    fig_enrichment(tables["enrichment"], figs["enrichment"])
    fig_density(queries, figs["density"])
    fig_time_to_red(queries, figs["time_to_red"])
    figs["post"] = figs["post"] if fig_post(tables["post"], figs["post"]) else None
    bullets = summary_bullets(tables["severity"], tables["lookahead"], tables["auc"], tables["density"], tables["post"], run)
    notes_path = out / "reports" / "interpretation.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else None
    pdf = out / "reports" / "STAGE15_REPORT.pdf"
    build_pdf(pdf, figs, tables, bullets, run, notes)
    (out / "reports" / "STAGE15_SUMMARY.md").write_text("# Stage 15 summary\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n",
                                                        encoding="utf-8")
    print(f"{pdf}\n" + "\n".join(f"- {b}" for b in bullets))


if __name__ == "__main__":
    main()
