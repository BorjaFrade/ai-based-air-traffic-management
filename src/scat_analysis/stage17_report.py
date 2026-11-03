#!/usr/bin/env python3
"""Stage 17 report: final colour proposal + merged detailed PDF (proposal + Stage 15 + Stage 16) + executive summary.

Needs: stage17_colour_schemes.py already run (tables/scheme_tiers.csv), Stage 15 and 16 PDFs.
Outputs (reports/): PROPUESTA_COLORES_DETALLADO.pdf, RESUMEN_EJECUTIVO_TUTORES.pdf, proposed_colour_profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GREEN, YELLOW, ORANGE, RED = "#00A651", "#FFD400", "#FF6F00", "#D7191C"
TIER_COL = {"verde": GREEN, "amarillo": YELLOW, "naranja": ORANGE, "rojo": RED}
PROPOSAL = "4 colores (propuesta)"
SPLIT_TEST, SPLIT_TRAIN = "test (4 sem)", "calibracion (9 sem)"


def get(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def figures(df, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    # A: proposal lifts (R), train vs test
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    tiers = ["verde", "amarillo", "naranja", "rojo"]
    for k, (sp, off) in enumerate(((SPLIT_TRAIN, -.18), (SPLIT_TEST, .18))):
        d = get(df, scheme=PROPOSAL, predictor="R", W=300, split=sp).set_index("tier").loc[tiers]
        axs[0].bar(np.arange(4) + off, d.order_rate * 100, .34, color=[TIER_COL[t] for t in tiers],
                   alpha=.55 if k == 0 else 1, edgecolor="k", linewidth=.4, label=sp)
        axs[1].errorbar(np.arange(4) + off, d.lift, yerr=[d.lift - d.lift_lo, d.lift_hi - d.lift], fmt="o", capsize=3,
                        color="k", alpha=.5 if k == 0 else 1, label=sp)
    for ax in axs:
        ax.set_xticks(range(4)); ax.set_xticklabels(tiers)
    axs[0].set_ylabel("% de pares con orden en los 5 min siguientes"); axs[1].set_ylabel("lift respecto a verde (estandarizado por densidad)")
    axs[1].axhline(1, color="k", lw=.5); axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
    fig.tight_layout(); p = fig_dir / "fig_proposal_lift.png"; fig.savefig(p, dpi=150); plt.close(fig); out["prop"] = p
    # B: R vs C, three windows
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, W in zip(axs, (120, 300, 600)):
        for pn, off, a in (("R", -.18, 1), ("C", .18, .45)):
            d = get(df, scheme=PROPOSAL, predictor=pn, W=W, split=SPLIT_TEST).set_index("tier").loc[tiers]
            ax.bar(np.arange(4) + off, d.lift, .34, color=[TIER_COL[t] for t in tiers], alpha=a, edgecolor="k", linewidth=.4,
                   hatch="" if pn == "R" else "//")
            ax.errorbar(np.arange(4) + off, d.lift, yerr=[d.lift - d.lift_lo, d.lift_hi - d.lift], fmt="none", ecolor="k", capsize=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(tiers, fontsize=8); ax.set_title(f"orden en {W // 60 if W >= 60 else W} min" if W != 120 else "orden en 2 min", fontsize=9)
    axs[0].set_ylabel("lift (test): liso = ruta, rayado = vel. constante")
    fig.tight_layout(); p = fig_dir / "fig_route_vs_cv.png"; fig.savefig(p, dpi=150); plt.close(fig); out["rc"] = p
    # C: scheme comparison: ratio between consecutive tiers
    q = get(df, predictor="R", W=300, split=SPLIT_TEST)
    q = q[q.ratio_prev.notna()]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    labels, y, lo, hi, cols = [], [], [], [], []
    for i, (sn, d) in enumerate(q.groupby("scheme", sort=False)):
        for _, rw in d.iterrows():
            labels.append(f"{sn}\n{rw.tier}"); y.append(rw.ratio_prev); lo.append(rw.ratio_prev - rw.ratio_lo)
            hi.append(rw.ratio_hi - rw.ratio_prev); cols.append(TIER_COL.get(rw.tier, "#999999"))
    ax.barh(range(len(y)), y, xerr=[lo, hi], color=cols, edgecolor="k", linewidth=.4, capsize=2)
    ax.axvline(1, color="k", lw=.8); ax.set_yticks(range(len(y))); ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("cociente de lift con el nivel anterior (test); 1 = el color extra no aporta")
    fig.tight_layout(); p = fig_dir / "fig_scheme_ratios.png"; fig.savefig(p, dpi=150); plt.close(fig); out["ratios"] = p
    return out


def styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    ss = getSampleStyleSheet()
    return dict(
        body=ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13),
        small=ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.5, leading=9.5),
        cell=ParagraphStyle("c", parent=ss["BodyText"], fontSize=8, leading=10),
        h1=ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, leading=21),
        h2=ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16),
        title=ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=25),
    )


def build_story(df, figs, st, executive: bool):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, Spacer, Table, TableStyle
    from PIL import Image as PI

    P = lambda t, s="body": Paragraph(t, st[s])  # noqa: E731

    def img(p, w=17):
        iw, ih = PI.open(p).size
        return Image(str(p), width=w * cm, height=w * cm * ih / iw)

    def table(rows, widths, head=True, colour_first=False, fs=8):
        data = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
        sty = [("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        if head:
            sty.append(("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey))
        if colour_first:
            for i, r in enumerate(rows[1:], start=1):
                c = TIER_COL.get(str(r[0]).lower().replace("<b>", "").replace("</b>", ""))
                if c:
                    sty.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(c)))
        t.setStyle(TableStyle(sty))
        return t

    t4 = get(df, scheme=PROPOSAL, predictor="R", W=300, split=SPLIT_TEST).set_index("tier")
    tr4 = get(df, scheme=PROPOSAL, predictor="R", W=300, split=SPLIT_TRAIN).set_index("tier")

    def f(x, d=2):
        return f"{x:.{d}f}".replace(".", ",")

    def pct(x):
        return f"{100 * x:.1f}".replace(".", ",") + " %"

    def pct2(x):
        return f"{100 * x:.2f}".replace(".", ",") + " %"

    def tier_row(name, rule, meaning):
        r = t4.loc[name]
        return [f"<b>{name}</b>", rule, meaning, pct(r.order_rate), f"{f(r.lift)} ({f(r.lift_lo)}-{f(r.lift_hi)})",
                f"{f(tr4.loc[name].lift)}", pct(r.share_pairs) if name != "verde" else "resto"]

    rows = [["Color", "Regla (predictor por ruta, 20 min)", "Significado", "Órdenes en 5 min", "Lift test (IC95%)", "Lift calib.", "% de pares"],
            ["<b>verde</b>", "ningún par con m < 2 previsto", "sin proximidad relevante prevista", pct(t4.loc["verde"].order_rate), "1,00 (referencia)", "1,00", "resto"],
            ["<b>amarillo</b>", "1 <= m < 2 (menos de 10 NM y 2000 ft a la vez, sin llegar a pérdida)", "proximidad prevista, vigilancia", pct(t4.loc["amarillo"].order_rate),
             f"{f(t4.loc['amarillo'].lift)} ({f(t4.loc['amarillo'].lift_lo)}-{f(t4.loc['amarillo'].lift_hi)})", f(tr4.loc["amarillo"].lift), pct2(t4.loc["amarillo"].share_pairs)],
            ["<b>naranja</b>", "m < 1 (pérdida de separación prevista) con mínimo a 5-20 min", "conflicto previsto con margen para resolverlo", pct(t4.loc["naranja"].order_rate),
             f"{f(t4.loc['naranja'].lift)} ({f(t4.loc['naranja'].lift_lo)}-{f(t4.loc['naranja'].lift_hi)})", f(tr4.loc["naranja"].lift), pct2(t4.loc["naranja"].share_pairs)],
            ["<b>rojo</b>", "m < 1 con mínimo en menos de 5 min (incluye pérdida actual)", "pérdida de separación inminente o en curso", pct(t4.loc["rojo"].order_rate),
             f"{f(t4.loc['rojo'].lift)} ({f(t4.loc['rojo'].lift_lo)}-{f(t4.loc['rojo'].lift_hi)})", f(tr4.loc["rojo"].lift), pct2(t4.loc["rojo"].share_pairs)]]
    S = []
    S += [P("Propuesta final de clasificación por colores", "h1" if not executive else "title")]
    S += [P("ENROUTE FL290-410, SCAT (13 semanas, 107 días). Fases 15 y 16.")]
    S += [Spacer(1, 6)]
    S += [P("1. Propuesta", "h2"),
          P("Se propone <b>4 colores</b> (verde, amarillo, naranja, rojo), definidos sobre la <b>separación normalizada mínima prevista</b> "
            "m = max(H/5 NM, V/1000 ft) en los próximos 20 min, calculada con la trayectoria que sigue la ruta prevista, y sobre el "
            "<b>tiempo hasta ese mínimo</b>. m = 1 equivale a la violación actual (5 NM y 1000 ft); m = 2 equivale a 10 NM y 2000 ft.")]
    S += [Spacer(1, 4), table(rows, [1.7, 4.3, 3.4, 1.8, 2.4, 1.3, 1.6], colour_first=True)]
    S += [P("Lift = veces más probable que haya una orden de nivel o rumbo a alguna de las dos aeronaves en los 5 min siguientes que en un par verde "
            "con la misma densidad de tráfico. Los niveles se calibraron con 9 semanas y se evaluaron en 4 semanas no usadas (columna test).", "small")]
    S += [Spacer(1, 6), img(figs["prop"], 15.5)]
    return S, t4, tr4, P, img, table, f, pct


def executive_story(df, figs, st):
    from reportlab.platypus import PageBreak, Spacer
    S, t4, tr4, P, img, table, f, pct = build_story(df, figs, st, executive=True)
    S += [Spacer(1, 4), P("2. Qué hemos hecho y qué respalda la propuesta", "h2")]
    pts = [
        "<b>Fase 15 (órdenes frente a controles, velocidad constante).</b> Las órdenes de nivel casi nunca ocurren con pérdida de separación actual (0,08 %) y "
        "van seguidas de una pérdida prevista con mediana de 6,2 min: el controlador actúa con antelación, como suponíamos. Pero la separación prevista distingue poco una orden "
        "de un control (AUC 0,52): la mayoría de las órdenes son rutinarias.",
        "<b>Fase 16 (todos los pares, 20 min, estandarizado por densidad).</b> Se analizan 21,6 millones de pares-instante con dos predictores (ruta y velocidad constante). "
        "La densidad de tráfico confunde: sin corregirla, el lift se infla.",
        f"<b>Señal.</b> Con m < 1 previsto la probabilidad de orden es {f(t4.loc['rojo'].lift, 1)} veces la de un par verde si el mínimo llega en menos de 5 min y "
        f"{f(t4.loc['naranja'].lift, 1)} veces si llega a 5-20 min; con 1 <= m < 2 es {f(t4.loc['amarillo'].lift, 1)} veces. Los tres escalones están separados (IC95% sin solapar) "
        "y se reproducen en las semanas de test y con ventanas de orden de 2, 5 y 10 min.",
        "<b>La predicción por ruta es necesaria.</b> Con velocidad constante el escalón rojo no supera al naranja (lift 2,1 frente a 2,2). Con ruta sube a 3,1. "
        "La ruta reduce el error vertical P90 a 10 min de 5600 ft a 800 ft; el error horizontal casi no cambia.",
        "<b>Qué no se sostiene.</b> No hay un salto del lift en m = 1,2 ni en m = 1,5: las envolventes 6 NM/1200 ft y 7,5 NM/1500 ft no son fronteras naturales. "
        "Dividir el amarillo en 1,5 da un orden no monótono, y un quinto color (rojo intenso, menos de 2 min) separa de forma marginal y con muy pocos casos.",
        "<b>Horizonte.</b> El lift con m < 1 es casi el mismo con el mínimo a 10-20 min que a 5-10 min: hay que mirar hasta 20 min, no solo 1, 2 y 5.",
    ]
    for t in pts:
        S += [P("- " + t), Spacer(1, 2)]
    S += [Spacer(1, 4), P("3. Qué cambia respecto al diseño anterior", "h2"),
          P("El rojo mantiene la geometría (5 NM y 1000 ft) pero pasa a evaluarse sobre la trayectoria prevista. El amarillo se amplía a 10 NM y 2000 ft (m < 2), "
            "porque el lift cae al nivel de fondo a partir de ahí. El naranja deja de ser una envolvente geométrica intermedia y pasa a significar conflicto previsto a 5-20 min, "
            "que es lo que los datos distinguen del rojo inminente.")]
    S += [Spacer(1, 4), P("4. Cautelas", "h2")]
    for t in ("Estudio observacional: el lift mide asociación con las órdenes, no causalidad ni peligro real.",
              "Las clases son contexto de riesgo, no un predictor de órdenes: incluso en rojo el 58 % de los pares no recibe orden en 5 min.",
              "La orden puede ir a cualquiera de las dos aeronaves y muchas tienen otros motivos; eso diluye el lift.",
              "Densidad controlada en 4 estratos; puede quedar confusión por sector u hora. Rejilla temporal de 10 s."):
        S += [P("- " + t), Spacer(1, 1)]
    S += [Spacer(1, 4), P("5. Próximos pasos", "h2"),
          P("Implementar la clasificación en el entorno con el predictor por ruta y horizonte de 20 min; comprobar si el agente entrenado con estos colores "
            "resuelve más conflictos que con el esquema antiguo; y validar el criterio sobre otro espacio aéreo o con datos de sectorización.")]
    return S


def detailed_story(df, figs, st):
    from reportlab.platypus import PageBreak, Spacer
    S, t4, tr4, P, img, table, f, pct = build_story(df, figs, st, executive=False)
    S += [PageBreak(), P("2. ¿3, 4 o más colores? Comparación de esquemas", "h2"),
          P("Se probaron esquemas de 2 a 5 colores, siempre con niveles disjuntos y ordenados por gravedad, con el predictor por ruta. Un color extra se justifica "
            "si su lift es claramente mayor que el del nivel anterior (el intervalo del cociente excluye 1) y si el orden es monótono. "
            "Resultados en las 4 semanas de test (ventana de orden: 5 min):")]
    q = get(df, predictor="R", W=300, split=SPLIT_TEST)
    rows = [["Esquema", "Nivel", "% pares", "Órdenes 5 min", "Lift (IC95%)", "Cociente con nivel anterior (IC95%)"]]
    for sn, d in q.groupby("scheme", sort=False):
        for _, r in d.iterrows():
            if r.tier == "verde":
                continue
            rr = f"{f(r.ratio_prev)} ({f(r.ratio_lo)}-{f(r.ratio_hi)})" if pd.notna(r.ratio_prev) else "-"
            rows.append([sn, r.tier, f"{100 * r.share_pairs:.2f}".replace(".", ",") + " %", pct(r.order_rate), f"{f(r.lift)} ({f(r.lift_lo)}-{f(r.lift_hi)})", rr])
    S += [Spacer(1, 4), table(rows, [5.2, 2.2, 1.6, 2.0, 3.0, 3.4], fs=7), Spacer(1, 6), img(figs["ratios"], 15)]
    S += [P("Lectura. Con 2 colores se pierde la distinción entre proximidad (lift 1,4) y conflicto previsto (2,4-2,5). Con 3 colores por m se pierde la distinción temporal: "
            "dividir el conflicto previsto en 5-20 min y menos de 5 min aporta un salto de 1,4 veces (IC95% 1,3-1,5), y dividir el amarillo del verde aporta 1,4 veces frente al fondo. "
            "Con 5 colores, partir el amarillo en 1,5 produce un nivel con lift menor que el anterior (no monótono) y partir el rojo en 2 min separa solo "
            "1,17 veces con menos de 1400 pares en 4 semanas. Por tanto 4 es el número que los datos sostienen; 3 es el mínimo defendible; más de 4 no se justifica. "
            "(Se interpreta '24' del mensaje como 2 o 4; si se quería otra cosa, el criterio es el mismo.)")]
    S += [PageBreak(), P("3. Robustez: predictor y ventana de orden", "h2"),
          P("El mismo esquema evaluado con el predictor de velocidad constante y con ventanas de orden de 2, 5 y 10 min (test):"), img(figs["rc"], 16.5)]
    rows = [["Predictor", "Ventana", "amarillo", "naranja", "rojo"]]
    for pn in ("R", "C"):
        for W in (120, 300, 600):
            d = get(df, scheme=PROPOSAL, predictor=pn, W=W, split=SPLIT_TEST).set_index("tier")
            rows.append([pn, f"{W // 60} min" if W >= 60 else f"{W} s"] + [f"{f(d.loc[t].lift)} ({f(d.loc[t].lift_lo)}-{f(d.loc[t].lift_hi)})" for t in ("amarillo", "naranja", "rojo")])
    S += [table(rows, [2.2, 2.2, 3.6, 3.6, 3.6]), Spacer(1, 4),
          P("Con la ruta, los tres niveles son crecientes en las tres ventanas. Con velocidad constante el rojo queda por debajo o igual al naranja: el rojo inminente no se "
            "identifica bien sin saber cuándo se nivelan los aviones.")]
    S += [Spacer(1, 6), P("4. Cómo ponerlo en práctica", "h2")]
    for t in ("Para cada instante y par de aeronaves a menos de 350 NM, predecir 20 min con el predictor por ruta (paso de 10 s) y calcular m_min y el tiempo hasta m_min.",
              "Clasificar: rojo si m_min < 1 y tiempo < 5 min; naranja si m_min < 1 y 5 min <= tiempo <= 20 min; amarillo si 1 <= m_min < 2; verde en otro caso.",
              "Actualizar separation_profiles.json (ENROUTE) con el bloque de reports/proposed_colour_profile.json y adaptar opensky_safety.classify_pair_dynamic.",
              "Mantener como informe los lifts por color para la memoria y repetir la evaluación si cambia el espacio aéreo."):
        S += [P("- " + t), Spacer(1, 1)]
    S += [Spacer(1, 4), P("5. Limitaciones", "h2")]
    for t in ("Estudio observacional: asociación con las órdenes, no con el riesgo real. Una orden puede deberse a otros motivos (rutas directas, meteorología, sectorización) y diluye el lift.",
              "La etiqueta es una orden a cualquiera de las dos aeronaves en la ventana; no se comprueba que sea por ese par.",
              "El lift se mide frente a verde, que incluye pares con m entre 2 y 12 con lift de fondo 1,3; por eso 'amarillo' sube solo 1,4 veces.",
              "Rejilla de 10 s: m cerca de 1 es aproximado. Densidad en 4 estratos. Una sola región (Suecia, 2016-2017).",
              "Las clases describen contexto de riesgo; en rojo el 58 % de los pares no recibe orden en 5 min."):
        S += [P("- " + t), Spacer(1, 1)]
    S += [PageBreak(), P("Anexo A. Fase 15 - separación prevista en órdenes frente a controles", "h2"),
          P("Las páginas siguientes reproducen el informe de la Fase 15 y, a continuación, el de la Fase 16.")]
    return S


def profile_json(t4):
    return {
        "ENROUTE_proposed_v4": {
            "logic": "predicted_min_normalised_separation_with_time_to_minimum",
            "predictor": "route_following (scat predicted trajectories, observed ground speed)",
            "lookahead_seconds": 1200, "grid_seconds": 10,
            "normalised_separation": "m = max(H / 5 NM, V / 1000 ft)",
            "states": {
                "GREEN_NORMAL_OPERATION": {"rule": "otherwise"},
                "YELLOW_PROXIMITY_WARNING": {"rule": "1 <= m_min < 2 within 20 min (inside 10 NM and 2000 ft, no predicted loss)"},
                "ORANGE_HIGH_RISK": {"rule": "m_min < 1 and time_to_min between 300 and 1200 s"},
                "RED_LOSS_OF_SEPARATION": {"rule": "m_min < 1 and time_to_min < 300 s (includes current loss)"},
            },
            "evidence_test_weeks_lift_vs_green_300s": {k: round(float(t4.loc[k].lift), 2) for k in ("amarillo", "naranja", "rojo")},
        }
    }


def merge_pdfs(parts, bookmarks, out):
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    for path, title in zip(parts, bookmarks):
        r = PdfReader(str(path))
        start = len(w.pages)
        for pg in r.pages:
            w.add_page(pg)
        w.add_outline_item(title, start)
    w.add_metadata({"/Title": "Propuesta de clasificación por colores (Fases 15-17)"})
    with open(out, "wb") as fh:
        w.write(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/processed/scat/17_colour_schemes/tables/scheme_tiers.csv")
    ap.add_argument("--fig-dir", default="data/processed/scat/17_colour_schemes/figures")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--stage15-pdf", default="data/processed/scat/15_predicted_separation/reports/STAGE15_REPORT.pdf")
    ap.add_argument("--stage16-pdf", default="reports/STAGE16_REPORT.pdf")
    a = ap.parse_args(argv)
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate

    df = pd.read_csv(a.tables)
    rep = Path(a.reports_dir); rep.mkdir(parents=True, exist_ok=True)
    figs = figures(df, Path(a.fig_dir))
    st = styles()
    kw = dict(pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    ex = rep / "RESUMEN_EJECUTIVO_TUTORES.pdf"
    SimpleDocTemplate(str(ex), title="Resumen ejecutivo - clasificación por colores", **kw).build(executive_story(df, figs, st))
    import tempfile
    prop = Path(tempfile.mkdtemp()) / "propuesta_detallada_tmp.pdf"
    SimpleDocTemplate(str(prop), title="Propuesta detallada", **kw).build(detailed_story(df, figs, st))
    merge_pdfs([prop, Path(a.stage15_pdf), Path(a.stage16_pdf)],
               ["Propuesta final y comparación de esquemas", "Anexo A - Fase 15", "Anexo B - Fase 16"],
               rep / "PROPUESTA_COLORES_DETALLADO.pdf")
    t4 = get(df, scheme=PROPOSAL, predictor="R", W=300, split=SPLIT_TEST).set_index("tier")
    (rep / "proposed_colour_profile.json").write_text(json.dumps(profile_json(t4), indent=2) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
