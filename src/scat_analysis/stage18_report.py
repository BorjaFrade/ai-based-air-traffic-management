#!/usr/bin/env python3
"""Stage 18 report: data-driven number of colours + final proposal.

Reads tables written by stage17_colour_schemes.py and stage18_optimal_k.py and writes
reports/PROPUESTA_COLORES_DETALLADO.pdf (Stage 18 pages + Stage 15 + Stage 16 reports),
reports/RESUMEN_EJECUTIVO_TUTORES.pdf and reports/proposed_colour_profile.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TIER_COL = {"verde": "#00A651", "amarillo": "#FFD400", "naranja": "#FF6F00", "rojo": "#D7191C"}
FINAL = "4 colores (final, amarillo 1.2-2)"


def fnum(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return fnum(100 * x, d) + " %"


def get(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def region_curves(rc: pd.DataFrame):
    out = {}
    for (p, W), d in rc.groupby(["predictor", "W"]):
        d = d.sort_values("K")
        g1 = d[d.K == 1].mean_gain.iloc[0]
        gmax = d.mean_gain.max()
        out[(p, W)] = pd.DataFrame({"K": d.K.values, "pct": 100 * (d.mean_gain.values - g1) / (gmax - g1),
                                    "marg_t": d.marg_t.values})
    return out


def figures(rc, nc, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    cur = region_curves(rc)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sty = {("R", 120): ("#1b9e77", "-o", "ruta, orden en 2 min"), ("R", 300): ("#d95f02", "-o", "ruta, orden en 5 min"),
           ("R", 600): ("#7570b3", "-o", "ruta, orden en 10 min"), ("C", 300): ("#666666", "--s", "vel. constante, 5 min")}
    for key, (c, m, lab) in sty.items():
        d = cur[key]; d = d[d.K <= 12]
        ax.plot(d.K, d.pct, m, color=c, ms=3.5, label=lab)
    ax.axvline(4, color="k", lw=.6, ls=":")
    ax.set_xlabel("número de niveles K (partición óptima de las celdas, validación cruzada por semanas)")
    ax.set_ylabel("% de la mejora alcanzable (fuera de muestra)")
    ax.legend(fontsize=7); ax.set_xticks(range(1, 13))
    fig.tight_layout(); p = fig_dir / "fig18_cv_curve.png"; fig.savefig(p, dpi=150); plt.close(fig); out["cv"] = p
    d = get(nc, predictor="R", W=300)
    d = d[d.scheme != "1 color"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cols = {2: "#bdbdbd", 3: "#9ecae1", 4: "#3182bd", 5: "#fdae6b", 6: "#e6550d"}
    ax.barh(range(len(d)), d.pct_of_achievable, xerr=1.96 * d.se / (d.gain_over_1 / d.pct_of_achievable) , color=[cols[k] for k in d.K],
            edgecolor="k", linewidth=.4, capsize=2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.scheme, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("% de la mejora alcanzable (orden en 5 min, ruta, validación cruzada)")
    fig.tight_layout(); p = fig_dir / "fig18_named.png"; fig.savefig(p, dpi=150); plt.close(fig); out["named"] = p
    return out


def styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    ss = getSampleStyleSheet()
    return dict(body=ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13),
                small=ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.5, leading=9.5),
                cell=ParagraphStyle("c", parent=ss["BodyText"], fontSize=8, leading=10),
                h2=ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16),
                title=ParagraphStyle("t", parent=ss["Title"], fontSize=19, leading=24))


def helpers(st):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, Table, TableStyle
    from PIL import Image as PI

    def P(t, s="body"):
        return Paragraph(t, st[s])

    def img(p, w=16):
        iw, ih = PI.open(p).size
        return Image(str(p), width=w * cm, height=w * cm * ih / iw)

    def table(rows, widths, colour_first=False):
        data = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
        sty = [("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]
        if colour_first:
            for i, r in enumerate(rows[1:], start=1):
                c = TIER_COL.get(str(r[0]).replace("<b>", "").replace("</b>", "").lower())
                if c:
                    sty.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(c)))
        t.setStyle(TableStyle(sty))
        return t
    return P, img, table


def final_table(tiers_test, tiers_train):
    rows = [["Color", "Regla (predictor por ruta, próximos 20 min)", "Órdenes en 5 min", "Lift test (IC95%)", "Lift calibración", "% de pares"]]
    rule = {"verde": "resto (incluye 1 <= m < 1,2: cruces legales a ~1000 ft, rutina)",
            "amarillo": "1,2 <= m < 2: dentro de 10 NM y 2000 ft pero sin entrar en 6 NM y 1200 ft",
            "naranja": "m < 1 (pérdida de separación prevista) con el mínimo dentro de 5-20 min",
            "rojo": "m < 1 con el mínimo en menos de 5 min (incluye pérdida actual)"}
    for t in ("verde", "amarillo", "naranja", "rojo"):
        a, b = tiers_test.loc[t], tiers_train.loc[t]
        lift = "1,00 (referencia)" if t == "verde" else f"{fnum(a.lift)} ({fnum(a.lift_lo)}-{fnum(a.lift_hi)})"
        share = "resto" if t == "verde" else fnum(100 * a.share_pairs, 2) + " %"
        rows.append([f"<b>{t}</b>", rule[t], pct(a.order_rate), lift, fnum(b.lift), share])
    return rows


def decision_text(inc):
    g = lambda p, W, s: inc[(inc.predictor == p) & (inc.W == W) & (inc.step == s)].iloc[0]  # noqa: E731
    return g


def stage18_story(tabs, figs, st):
    from reportlab.platypus import PageBreak, Spacer
    P, img, table = helpers(st)
    nc, inc, rc, sch = tabs["nc"], tabs["inc"], tabs["rc"], tabs["sch"]
    t_te = get(sch, scheme=FINAL, predictor="R", W=300, split="test (4 sem)").set_index("tier")
    t_tr = get(sch, scheme=FINAL, predictor="R", W=300, split="calibracion (9 sem)").set_index("tier")
    g = decision_text(inc)
    cur = region_curves(rc)
    c300 = cur[("R", 300)].set_index("K")
    S = [P("Número óptimo de colores y propuesta final", "title"),
         P("ENROUTE FL290-410, SCAT, 13 semanas. Fase 18 (selección de K) sobre los resultados de las fases 15-16.")]
    S += [P("1. Respuesta", "h2"),
          P("Los datos sostienen <b>4 colores</b> cuando el objetivo es anticipar las órdenes del controlador en los 5 minutos siguientes y se usa la predicción por ruta: "
            "pasar de 3 a 4 colores (separar el conflicto previsto por su inminencia) aporta una mejora significativa fuera de muestra, y un quinto color no la aporta. "
            "Con velocidad constante bastan 3; para anticipar órdenes en los 2 minutos siguientes el óptimo sube a 5. Con 24 niveles no hay ninguna base.")]
    S += [Spacer(1, 4), table(final_table(t_te, t_tr), [1.7, 6.2, 2.0, 3.0, 1.8, 1.6], colour_first=True),
          P("Lift = cuántas veces más probable es una orden de nivel o rumbo a alguna de las dos aeronaves en los 5 min siguientes que en un par verde con la misma densidad de tráfico. "
            "Calibración con 9 semanas y evaluación en 4 semanas no usadas.", "small")]
    S += [PageBreak(), P("2. Cómo se decidió el número de colores", "h2"),
          P("Cada par de aeronaves y cada instante cae en una de 16 celdas (4 tramos de m previsto por debajo de 2 x 4 tramos de tiempo hasta el mínimo) o en la base verde "
            "(m entre 2 y 12). Un esquema de K colores agrupa celdas en K niveles; el riesgo de cada nivel es su tasa de órdenes dividida por la esperada con la densidad de tráfico. "
            "Se mide la verosimilitud de las órdenes en una semana <b>no usada</b> para ajustar (validación cruzada dejando una semana fuera, 13 pliegues) y se compara cada K."),
          P("<b>a) Partición óptima libre.</b> Para cada K se busca la agrupación de celdas que maximiza la verosimilitud en entrenamiento y se evalúa fuera de muestra. "
            f"Con ruta y órdenes en 5 min, la mejora acumulada es {fnum(c300.loc[2].pct, 0)} % con 2 niveles, {fnum(c300.loc[3].pct, 0)} % con 3, {fnum(c300.loc[4].pct, 0)} % con 4, "
            f"{fnum(c300.loc[5].pct, 0)} % con 5, {fnum(c300.loc[6].pct, 0)} % con 6 y {fnum(c300.loc[8].pct, 0)} % con 8, y después se estanca. "
            "Es una cota: los niveles que añade a partir del 4.º agrupan celdas sueltas (por ejemplo un nivel por debajo del verde con cruces a ~1000 ft de separación vertical) y no son colores interpretables."),
          img(figs["cv"], 14.5)]
    S += [P("<b>b) Esquemas interpretables.</b> Se comparan esquemas con significado operativo (rojo, naranja, amarillo) con los mismos pliegues. "
            "La tabla da la ganancia de añadir un color, en unidades de log-verosimilitud, con su error estándar entre pliegues, el estadístico t emparejado y en cuántos de los 13 pliegues mejora.")]
    steps = ["1 -> 2 colores", "2 -> 3 colores", "3 -> 4 colores (rojo/naranja por tiempo)", "3 -> 4 colores (amarillo partido en 1.5)",
             "4 -> 5 colores (rojo en 2 min)", "4 -> 5 colores (amarillo por tiempo)", "4 -> 5 colores (amarillo en 1.5)"]
    rows = [["Paso", "Ganancia (5 min)", "t", "Pliegues mejores", "t con 2 min", "t con 10 min"]]
    for s_ in steps:
        a = g("R", 300, s_)
        rows.append([s_, f"{fnum(a.gain, 1)} (ET {fnum(a.se, 1)})", fnum(a.t, 1), f"{int(a.folds_better)}/13", fnum(g("R", 120, s_).t, 1), fnum(g("R", 600, s_).t, 1)])
    S += [Spacer(1, 3), table(rows, [6.0, 3.2, 1.3, 2.2, 2.1, 2.1]),
          P("Con la ruta y 5 min: separar rojo y naranja por tiempo es significativo (t = " + fnum(g("R", 300, steps[2]).t, 1) + ", 12 de 13 pliegues). "
            "Los demás cuartos colores y los quintos no lo son o aportan menos de 1 unidad frente a 175 acumuladas (rojo a 2 min: significativo pero 0,7 % de la mejora). "
            "Con órdenes a 2 min el quinto color (rojo a menos de 2 min) sí aporta (t = " + fnum(g("R", 120, steps[4]).t, 1) + "). "
            "Con órdenes a 10 min el cuarto es marginal (t = " + fnum(g("R", 600, steps[2]).t, 1) + ").", "small"),
          Spacer(1, 4), img(figs["named"], 14)]
    S += [PageBreak(), P("3. El número óptimo depende del predictor y del horizonte de la orden", "h2")]
    rows = [["Predictor", "Orden en", "K óptimo", "Evidencia"]]
    rows += [["ruta", "2 min", "5", f"rojo partido en 2 min: ganancia {fnum(g('R', 120, steps[4]).gain, 1)}, t = {fnum(g('R', 120, steps[4]).t, 1)}"],
             ["ruta", "5 min", "4", f"3->4 por tiempo: t = {fnum(g('R', 300, steps[2]).t, 1)}; 4->5: n.s. o <1 % de la mejora"],
             ["ruta", "10 min", "4 (3 aceptable)", f"3->4: t = {fnum(g('R', 600, steps[2]).t, 1)}; el 5.º por tiempo del amarillo t = {fnum(g('R', 600, steps[5]).t, 1)} (marginal)"],
             ["vel. constante", "5 y 10 min", "3", f"3->4 por tiempo: t = {fnum(g('C', 300, steps[2]).t, 1)} y {fnum(g('C', 600, steps[2]).t, 1)}"],
             ["vel. constante", "2 min", "5", f"3->4: t = {fnum(g('C', 120, steps[2]).t, 1)}; 4->5: t = {fnum(g('C', 120, steps[4]).t, 1)}"]]
    S += [table(rows, [3.0, 2.2, 2.8, 8.0]),
          P("Lectura: la dimensión temporal (cuánto falta para el mínimo) solo sirve si se sabe cuándo se nivelan los aviones; por eso con velocidad constante basta con 3 niveles. "
            "Elegimos 5 min como horizonte de referencia porque coincide con la ventana usada en las fases 13-16 y con el horizonte táctico del problema.")]
    S += [Spacer(1, 6), P("4. El límite inferior del amarillo: 1,2, no 1,0", "h2"),
          P(f"Mover el límite inferior del amarillo de m = 1,0 a m = 1,2 mejora fuera de muestra ({fnum(g('R', 300, 'amarillo desde 1.0 -> desde 1.2').gain, 1)} unidades, "
            f"t = {fnum(g('R', 300, 'amarillo desde 1.0 -> desde 1.2').t, 1)}, 13 de 13 pliegues). Los pares con 1 <= m < 1,2 se separan ~1000-1200 ft verticalmente al pasar: es el cruce legal habitual "
            "y su lift es el del fondo. Curiosamente, m = 1,2 es la envolvente naranja del diseño original (6 NM, 1200 ft), que queda como frontera inferior del amarillo.")]
    S += [Spacer(1, 6), P("5. ¿Y 24 colores?", "h2"),
          P("No. En la partición libre la ganancia fuera de muestra se estanca entre 8 y 10 niveles y K = 9 no mejora a K = 8 en 5 min; los niveles posteriores a ~5 aportan menos del 1 % de la mejora "
            "y son agrupaciones de celdas sin significado. Además la rejilla (16 celdas) limita lo que se puede afirmar: con una rejilla más fina de m y de tiempo podrían aparecer niveles intermedios, "
            "pero habría que ver si superan el ruido de las órdenes (la mayoría no está ligada a un conflicto).")]
    S += [Spacer(1, 6), P("6. Limitaciones", "h2")]
    for t in ("El criterio es asociación con las órdenes del controlador, no riesgo físico. Muchas órdenes tienen otros motivos y la orden puede ir a cualquiera de los dos aviones.",
              "La ganancia en log-verosimilitud trata los pares como independientes; por eso la significación se evalúa entre semanas (pliegues), no entre pares.",
              "El 'verde' incluye pares con m entre 2 y 12, con lift de fondo 1,3 respecto a pares lejanos (comparten flujos densos); no se usa como color.",
              "Rejilla temporal de 10 s, 16 celdas, densidad en 4 estratos, una sola región y un año (13 semanas).",
              "Las clases son contexto de riesgo, no un predictor de órdenes: incluso en rojo el 58 % de los pares no recibe orden en 5 min."):
        S += [P("- " + t), Spacer(1, 1)]
    S += [PageBreak(), P("Anexo A. Fase 15 - separación prevista en órdenes frente a controles", "h2"),
          P("Las páginas siguientes reproducen el informe de la Fase 15 y, a continuación, el de la Fase 16.")]
    return S, t_te, t_tr, g


def tier_figure(sch, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiers = ["verde", "amarillo", "naranja", "rojo"]
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for sp, off, al in (("calibracion (9 sem)", -.18, .5), ("test (4 sem)", .18, 1.0)):
        d = get(sch, scheme=FINAL, predictor="R", W=300, split=sp).set_index("tier").loc[tiers]
        axs[0].bar(np.arange(4) + off, 100 * d.order_rate, .34, color=[TIER_COL[t] for t in tiers], alpha=al, edgecolor="k", linewidth=.4,
                   label=sp)
        axs[1].errorbar(np.arange(4) + off, d.lift, yerr=[d.lift - d.lift_lo, d.lift_hi - d.lift], fmt="o", color="k", alpha=al, capsize=3, label=sp)
    for ax in axs:
        ax.set_xticks(range(4)); ax.set_xticklabels(tiers)
    axs[0].set_ylabel("% de pares con orden en 5 min"); axs[1].set_ylabel("lift respecto a verde"); axs[1].axhline(1, color="k", lw=.5)
    axs[0].legend(fontsize=7); axs[0].set_title("Tasa de órdenes por color", fontsize=9); axs[1].set_title("Lift con IC95% (bootstrap por días)", fontsize=9)
    fig.tight_layout(); p = fig_dir / "fig18_tiers.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def eq_image(latex, path: Path, size=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, "$" + latex + "$", fontsize=size)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    return path


def equation_images(fig_dir: Path):
    E = {
        "m": r"m(t)=\max\left(\frac{H(t)}{5\ \mathrm{NM}},\ \frac{V(t)}{1000\ \mathrm{ft}}\right),\qquad m_{\min}=\min_{0\leq t\leq 20\ \mathrm{min}} m(t),\qquad \tau=\arg\min_{t}\, m(t)",
        "lift": r"\mathrm{lift}_g=\frac{O_g}{E_g},\qquad E_g=\sum_{d} n_{g,d}\, r_{0,d}",
        "G": r"G_i(K)=\sum_{k=1}^{K}\left[\,O_{ik}\,\ln L_k-\left(L_k-1\right)E_{ik}\right],\qquad L_k=\frac{\sum_{j\neq i}O_{jk}}{\sum_{j\neq i}E_{jk}}",
        "d": r"d_i=G_i(K)-G_i(K-1)",
        "t": r"t=\frac{\bar d}{s_d/\sqrt{13}},\qquad \bar d=\frac{1}{13}\sum_{i=1}^{13}d_i,\qquad s_d^{2}=\frac{1}{12}\sum_{i=1}^{13}\left(d_i-\bar d\right)^{2}",
        "dp": r"\max_{\mathcal{P}_K}\ \sum_{k=1}^{K} O_k\,\ln\frac{O_k}{E_k}\qquad\mathrm{(particiones\ contiguas\ de\ las\ celdas\ ordenadas\ por\ }O_c/E_c\mathrm{)}",
    }
    return {k: eq_image(v, fig_dir / f"eq_{k}.png") for k, v in E.items()}


def geometry_figure(fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    R, Y, G = TIER_COL["rojo"], TIER_COL["amarillo"], TIER_COL["verde"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    ax = axs[0]  # plan view
    for r, c, ls, lab in ((10, Y, "-", "10 NM"), (6, "#888888", "--", "6 NM"), (5, R, "-", "5 NM")):
        ax.add_patch(Circle((0, 0), r, fill=(r in (10, 5)), facecolor=c, alpha=.35 if r != 6 else 0, edgecolor=c, linestyle=ls, linewidth=1.5))
        ax.text(0, r + .25, lab, fontsize=7, ha="center", va="bottom", color="k")
    ax.plot(0, 0, "k^", ms=8); ax.text(.4, -1.2, "A", fontsize=8)
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal"); ax.set_ylabel("distancia horizontal (NM)")
    ax.set_xlabel("distancia horizontal H (NM)"); ax.set_title("(a) Vista en planta: H", fontsize=9)
    ax = axs[1]  # side view
    ax.add_patch(Rectangle((-10, -2000), 20, 4000, facecolor=Y, alpha=.35, edgecolor=Y))
    ax.add_patch(Rectangle((-6, -1200), 12, 2400, facecolor="white", edgecolor="#888888", linestyle="--"))
    ax.add_patch(Rectangle((-5, -1000), 10, 2000, facecolor=R, alpha=.35, edgecolor=R))
    ax.plot(0, 0, "k^", ms=8); ax.text(.5, 120, "A", fontsize=8)
    ax.set_xlim(-11, 11); ax.set_ylim(-2400, 2400)
    ax.set_xlabel("distancia horizontal a A (NM)"); ax.set_ylabel("diferencia de altitud V (ft)")
    ax.set_title("(b) Vista lateral: V", fontsize=9)
    ax.text(0, 1450, "amarillo", ha="center", fontsize=7); ax.text(0, 450, "rojo/naranja", ha="center", fontsize=7)
    ax = axs[2]  # H-V plane
    ax.add_patch(Rectangle((0, 0), 10, 2000, facecolor=Y, alpha=.45, edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), 6, 1200, facecolor=G, alpha=.25, edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), 5, 1000, facecolor=R, alpha=.45, edgecolor="none"))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2400)
    ax.text(2.3, 450, "m<1", fontsize=8, ha="center"); ax.text(5.4, 1060, "verde\n1<=m<1,2", fontsize=6, ha="center")
    ax.text(8, 1650, "amarillo\n1,2<=m<2", fontsize=7, ha="center"); ax.text(10.8, 2250, "verde", fontsize=7, ha="center")
    for (h, v, lab) in ((8, 1500, "ej. 1: m=1,6"), (4, 800, "ej. 2: m=0,8"), (3, 1000, "ej. 3: m=1,0")):
        ax.plot(h, v, "ko", ms=4); ax.text(h + .25, v - 120, lab, fontsize=6)
    ax.set_xlabel("H en el punto de mínima separación (NM)"); ax.set_ylabel("V en ese punto (ft)")
    ax.set_title("(c) Plano H-V: colores por m_min", fontsize=9)
    fig.tight_layout(); p = fig_dir / "fig18_geometry.png"; fig.savefig(p, dpi=170); plt.close(fig)
    return p


def exec_story(tabs, figs, st, t_te, t_tr, g):
    from reportlab.platypus import Spacer
    P, img, table = helpers(st)
    eq = figs["eq"]
    steps = ["3 -> 4 colores (rojo/naranja por tiempo)", "4 -> 5 colores (rojo en 2 min)"]

    def E(key, w):
        return img(eq[key], w)

    S = [P("Resumen ejecutivo: número de colores y clasificación propuesta", "title"),
         P("ENROUTE FL290-410, SCAT (13 semanas, 107 días). Basado en las fases 15 y 16 y en una selección estadística del número de colores (fase 18).")]
    S += [P("1. Decisión", "h2"),
          P("<b>4 colores</b> (verde, amarillo, naranja, rojo), definidos sobre la separación normalizada mínima prevista m_min en los próximos 20 min, "
            "calculada con la trayectoria que sigue la ruta, y sobre el tiempo tau hasta ese mínimo."),
          Spacer(1, 3), table(final_table(t_te, t_tr), [1.7, 6.2, 2.0, 3.0, 1.8, 1.6], colour_first=True),
          P("Lift: veces más probable que haya una orden de nivel o rumbo a alguna de las dos aeronaves en los 5 min siguientes que en un par verde con la misma densidad de tráfico. "
            "Calibración con 9 semanas; evaluación en 4 semanas no usadas.", "small"),
          Spacer(1, 3), img(figs["tiers"], 11.5)]
    S += [P("2. Definiciones", "h2"),
          P("H(t) y V(t) son la distancia horizontal y la diferencia de altitud entre las dos aeronaves en el instante t, según la trayectoria prevista "
            "(siguiendo la ruta, a la velocidad observada, cada 10 s). m = 1 equivale a 5 NM y 1000 ft; m = 1,2 a 6 NM y 1200 ft; m = 2 a 10 NM y 2000 ft."),
          E("m", 14.5),
          P("<b>Lift</b> del grupo g: órdenes observadas O_g entre órdenes esperadas E_g, donde n_(g,d) son los pares del grupo en el estrato de densidad d y r_(0,d) la tasa de órdenes "
            "(a alguna de las dos aeronaves, en los 5 min siguientes) de los pares lejanos en ese estrato."),
          E("lift", 6.2)]
    S += [P("3. Algoritmo para elegir el número de colores K", "h2"),
          P("<b>Unidades.</b> Cada par en cada instante cae en una de 16 celdas c (4 tramos de m_min por debajo de 2 x 4 tramos de tau) o en la base verde (m_min entre 2 y 12). Se agrupan celdas, no pares. "
            "<b>Partición para un K dado.</b> Con O_c y E_c las órdenes observadas y esperadas de la celda, se ordenan las celdas por O_c/E_c y, por programación dinámica, se busca la partición "
            "en K tramos contiguos que maximiza la log-verosimilitud de Poisson (análoga a la segmentación óptima de Fisher/Jenks; exacta, sin inicialización aleatoria):"),
          E("dp", 14.5),
          P("<b>Elección de K por validación cruzada dejando una semana fuera (13 pliegues).</b> En cada semana i, con la partición y los riesgos L_k ajustados en las otras 12 semanas, "
            "se mide la verosimilitud fuera de muestra (O_ik y E_ik son las órdenes observadas y esperadas del nivel k en la semana i):"),
          E("G", 14.5),
          P("La ganancia de añadir un color en esa semana y el estadístico <b>t</b> (t de Student emparejado sobre las 13 semanas) son:"),
          E("d", 6.0), E("t", 13.5),
          P("|t| > 2 indica que el color extra mejora de forma consistente en casi todas las semanas, no solo en promedio. "
            "Como la partición libre agrupa celdas sin significado a partir del 4.º nivel, se comparan además esquemas con significado operativo (umbrales de m_min y de tau) con los mismos pliegues.")]
    S += [P("4. Resultados que sostienen el número de colores", "h2"),
          P(f"<b>(a)</b> La mejora fuera de muestra se estanca hacia 5-8 niveles: no hay base para más (24 incluidos). "
            f"<b>(b)</b> De 1 a 2 y de 2 a 3 colores la mejora es grande y consistente (13/13 pliegues). De 3 a 4 (separar rojo y naranja por tau) es significativa "
            f"(t = {fnum(g('R', 300, steps[0]).t, 1)}, 12/13 pliegues); el quinto color no aporta a 5 min (rojo a 2 min: t = {fnum(g('R', 300, steps[1]).t, 1)}, solo 0,7 % de la mejora). "
            "<b>(c)</b> El óptimo depende del horizonte: para órdenes a 2 min sale 5 colores; con velocidad constante bastan 3, porque sin la ruta tau no es fiable. "
            "<b>(d)</b> El amarillo debe empezar en m = 1,2 y no en 1,0 (t = 13,7, 13/13 pliegues)."),
          Spacer(1, 3), img(figs["cv"], 10)]
    rows = [["Paso", "Ganancia (5 min)", "t", "Pliegues mejores", "t (órdenes a 2 min)", "t (órdenes a 10 min)"]]
    for s_ in ["1 -> 2 colores", "2 -> 3 colores", "3 -> 4 colores (rojo/naranja por tiempo)", "4 -> 5 colores (rojo en 2 min)", "4 -> 5 colores (amarillo por tiempo)"]:
        a = g("R", 300, s_)
        rows.append([s_, f"{fnum(a.gain, 1)} (ET {fnum(a.se, 1)})", fnum(a.t, 1), f"{int(a.folds_better)}/13", fnum(g("R", 120, s_).t, 1), fnum(g("R", 600, s_).t, 1)])
    S += [Spacer(1, 3), table(rows, [6.0, 3.2, 1.3, 2.2, 2.3, 2.3]), Spacer(1, 4),
          P("Mapa de riesgo (izquierda: ruta; derecha: velocidad constante): lift por celda de m_min (filas) y tau (columnas). El salto claro está por debajo de m_min = 1 y es mayor con ruta cuando tau es corto; "
            "entre 1,2 y 2 el lift es moderado y casi no depende de tau; la banda 1-1,2 no se distingue del fondo.", "small"),
          img(figs["heat"], 11.5),
          P("Cautelas: estudio observacional (asociación con las órdenes, no con peligro real); la orden puede ir a cualquiera de las dos aeronaves y muchas tienen otros motivos; "
            "los colores son contexto de riesgo, no un predictor de órdenes (incluso en rojo el 58 % de los pares no recibe orden en 5 min); la rejilla es de 16 celdas.", "small")]
    S += [P("5. Qué significa cada color en distancias", "h2"),
          P("Los colores dependen del punto de <b>mínima separación</b> de la trayectoria prevista (m_min). Como m < c equivale a H < 5c NM y V < 1000c ft a la vez, los umbrales son cajas anidadas en el plano (H, V):"),
          img(figs["geo"], 16.5)]
    rows = [["Color", "Condición", "Dónde cae el punto de mínima separación", "Cuándo (tau)"],
            ["<b>rojo</b>", "m_min < 1", "dentro de la caja de 5 NM y 1000 ft (H < 5 y V < 1000 a la vez)", "menos de 5 min (incluye pérdida actual)"],
            ["<b>naranja</b>", "m_min < 1", "igual que rojo", "entre 5 y 20 min"],
            ["<b>amarillo</b>", "1,2 <= m_min < 2", "dentro de la caja de 10 NM y 2000 ft pero fuera de la de 6 NM y 1200 ft", "cualquier momento de los 20 min"],
            ["<b>verde</b>", "m_min >= 2, o 1 <= m_min < 1,2", "fuera de la caja de 10 NM y 2000 ft; o dentro de la de 6 NM y 1200 ft pero fuera de la de 5 NM y 1000 ft", "-"]]
    S += [table(rows, [1.6, 3.4, 8.0, 4.4], colour_first=True),
          P("<b>Caso H < 6 NM y V < 1200 ft sin llegar a H < 5 y V < 1000:</b> el par respeta el mínimo de separación por poco (m_min entre 1 y 1,2) y se asocia a <b>verde</b>. "
            "Es lo que muestran los datos: su lift (1,05-1,24) es el del fondo, no el del amarillo (1,6). El 94 % de estos pares (con el mínimo en menos de 1 min) tiene V entre 900 y 1100 ft, es decir, "
            "niveles de vuelo adyacentes con separación vertical planificada, que el controlador trata como rutina. Ejemplos de la figura (c): 1) 8 NM y 1500 ft, m = 1,6, amarillo; "
            "2) 4 NM y 800 ft, m = 0,8, rojo o naranja según tau; 3) 3 NM y 1000 ft exactos, m = 1,0, verde (la desigualdad es estricta). "
            "Consecuencia de diseño: un cruce cercano en horizontal entre niveles adyacentes queda en verde; si se quisiera avisar de él, sería una decisión de diseño, no algo que apoyen los datos de órdenes.", "small")]
    return S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables18", default="data/processed/scat/18_optimal_k/tables")
    ap.add_argument("--tables17", default="data/processed/scat/17_colour_schemes/tables/scheme_tiers.csv")
    ap.add_argument("--fig-dir", default="data/processed/scat/18_optimal_k/figures")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--heat-png", default="data/processed/scat/16_pair_hazard/figures/fig1_lift_heatmap.png")
    ap.add_argument("--stage15-pdf", default="data/processed/scat/15_predicted_separation/reports/STAGE15_REPORT.pdf")
    ap.add_argument("--stage16-pdf", default="reports/STAGE16_REPORT.pdf")
    a = ap.parse_args(argv)
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate

    t18 = Path(a.tables18)
    tabs = dict(nc=pd.read_csv(t18 / "named_schemes_cv.csv"), inc=pd.read_csv(t18 / "named_increments.csv"),
                rc=pd.read_csv(t18 / "region_cv.csv"), sch=pd.read_csv(a.tables17))
    figs = figures(tabs["rc"], tabs["nc"], Path(a.fig_dir))
    figs["tiers"] = tier_figure(tabs["sch"], Path(a.fig_dir))
    figs["heat"] = Path(a.heat_png)
    figs["eq"] = equation_images(Path(a.fig_dir))
    figs["geo"] = geometry_figure(Path(a.fig_dir))
    st = styles()
    rep = Path(a.reports_dir); rep.mkdir(parents=True, exist_ok=True)
    kw = dict(pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story, t_te, t_tr, g = stage18_story(tabs, figs, st)
    tmp = Path(tempfile.mkdtemp())
    SimpleDocTemplate(str(tmp / "p18.pdf"), title="Fase 18", **kw).build(story)
    SimpleDocTemplate(str(rep / "RESUMEN_EJECUTIVO_TUTORES.pdf"), title="Resumen ejecutivo", **kw).build(exec_story(tabs, figs, st, t_te, t_tr, g))
    w = PdfWriter()
    for path, title in ((tmp / "p18.pdf", "Número óptimo de colores y propuesta final"), (Path(a.stage15_pdf), "Anexo A - Fase 15"),
                        (Path(a.stage16_pdf), "Anexo B - Fase 16")):
        start = len(w.pages)
        for pg in PdfReader(str(path)).pages:
            w.add_page(pg)
        w.add_outline_item(title, start)
    with open(rep / "PROPUESTA_COLORES_DETALLADO.pdf", "wb") as fh:
        w.write(fh)
    (rep / "proposed_colour_profile.json").write_text(json.dumps({
        "ENROUTE_proposed_v4": {
            "logic": "predicted_min_normalised_separation_with_time_to_minimum",
            "predictor": "route_following (scat predicted trajectories, observed ground speed)",
            "lookahead_seconds": 1200, "grid_seconds": 10, "normalised_separation": "m = max(H / 5 NM, V / 1000 ft)",
            "states": {
                "GREEN_NORMAL_OPERATION": {"rule": "otherwise (includes 1 <= m_min < 1.2: routine 1000 ft crossings)"},
                "YELLOW_PROXIMITY_WARNING": {"rule": "1.2 <= m_min < 2 within 20 min (inside 10 NM and 2000 ft, not inside 6 NM and 1200 ft)"},
                "ORANGE_HIGH_RISK": {"rule": "m_min < 1 and time_to_min between 300 and 1200 s"},
                "RED_LOSS_OF_SEPARATION": {"rule": "m_min < 1 and time_to_min < 300 s (includes current loss)"}},
            "evidence_test_weeks_lift_vs_green_300s": {k: round(float(t_te.loc[k].lift), 2) for k in ("amarillo", "naranja", "rojo")}}}, indent=2) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
