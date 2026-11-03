#!/usr/bin/env python3
"""Write the executive summary as a self-contained LaTeX project (main.tex + figs/) from the Stage 17/18 tables.

Usage (from the project root):  python3 src/scat_analysis/stage18_tex.py [--out-dir reports/latex_resumen] [--compile]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage18_report as r18  # noqa: E402

fnum, get, FINAL = r18.fnum, r18.get, r18.FINAL


TEMPLATE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{spanish.ldf}{\usepackage[spanish]{babel}}{}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{array}
\usepackage{float}
\usepackage[margin=1.8cm]{geometry}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}

\definecolor{cverde}{HTML}{00A651}
\definecolor{camarillo}{HTML}{FFD400}
\definecolor{cnaranja}{HTML}{FF6F00}
\definecolor{crojo}{HTML}{D7191C}
\newcommand{\mmin}{m_{\min}}
\renewcommand{\arraystretch}{1.25}

\title{Resumen ejecutivo: número de colores y clasificación propuesta}
\author{ENROUTE FL290--410, SCAT (13 semanas, 107 días)}
\date{}

\begin{document}
\maketitle

\noindent Basado en las fases 15 y 16 y en una selección estadística del número de colores (fase 18).

\section{Decisión}
Se propone \textbf{4 colores} (verde, amarillo, naranja, rojo), definidos sobre la separación normalizada mínima prevista $\mmin$ en los próximos 20 min, calculada con la trayectoria que sigue la ruta, y sobre el tiempo $\tau$ hasta ese mínimo.

\begin{table}[H]
\centering\small
\begin{tabular}{l p{6.0cm} r r r r}
\toprule
\textbf{Color} & \textbf{Regla (predictor por ruta, 20 min)} & \textbf{Órdenes en 5 min} & \textbf{Lift test (IC95\,\%)} & \textbf{Lift calib.} & \textbf{\% de pares}\\
\midrule
\cellcolor{cverde}\textbf{verde} & resto (incluye $1\le \mmin<1{,}2$: cruces legales a $\sim$1000 ft, rutina) & %VERDE_RATE% & 1,00 (referencia) & 1,00 & resto\\
\cellcolor{camarillo}\textbf{amarillo} & $1{,}2\le \mmin<2$: dentro de 10 NM y 2000 ft, sin entrar en 6 NM y 1200 ft & %AM_RATE% & %AM_LIFT% & %AM_TR% & %AM_SH%\\
\cellcolor{cnaranja}\textbf{naranja} & $\mmin<1$ (pérdida de separación prevista) con el mínimo dentro de 5--20 min & %NA_RATE% & %NA_LIFT% & %NA_TR% & %NA_SH%\\
\cellcolor{crojo}\textbf{rojo} & $\mmin<1$ con el mínimo en menos de 5 min (incluye pérdida actual) & %RO_RATE% & %RO_LIFT% & %RO_TR% & %RO_SH%\\
\bottomrule
\end{tabular}
\caption{Clasificación propuesta. Lift: veces más probable que haya una orden de nivel o rumbo a alguna de las dos aeronaves en los 5 min siguientes que en un par verde con la misma densidad de tráfico. Calibración con 9 semanas; evaluación en 4 semanas no usadas.}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figs/fig18_tiers.png}
\caption{Tasa de órdenes y lift por color, calibración frente a test (IC95\,\% por bootstrap sobre días).}
\end{figure}

\section{Definiciones}
$H(t)$ y $V(t)$ son la distancia horizontal y la diferencia de altitud entre las dos aeronaves en el instante $t$, según la trayectoria prevista (siguiendo la ruta, a la velocidad observada, cada 10 s). $m=1$ equivale a 5 NM y 1000 ft; $m=1{,}2$ a 6 NM y 1200 ft; $m=2$ a 10 NM y 2000 ft.
\begin{align}
m(t) &= \max\!\left(\frac{H(t)}{5\ \mathrm{NM}},\ \frac{V(t)}{1000\ \mathrm{ft}}\right), &
\mmin &= \min_{0\le t\le 20\ \mathrm{min}} m(t), &
\tau &= \operatorname*{arg\,min}_{t}\, m(t).
\end{align}
El \textbf{lift} del grupo $g$ es el cociente entre las órdenes observadas $O_g$ y las esperadas $E_g$:
\begin{equation}
\mathrm{lift}_g=\frac{O_g}{E_g},\qquad E_g=\sum_{d} n_{g,d}\,r_{0,d},
\end{equation}
donde $n_{g,d}$ son los pares del grupo en el estrato de densidad $d$ y $r_{0,d}$ la tasa de órdenes (a alguna de las dos aeronaves, en los 5 min siguientes) de los pares lejanos en ese estrato.

\section{Algoritmo para elegir el número de colores $K$}
\paragraph{Unidades.} Cada par de aeronaves en cada instante cae en una de 16 celdas $c$ (4 tramos de $\mmin$ por debajo de 2 $\times$ 4 tramos de $\tau$) o en la base verde ($\mmin$ entre 2 y 12). Se agrupan celdas, no pares.

\paragraph{Partición para un $K$ dado.} Con $O_c$ y $E_c$ las órdenes observadas y esperadas de la celda, se ordenan las celdas por $O_c/E_c$ y, por programación dinámica, se busca la partición $\mathcal{P}_K$ en $K$ tramos contiguos que maximiza la log-verosimilitud de Poisson (análoga a la segmentación óptima de Fisher/Jenks; exacta, sin inicialización aleatoria):
\begin{equation}
\max_{\mathcal{P}_K}\ \sum_{k=1}^{K} O_k\,\ln\frac{O_k}{E_k}.
\end{equation}

\paragraph{Elección de $K$ por validación cruzada dejando una semana fuera (13 pliegues).} En cada semana $i$, con la partición y los riesgos $L_k$ ajustados en las otras 12 semanas, se mide la verosimilitud fuera de muestra ($O_{ik}$ y $E_{ik}$ son las órdenes observadas y esperadas del nivel $k$ en la semana $i$):
\begin{equation}
G_i(K)=\sum_{k=1}^{K}\Big[\,O_{ik}\ln L_k-\big(L_k-1\big)E_{ik}\Big],\qquad
L_k=\frac{\sum_{j\ne i}O_{jk}}{\sum_{j\ne i}E_{jk}}.
\end{equation}
La ganancia de añadir un color en esa semana y el estadístico $t$ (t de Student emparejado sobre las 13 semanas) son
\begin{equation}
d_i=G_i(K)-G_i(K-1),\qquad
t=\frac{\bar d}{s_d/\sqrt{13}},\qquad
\bar d=\frac{1}{13}\sum_{i=1}^{13}d_i,\qquad
s_d^{2}=\frac{1}{12}\sum_{i=1}^{13}\big(d_i-\bar d\big)^{2}.
\end{equation}
$|t|>2$ indica que el color extra mejora de forma consistente en casi todas las semanas, no solo en promedio. Como la partición libre agrupa celdas sin significado a partir del 4.º nivel, se comparan además esquemas con significado operativo (umbrales de $\mmin$ y de $\tau$) con los mismos pliegues.

\section{Resultados que sostienen el número de colores}
\textbf{(a)} La mejora fuera de muestra se estanca hacia 5--8 niveles: no hay base para más (24 incluidos). \textbf{(b)} De 1 a 2 y de 2 a 3 colores la mejora es grande y consistente (13/13 pliegues). De 3 a 4 (separar rojo y naranja por $\tau$) es significativa ($t=%T34%$, %F34%/13 pliegues); el quinto color no aporta a 5 min (rojo a 2 min: $t=%T45%$, solo 0,7\,\% de la mejora). \textbf{(c)} El óptimo depende del horizonte: para órdenes a 2 min sale 5 colores; con velocidad constante bastan 3, porque sin la ruta $\tau$ no es fiable. \textbf{(d)} El amarillo debe empezar en $m=1{,}2$ y no en 1,0 ($t=%TY%$, %FY%/13 pliegues).

\begin{figure}[H]
\centering
\includegraphics[width=0.62\textwidth]{figs/fig18_cv_curve.png}
\caption{Mejora fuera de muestra (\% de la alcanzable) según el número de niveles $K$, con validación cruzada por semanas.}
\end{figure}

\begin{table}[H]
\centering\small
\begin{tabular}{lrrrrr}
\toprule
\textbf{Paso} & \textbf{Ganancia (5 min)} & $t$ & \textbf{Pliegues mejores} & $t$ (2 min) & $t$ (10 min)\\
\midrule
%STEPROWS%
\bottomrule
\end{tabular}
\caption{Ganancia de cada color adicional (log-verosimilitud fuera de muestra; ET = error estándar entre pliegues).}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figs/fig1_lift_heatmap.png}
\caption{Mapa de riesgo (izquierda: ruta; derecha: velocidad constante): lift por celda de $\mmin$ (filas) y $\tau$ (columnas). El salto claro está por debajo de $\mmin=1$ y es mayor con ruta cuando $\tau$ es corto; entre 1,2 y 2 el lift es moderado y casi no depende de $\tau$; la banda 1--1,2 no se distingue del fondo.}
\end{figure}

\noindent\textbf{Cautelas.} Estudio observacional (asociación con las órdenes, no con peligro real); la orden puede ir a cualquiera de las dos aeronaves y muchas tienen otros motivos; los colores son contexto de riesgo, no un predictor de órdenes (incluso en rojo el 58\,\% de los pares no recibe orden en 5 min); la rejilla es de 16 celdas.

\section{Qué significa cada color en distancias}
Los colores dependen del punto de \textbf{mínima separación} de la trayectoria prevista ($\mmin$). Como $m<c$ equivale a $H<5c$ NM y $V<1000c$ ft a la vez, los umbrales son cajas anidadas en el plano $(H,V)$:

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figs/fig18_geometry.png}
\caption{(a) vista en planta (distancia horizontal $H$); (b) vista lateral (diferencia de altitud $V$); (c) plano $H$--$V$ con las regiones de cada color y tres ejemplos.}
\end{figure}

\begin{table}[H]
\centering\small
\begin{tabular}{l l p{6.2cm} p{3.0cm}}
\toprule
\textbf{Color} & \textbf{Condición} & \textbf{Dónde cae el punto de mínima separación} & \textbf{Cuándo ($\tau$)}\\
\midrule
\cellcolor{crojo}\textbf{rojo} & $\mmin<1$ & dentro de la caja de 5 NM y 1000 ft ($H<5$ y $V<1000$ a la vez) & menos de 5 min (incluye pérdida actual)\\
\cellcolor{cnaranja}\textbf{naranja} & $\mmin<1$ & igual que rojo & entre 5 y 20 min\\
\cellcolor{camarillo}\textbf{amarillo} & $1{,}2\le\mmin<2$ & dentro de la caja de 10 NM y 2000 ft pero fuera de la de 6 NM y 1200 ft & cualquier momento de los 20 min\\
\cellcolor{cverde}\textbf{verde} & $\mmin\ge2$, o $1\le\mmin<1{,}2$ & fuera de la caja de 10 NM y 2000 ft; o dentro de la de 6 NM y 1200 ft pero fuera de la de 5 NM y 1000 ft & ---\\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Caso $H<6$ NM y $V<1200$ ft sin llegar a $H<5$ y $V<1000$.} El par respeta el mínimo de separación por poco ($\mmin$ entre 1 y 1,2) y se asocia a \textbf{verde}. Es lo que muestran los datos: su lift (1,05--1,24) es el del fondo, no el del amarillo (1,6). El 94\,\% de estos pares (con el mínimo en menos de 1 min) tiene $V$ entre 900 y 1100 ft, es decir, niveles de vuelo adyacentes con separación vertical planificada, que el controlador trata como rutina. Ejemplos de la figura (c): (1) 8 NM y 1500 ft, $m=1{,}6$, amarillo; (2) 4 NM y 800 ft, $m=0{,}8$, rojo o naranja según $\tau$; (3) 3 NM y 1000 ft exactos, $m=1{,}0$, verde (la desigualdad es estricta). Consecuencia de diseño: un cruce cercano en horizontal entre niveles adyacentes queda en verde; si se quisiera avisar de él, sería una decisión de diseño, no algo que apoyen los datos de órdenes.

\end{document}
"""


def build(tabs, out_dir: Path, fig_dir: Path, heat_png: Path):
    sch, inc = tabs["sch"], tabs["inc"]
    t_te = get(sch, scheme=FINAL, predictor="R", W=300, split="test (4 sem)").set_index("tier")
    t_tr = get(sch, scheme=FINAL, predictor="R", W=300, split="calibracion (9 sem)").set_index("tier")
    g = lambda p, W, s: inc[(inc.predictor == p) & (inc.W == W) & (inc.step == s)].iloc[0]  # noqa: E731
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)
    figs = r18.figures(tabs["rc"], tabs["nc"], fig_dir)
    tier = r18.tier_figure(tabs["sch"], fig_dir)
    geo = r18.geometry_figure(fig_dir)
    for src, name in ((tier, "fig18_tiers.png"), (figs["cv"], "fig18_cv_curve.png"), (geo, "fig18_geometry.png"), (heat_png, "fig1_lift_heatmap.png")):
        shutil.copy(src, out_dir / "figs" / name)

    def lift(t):
        a = t_te.loc[t]
        return f"{fnum(a.lift)} ({fnum(a.lift_lo)}--{fnum(a.lift_hi)})"

    rep = {
        "%VERDE_RATE%": r18.pct(t_te.loc["verde"].order_rate).replace(" %", "\\,\\%"),
        "%AM_RATE%": r18.pct(t_te.loc["amarillo"].order_rate).replace(" %", "\\,\\%"), "%AM_LIFT%": lift("amarillo"), "%AM_TR%": fnum(t_tr.loc["amarillo"].lift),
        "%AM_SH%": fnum(100 * t_te.loc["amarillo"].share_pairs, 2) + "\\,\\%",
        "%NA_RATE%": r18.pct(t_te.loc["naranja"].order_rate).replace(" %", "\\,\\%"), "%NA_LIFT%": lift("naranja"), "%NA_TR%": fnum(t_tr.loc["naranja"].lift),
        "%NA_SH%": fnum(100 * t_te.loc["naranja"].share_pairs, 2) + "\\,\\%",
        "%RO_RATE%": r18.pct(t_te.loc["rojo"].order_rate).replace(" %", "\\,\\%"), "%RO_LIFT%": lift("rojo"), "%RO_TR%": fnum(t_tr.loc["rojo"].lift),
        "%RO_SH%": fnum(100 * t_te.loc["rojo"].share_pairs, 2) + "\\,\\%",
    }
    s34, s45, sy = "3 -> 4 colores (rojo/naranja por tiempo)", "4 -> 5 colores (rojo en 2 min)", "amarillo desde 1.0 -> desde 1.2"
    rep.update({"%T34%": fnum(g("R", 300, s34).t, 1), "%F34%": str(int(g("R", 300, s34).folds_better)),
                "%T45%": fnum(g("R", 300, s45).t, 1), "%TY%": fnum(g("R", 300, sy).t, 1), "%FY%": str(int(g("R", 300, sy).folds_better))})
    names = {"1 -> 2 colores": r"1 $\to$ 2 colores", "2 -> 3 colores": r"2 $\to$ 3 colores",
             s34: r"3 $\to$ 4 (rojo/naranja por $\tau$)", s45: r"4 $\to$ 5 (rojo en 2 min)", "4 -> 5 colores (amarillo por tiempo)": r"4 $\to$ 5 (amarillo por $\tau$)"}
    rows = []
    for k, lab in names.items():
        a = g("R", 300, k)
        rows.append(f"{lab} & {fnum(a.gain, 1)} (ET {fnum(a.se, 1)}) & {fnum(a.t, 1)} & {int(a.folds_better)}/13 & {fnum(g('R', 120, k).t, 1)} & {fnum(g('R', 600, k).t, 1)}\\\\")
    rep["%STEPROWS%"] = "\n".join(rows)
    tex = TEMPLATE
    for k, v in rep.items():
        tex = tex.replace(k, v)
    (out_dir / "main.tex").write_text(tex, encoding="utf-8")
    return out_dir / "main.tex"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables18", default="data/processed/scat/18_optimal_k/tables")
    ap.add_argument("--tables17", default="data/processed/scat/17_colour_schemes/tables/scheme_tiers.csv")
    ap.add_argument("--fig-dir", default="data/processed/scat/18_optimal_k/figures")
    ap.add_argument("--heat-png", default="data/processed/scat/16_pair_hazard/figures/fig1_lift_heatmap.png")
    ap.add_argument("--out-dir", default="reports/latex_resumen")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args(argv)
    t18 = Path(a.tables18)
    tabs = dict(nc=pd.read_csv(t18 / "named_schemes_cv.csv"), inc=pd.read_csv(t18 / "named_increments.csv"),
                rc=pd.read_csv(t18 / "region_cv.csv"), sch=pd.read_csv(a.tables17))
    path = build(tabs, Path(a.out_dir), Path(a.fig_dir), Path(a.heat_png))
    print("wrote", path)
    if a.compile:
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=path.parent, capture_output=True, text=True, errors="replace")
        print(r.stdout[-800:])


if __name__ == "__main__":
    main()
