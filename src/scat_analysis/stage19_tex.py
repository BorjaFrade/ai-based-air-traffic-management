#!/usr/bin/env python3
"""Stage 19: revised executive summary (LaTeX) from the Stage 19 calibration-only tables.

Usage: python3 stage19_tex.py --tables DIR --heat-png PNG --out-dir DIR [--compile]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage18_report as r18  # noqa: E402

fnum, get, FINAL = r18.fnum, r18.get, r18.FINAL


def geometry_figure_v2(fig_dir: Path):
    """Geometry figure with every region painted: green (m>=2 and 1<=m<1.2), yellow (1.2<=m<2), red/orange (m<1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    G, Y, R = "#9fd8a8", "#ffe680", "#f0a08c"
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    ax = axs[0]  # plan view, V = 0
    ax.set_facecolor(G)
    ax.add_patch(Circle((0, 0), 10, facecolor=Y, edgecolor="#b59b00", linewidth=1))
    ax.add_patch(Circle((0, 0), 6, facecolor=G, edgecolor="#4a8f57", linewidth=1, linestyle="--"))
    ax.add_patch(Circle((0, 0), 5, facecolor=R, edgecolor="#b5483a", linewidth=1))
    ax.plot(0, 0, "k^", ms=8); ax.text(.4, -1.2, "A", fontsize=8)
    ax.text(0, 2.2, "rojo/\nnaranja", ha="center", fontsize=7)
    ax.text(5.5, 0, "verde", ha="center", va="center", fontsize=6, rotation=90)
    ax.text(0, 7.8, "amarillo", ha="center", fontsize=7)
    ax.text(-10.8, 10.8, "verde", fontsize=7, va="top")
    ax.text(-3.6, -3.6, "5 NM", fontsize=6, ha="right", va="top"); ax.text(0, -6.15, "6 NM", fontsize=6, ha="center", va="top")
    ax.text(7.2, -7.2, "10 NM", fontsize=6, ha="left", va="top")
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal")
    ax.set_xlabel("distancia horizontal H (NM)"); ax.set_ylabel("distancia horizontal (NM)")
    ax.set_title("(a) Vista en planta (V = 0)", fontsize=9)
    ax = axs[1]  # side view: x = horizontal distance to A, y = altitude difference
    ax.set_facecolor(G)
    ax.add_patch(Rectangle((-10, -2000), 20, 4000, facecolor=Y, edgecolor="#b59b00"))
    ax.add_patch(Rectangle((-6, -1200), 12, 2400, facecolor=G, edgecolor="#4a8f57", linestyle="--"))
    ax.add_patch(Rectangle((-5, -1000), 10, 2000, facecolor=R, edgecolor="#b5483a"))
    ax.plot(0, 0, "k^", ms=8); ax.text(.5, 120, "A", fontsize=8)
    ax.text(0, 450, "rojo/naranja", ha="center", fontsize=7); ax.text(0, 1500, "amarillo", ha="center", fontsize=7)
    ax.text(0, 1085, "verde", ha="center", fontsize=6); ax.text(-10.7, 2300, "verde", fontsize=7, va="top")
    ax.set_xlim(-11, 11); ax.set_ylim(-2400, 2400)
    ax.set_xlabel("distancia horizontal a A (NM)"); ax.set_ylabel("diferencia de altitud V (ft)")
    ax.set_title("(b) Vista lateral: V", fontsize=9)
    ax = axs[2]  # H-V plane
    ax.set_facecolor(G)
    ax.add_patch(Rectangle((0, 0), 10, 2000, facecolor=Y, edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), 6, 1200, facecolor=G, edgecolor="#4a8f57", linestyle="--"))
    ax.add_patch(Rectangle((0, 0), 5, 1000, facecolor=R, edgecolor="#b5483a"))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2400)
    ax.text(2.5, 450, "m<1\nrojo/naranja", fontsize=7, ha="center"); ax.text(5.5, 1090, "verde 1<=m<1,2", fontsize=5.5, ha="center")
    ax.text(8, 1750, "amarillo\n1,2<=m<2", fontsize=7, ha="center"); ax.text(11, 1000, "verde\nm>=2", fontsize=7, ha="center")
    for (h, v, lab) in ((8, 1500, "ej. 1: m=1,6"), (4, 800, "ej. 2: m=0,8"), (3, 1000, "ej. 3: m=1,0")):
        ax.plot(h, v, "ko", ms=4); ax.text(h + .25, v - 120, lab, fontsize=6)
    ax.set_xlabel("H en el punto de mínima separación (NM)"); ax.set_ylabel("V en ese punto (ft)")
    ax.set_title("(c) Plano H-V: colores por m_min", fontsize=9)
    fig.tight_layout(); p = fig_dir / "fig18_geometry_v2.png"; fig.savefig(p, dpi=170); plt.close(fig)
    return p


def thou(n):
    return f"{int(round(n)):,}".replace(",", "\\,")


TEMPLATE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{spanish.ldf}{\usepackage[spanish]{babel}}{}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{array}
\usepackage{float}
\usepackage{enumitem}
\usepackage[margin=1.8cm]{geometry}
\IfFileExists{lmodern.sty}{\usepackage{microtype}}{}
\usepackage[hidelinks]{hyperref}
\emergencystretch=2em

\definecolor{cverde}{HTML}{00A651}
\definecolor{camarillo}{HTML}{FFD400}
\definecolor{cnaranja}{HTML}{FF6F00}
\definecolor{crojo}{HTML}{D7191C}
\newcommand{\mmin}{m_{\min}}
\renewcommand{\arraystretch}{1.25}
\setlist{itemsep=2pt,topsep=3pt}

\title{Resumen ejecutivo: número de colores y clasificación propuesta\\[2pt]\large Versión revisada}
\author{ENROUTE FL290--410, SCAT (13 semanas, 91 días)}
\date{}

\begin{document}
\renewcommand{\tablename}{Cuadro}\renewcommand{\figurename}{Figura}
\maketitle

\noindent Resume las fases 16 a 18 del procesado (las fases 1--15 son preparación y análisis previo; véase la introducción del capítulo de metodología).

\section{Decisión}
Se propone \textbf{4 colores} (verde, amarillo, naranja, rojo), definidos sobre la separación normalizada mínima prevista $\mmin$ en los próximos 20 min, calculada con la trayectoria que sigue la ruta, y sobre el tiempo $\tau$ hasta ese mínimo.

\begin{table}[H]
\centering\footnotesize
\begin{tabular}{l p{4.0cm} >{\raggedleft\arraybackslash}p{3.1cm} >{\raggedleft\arraybackslash}p{1.5cm} >{\raggedleft\arraybackslash}p{2.5cm} >{\raggedleft\arraybackslash}p{1.3cm}}
\toprule
\textbf{Color} & \textbf{Regla (predictor por ruta, 20 min)} & \textbf{Pares-instante calib. / test} & \textbf{Con orden (test)} & \textbf{Lift test (IC95\,\%)} & \textbf{Lift calib.}\\
\midrule
\cellcolor{cverde}\textbf{verde} & resto (incluye $1\le \mmin<1{,}2$ y $\mmin\ge2$) & %VE_N% & %VE_RATE% & 1,00 (referencia) & 1,00\\
\cellcolor{camarillo}\textbf{amarillo} & $1{,}2\le \mmin<2$: dentro de 10 NM y 2000 ft, sin entrar en 6 NM y 1200 ft & %AM_N% & %AM_RATE% & %AM_LIFT% & %AM_TR%\\
\cellcolor{cnaranja}\textbf{naranja} & $\mmin<1$ (pérdida de separación prevista) con el mínimo dentro de 5--20 min & %NA_N% & %NA_RATE% & %NA_LIFT% & %NA_TR%\\
\cellcolor{crojo}\textbf{rojo} & $\mmin<1$ con el mínimo en menos de 5 min (incluye pérdida actual) & %RO_N% & %RO_RATE% & %RO_LIFT% & %RO_TR%\\
\bottomrule
\end{tabular}
\caption{Clasificación propuesta. Un \emph{par-instante} es una pareja de aeronaves evaluada en un instante (cada 120~s). Lift: veces más probable que haya una orden de nivel o rumbo a alguna de las dos aeronaves en los 5 min siguientes que en un par verde con el mismo tráfico. Calibración: 9 semanas (63 días); test: 4 semanas (28 días). Los cortes y $K$ se eligieron solo con las de calibración.}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figs/fig18_tiers.png}
\caption{Tasa de órdenes y lift por color, calibración frente a test (IC95\,\% por bootstrap sobre días).}
\end{figure}

\noindent\textbf{Sobre el número de colores.} Con el horizonte de 5 min, 4 colores alcanzan el %PCT4% de la mejora máxima alcanzable (3 colores: %PCT3%\,\%). Un quinto color (partir el rojo en 2 min) es estadísticamente detectable, pero solo añade %DPP% puntos porcentuales: \textbf{elegir 4 en lugar de 5 es una decisión de parsimonia}, no algo que imponga el test.

\section{Datos y definiciones}
\paragraph{Datos.} 13 semanas de SCAT (tráfico sobre Suecia, de sábado a viernes) entre el 15/10/2016 y el 22/09/2017: 91 días. Calibración (9 semanas, 63 días); test (4 semanas, 28 días: 07/01, 01/04, 24/06 y 16/09/2017), una de cada tres, fijadas por posición antes del análisis.

\paragraph{Par-instante.} Cada 120~s se toma una ``foto'' del tráfico y se evalúa cada pareja de aeronaves simultáneas en FL290--FL410 a menos de 350~NM, con la trayectoria prevista de los 20 min siguientes calculada cada 10~s. La inmensa mayoría nunca pierde la separación ($\mmin\ge1$); lo que se mide es cuánto se acercan. Las que ni se aproximan ($\mmin\ge12$, en la práctica a más de 60~NM) son los \emph{pares lejanos}.

\paragraph{Orden.} Autorización de nivel o de rumbo a cualquiera de las dos aeronaves en los 300~s siguientes (no velocidad; no se distingue entre desvíos del plan y órdenes planificadas).

\paragraph{Separación normalizada.} $H(t)$ y $V(t)$ son la distancia horizontal y la diferencia de altitud entre las dos aeronaves, según la trayectoria prevista (siguiendo la ruta, a la velocidad observada). $m=1$ equivale a 5 NM y 1000 ft; $m=1{,}2$ a 6 NM y 1200 ft; $m=2$ a 10 NM y 2000 ft.
\begin{align}
m(t) &= \max\!\left(\frac{H(t)}{5\ \mathrm{NM}},\ \frac{V(t)}{1000\ \mathrm{ft}}\right), &
\mmin &= \min_{0\le t\le 20\ \mathrm{min}} m(t), &
\tau &= \operatorname*{arg\,min}_{t}\, m(t).
\end{align}

\paragraph{Lift y tráfico similar.} El \textbf{lift} del grupo $g$ es el cociente entre las órdenes observadas $O_g$ y las esperadas $E_g$ si el grupo fuese verde, con el mismo tráfico:
\begin{equation}
\mathrm{lift}_g=\frac{O_g}{E_g},\qquad E_g=\sum_{d=1}^{4} n_{g,d}\,r_{0,d}.
\end{equation}
$n_{g,d}$ son los pares-instante del grupo en el estrato de densidad $d$ y $r_{0,d}$ la tasa de órdenes de los pares \emph{verdes} en ese estrato (cuatro números, no uno). Dos instantes tienen \emph{tráfico similar} si están en el mismo estrato: el número total de aeronaves simultáneas en FL290--FL410 en todo el espacio aéreo de SCAT (no por sector). No se corrige por nivel de vuelo, velocidad, sector ni hora.
\begin{center}\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Estrato (aeronaves) & 0--15 & 15--25 & 25--35 & $>35$\\
\midrule
$r_{0,d}$ (verde, 5 min; semanas de calibración) & %R0_0% & %R0_1% & %R0_2% & %R0_3%\\
\bottomrule
\end{tabular}
\end{center}

\paragraph{Bootstrap por días.} Para estimar la incertidumbre de una cifra se remuestrean los días con reemplazo (se sortean tantos días como hay, con repetición), se recalcula la cifra y se repite 400 veces; el IC95\,\% va del percentil 2,5 al 97,5. Se remuestrean días completos, y no pares-instante sueltos, porque los pares del mismo día y encuentro no son independientes.

\section{Algoritmo para elegir el número de colores $K$}
\paragraph{Unidades.} La malla completa tiene 9 franjas de $\mmin$ ($<1$, 1--1,2, 1,2--1,5, 1,5--2, 2--3, 3--4, 4--6, 6--8, 8--12) $\times$ 4 de $\tau$ ($<2$, 2--5, 5--10, 10--20 min). Para elegir $K$ se usan las 16 celdas con $\mmin<2$ y una celda verde que agrupa $\mmin$ entre 2 y 12 (los pares lejanos quedan fuera). Se agrupan celdas, no pares. En esta parte, $E_c$ se calcula con la tasa de órdenes de los pares lejanos; es solo un factor de escala común que no afecta a la partición ni a las diferencias entre $K$.

\paragraph{Partición para un $K$ dado.} Con $O_c$ y $E_c$ las órdenes observadas y esperadas de la celda, se ordenan las celdas por $O_c/E_c$ y, por programación dinámica, se busca la partición $\mathcal{P}_K$ en $K$ tramos contiguos que maximiza la log-verosimilitud de Poisson (análoga a la segmentación óptima de Fisher/Jenks; exacta, sin inicialización aleatoria):
\begin{equation}
\max_{\mathcal{P}_K}\ \sum_{k=1}^{K} O_k\,\ln\frac{O_k}{E_k}.
\end{equation}

\paragraph{Elección de $K$ por validación cruzada dejando una semana fuera (9 pliegues, solo semanas de calibración).} En cada semana $i$, con la partición y los riesgos $L_k$ ajustados en las otras 8, se mide la verosimilitud fuera de muestra ($O_{ik}$ y $E_{ik}$ son las órdenes observadas y esperadas del nivel $k$ en la semana $i$):
\begin{equation}
G_i(K)=\sum_{k=1}^{K}\Big[\,O_{ik}\ln L_k-\big(L_k-1\big)E_{ik}\Big],\qquad
L_k=\frac{\sum_{j\ne i}O_{jk}}{\sum_{j\ne i}E_{jk}}.
\end{equation}
La ganancia de añadir un color en esa semana y el estadístico $t$ (t de Student emparejado sobre las 9 semanas) son
\begin{equation}
d_i=G_i(K)-G_i(K-1),\qquad
t=\frac{\bar d}{s_d/\sqrt{9}},\qquad
\bar d=\frac{1}{9}\sum_{i=1}^{9}d_i,\qquad
s_d^{2}=\frac{1}{8}\sum_{i=1}^{9}\big(d_i-\bar d\big)^{2}.
\end{equation}
Como la partición libre agrupa celdas sin significado a partir del 4.º nivel, se comparan además esquemas con significado operativo (umbrales de $\mmin$ y de $\tau$) con los mismos pliegues.

\section{Resultados que sostienen el número de colores}
Todo con las 9 semanas de calibración. \textbf{(a)} La mejora fuera de muestra se estanca pronto: con la partición libre, 4 niveles alcanzan el %PCTF4%\,\% de la máxima y 5 niveles el %PCTF5%\,\%; hacia 6--9 niveles la mejora es marginal y 24 niveles no se sostienen. \textbf{(b)} De 1 a 2 y de 2 a 3 colores la mejora es grande y consistente (9/9 pliegues). De 3 a 4 (separar rojo y naranja por $\tau$) es significativa ($t=%T34%$, %F34%/9 pliegues; de %PCT3%\,\% a %PCT4%\,\% de lo alcanzable). El quinto color (rojo en 2 min) es detectable ($t=%T45%$, 9/9) pero solo suma %DPP% puntos; partir el amarillo por $\tau$ no aporta ($t=%TYA%$). \textbf{(c)} El óptimo depende del horizonte: para órdenes a 2 min el paso 3$\to$4 por $\tau$ deja de ser significativo ($t=%T34_120%$) y es el rojo en 2 min el que aporta ($t=%T45_120%$), es decir, 5 colores; con velocidad constante, de 3 a 4 colores solo se ganan %DC% puntos, porque sin la ruta $\tau$ no es fiable. \textbf{(d)} El amarillo debe empezar en $m=1{,}2$ y no en 1,0 ($t=%TY%$, %FY%/9 pliegues).

\begin{figure}[H]
\centering
\includegraphics[width=0.62\textwidth]{figs/fig18_cv_curve.png}
\caption{Mejora fuera de muestra (\% de la alcanzable) según el número de niveles $K$, con validación cruzada por semanas (solo calibración).}
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
\caption{Ganancia de cada color adicional (log-verosimilitud fuera de muestra; ET = error estándar entre pliegues; 9 pliegues).}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figs/fig1_lift_heatmap.png}
\caption{Mapa de riesgo (izquierda: ruta; derecha: velocidad constante; semanas de calibración): lift \emph{respecto al verde} por celda de $\mmin$ (filas) y $\tau$ (columnas). El salto está por debajo de $\mmin=1$ y es mayor con ruta cuando $\tau$ es corto. Entre 1,2 y 2 el lift es moderado. Las celdas con $\mmin$ entre 2 y 12 ya valen 1,1--1,6: el ``fondo'' no es plano, y el amarillo se distingue de él por un margen moderado.}
\end{figure}

\section{Qué significa cada color en distancias}
Los colores dependen del punto de \textbf{mínima separación} de la trayectoria prevista ($\mmin$). Como $m<c$ equivale a $H<5c$ NM y $V<1000c$ ft a la vez, los umbrales son cajas anidadas en el plano $(H,V)$:

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figs/fig18_geometry.png}
\caption{(a) vista en planta (distancia horizontal $H$, con $V=0$); (b) vista lateral (diferencia de altitud $V$); (c) plano $H$--$V$ con las regiones de cada color y tres ejemplos. En verde: $\mmin\ge2$ (fuera de la caja de 10 NM y 2000 ft) y la franja $1\le\mmin<1{,}2$ (entre las cajas de 5 NM/1000 ft y 6 NM/1200 ft).}
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

\end{document}
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--heat-png", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra", default="")  # json with extra numbers (band CV)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args(argv)
    t = Path(a.tables)
    out = Path(a.out_dir)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    sch = pd.read_csv(t / "scheme_tiers.csv")
    inc = pd.read_csv(t / "named_increments.csv")
    nc = pd.read_csv(t / "named_schemes_cv.csv")
    rc = pd.read_csv(t / "region_cv.csv")
    cvs = pd.read_csv(t / "cv_summary.csv")
    cnt = pd.read_csv(t / "counts_by_colour.csv")
    band = pd.read_csv(t / "band_vs_background.csv")
    audit = json.loads((t / "audit.json").read_text())
    extra = json.loads(Path(a.extra).read_text()) if a.extra else {}

    fig_dir = out / "_figs"
    figs = r18.figures(rc, nc, fig_dir)
    tier = r18.tier_figure(sch, fig_dir)
    geo = geometry_figure_v2(fig_dir)
    for src, name in ((tier, "fig18_tiers.png"), (figs["cv"], "fig18_cv_curve.png"), (geo, "fig18_geometry.png"), (Path(a.heat_png), "fig1_lift_heatmap.png")):
        shutil.copy(src, out / "figs" / name)

    t_te = get(sch, scheme=FINAL, predictor="R", W=300, split="test (4 sem)").set_index("tier")
    t_tr = get(sch, scheme=FINAL, predictor="R", W=300, split="calibracion (9 sem)").set_index("tier")
    g = lambda p, W, s: inc[(inc.predictor == p) & (inc.W == W) & (inc.step == s)].iloc[0]  # noqa: E731
    pa = lambda p, W, s: float(nc[(nc.predictor == p) & (nc.W == W) & (nc.scheme == s)].pct_of_achievable.iloc[0])  # noqa: E731
    cnt_ = lambda split, c: cnt[(cnt.split == split) & (cnt.color == c)].iloc[0]  # noqa: E731

    def lift(x):
        return f"{fnum(x.lift)} ({fnum(x.lift_lo)}--{fnum(x.lift_hi)})"

    def rate(x):
        return r18.pct(x.order_rate).replace(" %", "\\,\\%")

    rep = {}
    for key, nm in (("VE", "verde"), ("AM", "amarillo"), ("NA", "naranja"), ("RO", "rojo")):
        c1, c2 = cnt_("calibracion (9 sem)", nm), cnt_("test (4 sem)", nm)
        rep[f"%{key}_N%"] = f"{thou(c1.n_pairs)} / {thou(c2.n_pairs)}"
        rep[f"%{key}_RATE%"] = rate(t_te.loc[nm])
        if nm != "verde":
            rep[f"%{key}_LIFT%"] = lift(t_te.loc[nm])
            rep[f"%{key}_TR%"] = fnum(t_tr.loc[nm].lift)
    rep["%RO_NOORD%"] = fnum(100 * (1 - t_te.loc["rojo"].order_rate), 0)

    s3 = "3 colores: rojo m<1 | amarillo 1.2<=m<2"
    s4 = "4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2"
    s5 = "5 colores: 4 colores + rojo partido en 2 min"
    p3, p4, p5 = pa("R", 300, s3), pa("R", 300, s4), pa("R", 300, s5)
    rep.update({"%PCT3%": fnum(p3, 1), "%PCT4%": fnum(p4, 1), "%DPP%": fnum(p5 - p4, 1)})
    rep["%DC%"] = fnum(pa("C", 300, s4) - pa("C", 300, s3), 1)
    f = cvs[(cvs.predictor == "R") & (cvs.W == 300) & (cvs.family == "free")].sort_values("K")
    g1, gm = f[f.K == 1].mean_gain.iloc[0], f.mean_gain.max()
    pf = lambda k: 100 * (f[f.K == k].mean_gain.iloc[0] - g1) / (gm - g1)  # noqa: E731
    rep.update({"%PCTF4%": fnum(pf(4), 1), "%PCTF5%": fnum(pf(5), 1)})

    s34, s45, sy, sya = "3 -> 4 colores (rojo/naranja por tiempo)", "4 -> 5 colores (rojo en 2 min)", "amarillo desde 1.0 -> desde 1.2", "4 -> 5 colores (amarillo por tiempo)"
    rep.update({"%T34%": fnum(g("R", 300, s34).t, 1), "%F34%": str(int(g("R", 300, s34).folds_better)),
                "%T45%": fnum(g("R", 300, s45).t, 1), "%TYA%": fnum(g("R", 300, sya).t, 1),
                "%TY%": fnum(g("R", 300, sy).t, 1), "%FY%": str(int(g("R", 300, sy).folds_better)),
                "%T34_120%": fnum(g("R", 120, s34).t, 1), "%T45_120%": fnum(g("R", 120, s45).t, 1)})
    names = {"1 -> 2 colores": r"1 $\to$ 2 colores", "2 -> 3 colores": r"2 $\to$ 3 colores",
             s34: r"3 $\to$ 4 (rojo/naranja por $\tau$)", s45: r"4 $\to$ 5 (rojo en 2 min)", sya: r"4 $\to$ 5 (amarillo por $\tau$)"}
    rows = []
    for k, lab in names.items():
        x = g("R", 300, k)
        rows.append(f"{lab} & {fnum(x.gain, 1)} (ET {fnum(x.se, 1)}) & {fnum(x.t, 1)} & {int(x.folds_better)}/9 & {fnum(g('R', 120, k).t, 1)} & {fnum(g('R', 600, k).t, 1)}\\\\")
    rep["%STEPROWS%"] = "\n".join(rows)
    r0 = audit["r0_green_by_density"]
    for i in range(4):
        rep[f"%R0_{i}%"] = r18.pct(r0[i]).replace(" %", "\\,\\%")

    def bl(split, grp, tier):
        d = band[(band.split == split) & (band.group == grp) & (band.tier == tier)].iloc[0]
        return fnum(d.lift)
    cs, ts = "calibracion (9 sem)", "test (4 sem)"
    for key, tr in (("B1", "1-1.2, tau<2 min"), ("B2", "1-1.2, tau 2-5 min"), ("B3", "1-1.2, tau 5-20 min")):
        rep[f"%{key}C%"], rep[f"%{key}T%"] = bl(cs, "banda 1-1.2 por tau", tr), bl(ts, "banda 1-1.2 por tau", tr)
    ga = "banda 1-1.2 (todo tau) y amarillo"
    rep.update({"%BALLC%": bl(cs, ga, "banda 1-1.2"), "%BALLT%": bl(ts, ga, "banda 1-1.2"),
                "%AMC%": bl(cs, ga, "amarillo"), "%AMT%": bl(ts, ga, "amarillo")})
    rep.update({"%BDIFF%": fnum(extra.get("diff", 3.7), 1), "%BT%": fnum(extra.get("t", 2.3), 1), "%BF%": str(extra.get("folds", 6))})
    for k in ("%T34%", "%T45%", "%TYA%", "%TY%", "%T34_120%", "%T45_120%", "%BT%"):  # math mode: avoid the space after a comma
        rep[k] = rep[k].replace(",", "{,}")
    tex = TEMPLATE
    for k, v in rep.items():
        tex = tex.replace(k, v)
    left = [w for w in __import__("re").findall(r"%[A-Z0-9_]+%", tex)]
    if left:
        print("UNREPLACED", left)
    (out / "main.tex").write_text(tex, encoding="utf-8")
    print("wrote", out / "main.tex")
    if a.compile:
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=out, capture_output=True, text=True, errors="replace")
        print(r.stdout[-600:])


if __name__ == "__main__":
    main()
