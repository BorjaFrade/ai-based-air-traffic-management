#!/usr/bin/env python3
"""Stage 19: re-run of the colour selection with ONLY the 9 calibration weeks (+ audit of counts and bands).

Why. In Stage 16-18 the cut-points (m_min = 1.2, tau = 5 min) and the number of colours K were chosen with tables
and a cross-validation that included the 4 test weeks. Here:
  * border days (outside the nominal SCAT week) are dropped (91 days instead of 107);
  * the whole selection (free DP, region DP, named schemes, increments) is repeated with leave-one-week-out CV
    over the 9 calibration weeks only (9 folds);
  * the colour tiers are then evaluated (Stage 17 tables, green = reference) on calibration and on the 4 test weeks;
  * absolute counts per colour, lifts of the 1-1.2 band relative to green, and a heat map relative to green.

Outputs: data/processed/scat/19_calib_selection/{tables,figures}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage16_report as r16  # noqa: E402
import stage17_colour_schemes as s17  # noqa: E402
import stage18_optimal_k as s18  # noqa: E402

BASE = "data/processed/scat/16_pair_hazard"
OUT = Path("data/processed/scat/19_calib_selection")
LABEL_W = r16.LABEL_W
FINAL = "4 colores (final, amarillo 1.2-2)"
M_LAB = ["<1", "1-1.2", "1.2-1.5", "1.5-2", "2-3", "3-4", "4-6", "6-8", "8-12"]
T_LAB = ["<2 min", "2-5 min", "5-10 min", "10-20 min"]


def green_mask(tiers: dict) -> np.ndarray:
    u = np.zeros((10, 5), dtype=bool)
    for m in tiers.values():
        u |= m
    return ~u


def heat_vs_green(T, p, j, tiers):
    """Lift per (m, tau) cell relative to green (density-stratified). T [day,pred,dens,m,tau,4]."""
    X = T[:, p].sum(0)  # dens,m,tau,4
    g = green_mask(tiers)
    n_g = (X[..., 0] * g).sum((1, 2))
    l_g = (X[..., j] * g).sum((1, 2))
    r0 = np.where(n_g > 0, l_g / np.maximum(n_g, 1), 0.0)
    M = np.full((9, 4), np.nan)
    N = np.zeros((9, 4))
    for m in range(9):
        for t in range(4):
            n = X[:, m, t, 0]
            e = (n * r0).sum()
            M[m, t] = X[:, m, t, j].sum() / e if e > 0 else np.nan
            N[m, t] = n.sum()
    return M, N, r0


def heat_figure(Ms, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, (pn, M) in zip(axs, Ms.items()):
        im = ax.imshow(M, origin="lower", cmap="RdYlBu_r", vmin=0.6, vmax=3.2, aspect="auto")
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                ax.text(k, i, f"{M[i, k]:.2f}".replace(".", ","), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(4)); ax.set_xticklabels(T_LAB, fontsize=8)
        ax.set_yticks(range(9)); ax.set_yticklabels(M_LAB, fontsize=8)
        ax.set_title("Prediccion: " + ("ruta (R)" if pn == "R" else "vel. constante (C)"), fontsize=9)
        ax.set_xlabel("tiempo hasta la separacion minima")
    axs[0].set_ylabel("separacion normalizada minima prevista m")
    fig.colorbar(im, ax=axs, label="lift respecto al verde (orden en 5 min)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cells, _, _ = r16.load_tables(Path(BASE) / "tables")
    weeks = sorted(cells.week.unique())
    test = weeks[3::3]
    calib = [w for w in weeks if w not in test]
    od = OUT / "tables"
    od.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    audit = dict(weeks=weeks, calib=calib, test=test)
    days_all, _ = r16.cell_tensor(cells, weeks)
    days_cal, _ = r16.cell_tensor(cells, calib)
    days_te, _ = r16.cell_tensor(cells, test)
    audit["n_days"] = dict(all=len(days_all), calib=len(days_cal), test=len(days_te))

    # --- selection with calibration weeks only ------------------------------------------------------
    allsum, choice = [], {}
    for pn_i, pn in enumerate(r16.PRED):
        for W in LABEL_W:
            j = 1 + LABEL_W.index(W)
            cv, _ = s18.run(cells, calib, pn_i, j)
            s = s18.summarise(cv); s["predictor"], s["W"] = pn, W
            allsum.append(s)
            choice[f"{pn}_{W}"] = s18.choose_k(s)
    pd.concat(allsum).to_csv(od / "cv_summary.csv", index=False)
    (od / "k_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    s18.region_main(BASE, str(OUT), weeks=calib)
    s18.named_main(BASE, str(OUT), weeks=calib)
    s18.structured_main(BASE, str(OUT), weeks=calib)

    # --- tiers of the (frozen) scheme on calibration / test / all (Stage 17 machinery, green = reference) ----
    s17.main(["--base-dir", BASE, "--out-dir", str(OUT)])
    sch = pd.read_csv(od / "scheme_tiers.csv")

    # --- absolute counts per colour (route predictor, W=300) -------------------------------------------
    tiers = s17.SCHEMES[FINAL]
    rng = np.random.default_rng(1919)
    rows, band = [], []
    for split, wk in (("calibracion (9 sem)", calib), ("test (4 sem)", test), ("todas (13 sem)", weeks)):
        days, T = r16.cell_tensor(cells, wk)
        wts = r16.boot_weights(len(days), 400, rng)
        X = T[:, 0]  # route predictor
        for nm, mk in [("verde", green_mask(tiers))] + list(tiers.items()):
            rows.append(dict(split=split, color=nm, n_pairs=float((X[..., 0] * mk).sum()),
                             n_ordered=float((X[..., 2] * mk).sum())))
        # band 1-1.2 relative to the background (m >= 2, i.e. green without the band) and amarillo for comparison
        for label, tdict in (("banda 1-1.2 (todo tau) y amarillo", {"banda 1-1.2": s17.mask(1, 1, 0, 3)}),
                             ("banda 1-1.2 por tau", {"1-1.2, tau<2 min": s17.mask(1, 1, 0, 0), "1-1.2, tau 2-5 min": s17.mask(1, 1, 1, 1),
                                                      "1-1.2, tau 5-20 min": s17.mask(1, 1, 2, 3)})):
            full = dict(tdict); full.update(tiers)  # fondo = m>=2: verde sin la banda; los tiers finales se excluyen de la referencia
            for r in s17.tier_stats(T, 0, 2, full, wts)[1:]:
                band.append(dict(split=split, group=label, **r))
    pd.DataFrame(rows).to_csv(od / "counts_by_colour.csv", index=False)
    pd.DataFrame(band).to_csv(od / "band_vs_background.csv", index=False)

    # --- density-stratum reference rates (green) and heat map vs green (calibration weeks) ---------------
    _, Tc = r16.cell_tensor(cells, calib)
    Ms = {}
    for pn_i, pn in enumerate(r16.PRED):
        M, N, r0 = heat_vs_green(Tc, pn_i, 2, tiers)
        Ms[pn] = M
        if pn == "R":
            audit["r0_green_by_density"] = [float(x) for x in r0]
            pd.DataFrame(M, index=M_LAB, columns=T_LAB).round(3).to_csv(od / "heat_vs_green_R.csv")
            pd.DataFrame(N, index=M_LAB, columns=T_LAB).to_csv(od / "heat_pairs_R.csv")
    heat_figure(Ms, OUT / "figures" / "fig1_lift_heatmap_vs_green.png")
    (od / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
