#!/usr/bin/env python3
"""Stage 17: compare 2/3/4/5-colour schemes using the Stage 16 pair tables.

A scheme = disjoint tiers defined on the (m_min, tau) cells of the route-following predictor
(m = min over the next 20 min of max(H/5 NM, V/1000 ft); tau = time of that minimum).
Green = every pair not in a tier. For each tier we report, in held-out weeks and in the weeks used
to calibrate, the density-standardised lift of "an order is issued to either aircraft in the next W s"
with respect to green, plus the lift ratio between consecutive tiers (does the extra colour add
discrimination?). Uncertainty: bootstrap over days.

Outputs: data/processed/scat/17_colour_schemes/tables/*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage16_report as r16  # noqa: E402

LABEL_W = r16.LABEL_W


def mask(m_lo: int, m_hi: int, t_lo: int, t_hi: int) -> np.ndarray:
    """Cells m_idx in [m_lo, m_hi], tau_idx in [t_lo, t_hi] (inclusive)."""
    mk = np.zeros((10, 5), dtype=bool)
    mk[m_lo:m_hi + 1, t_lo:t_hi + 1] = True
    return mk


# m_idx: 0 <1 | 1 1-1.2 | 2 1.2-1.5 | 3 1.5-2 | 4 2-3 ... ; tau_idx: 0 <2m | 1 2-5 | 2 5-10 | 3 10-20
SCHEMES = {
    "2 colores (alerta: m<2)": {"alerta": mask(0, 3, 0, 3)},
    "2 colores (alerta: m<1)": {"alerta": mask(0, 0, 0, 3)},
    "3 colores (m<2 / m<1)": {"amarillo": mask(1, 3, 0, 3), "rojo": mask(0, 0, 0, 3)},
    "3 colores (m<1 en <5 min / en 5-20 min)": {"naranja": mask(0, 0, 2, 3), "rojo": mask(0, 0, 0, 1)},
    "4 colores (propuesta)": {"amarillo": mask(1, 3, 0, 3), "naranja": mask(0, 0, 2, 3), "rojo": mask(0, 0, 0, 1)},
    "4 colores (final, amarillo 1.2-2)": {"amarillo": mask(2, 3, 0, 3), "naranja": mask(0, 0, 2, 3), "rojo": mask(0, 0, 0, 1)},
    "5 colores (amarillo partido en 1.5)": {"amarillo claro": mask(3, 3, 0, 3), "amarillo": mask(1, 2, 0, 3),
                                           "naranja": mask(0, 0, 2, 3), "rojo": mask(0, 0, 0, 1)},
    "5 colores (rojo partido en 2 min)": {"amarillo": mask(1, 3, 0, 3), "naranja": mask(0, 0, 2, 3),
                                          "rojo": mask(0, 0, 1, 1), "rojo intenso": mask(0, 0, 0, 0)},
}


def tier_stats(T, p, j, tiers: dict, wts):
    """Per tier: n, rate, lift vs green (point + boot) and lift ratio to the previous tier (tiers ordered by severity)."""
    X = T[:, p]
    union = np.zeros((10, 5), dtype=bool)
    for mk in tiers.values():
        union |= mk
    green = ~union
    names = ["verde"] + list(tiers)
    masks = [green] + list(tiers.values())
    n_d = np.stack([(X[..., 0] * mk).sum((2, 3)) for mk in masks])  # tier, day, dens
    l_d = np.stack([(X[..., j] * mk).sum((2, 3)) for mk in masks])

    def lifts(wv):
        ns, ls = np.einsum("bd,tdk->tbk", wv, n_d), np.einsum("bd,tdk->tbk", wv, l_d)  # tier,B,dens
        with np.errstate(divide="ignore", invalid="ignore"):
            rg = np.where(ns[0] > 0, ls[0] / ns[0], 0.0)  # B,dens
            exp = (ns * rg[None]).sum(2)
            return ls.sum(2) / exp, ns.sum(2), ls.sum(2)  # tier,B

    pt, n_pt, l_pt = lifts(np.ones((1, n_d.shape[1])))
    bt, _, _ = lifts(wts)
    rows = []
    for k, nm in enumerate(names):
        lo, hi = np.nanpercentile(bt[k], [2.5, 97.5])
        row = dict(tier=nm, n_pairs=float(n_pt[k, 0]), order_rate=float(l_pt[k, 0] / max(n_pt[k, 0], 1)),
                   lift=float(pt[k, 0]), lift_lo=float(lo), lift_hi=float(hi), share_pairs=float(n_pt[k, 0] / n_pt[:, 0].sum()))
        if k >= 2:
            ratio = bt[k] / bt[k - 1]
            row["ratio_prev"] = float(pt[k, 0] / pt[k - 1, 0])
            row["ratio_lo"], row["ratio_hi"] = (float(v) for v in np.nanpercentile(ratio, [2.5, 97.5]))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="data/processed/scat/16_pair_hazard")
    ap.add_argument("--out-dir", default="data/processed/scat/17_colour_schemes")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args(argv)
    cells, _, _ = r16.load_tables(Path(a.base_dir) / "tables")
    weeks = sorted(cells.week.unique())
    test = weeks[3::3]
    train = [w for w in weeks if w not in test]
    rng = np.random.default_rng(1717)
    out = []
    for split, wk in (("calibracion (9 sem)", train), ("test (4 sem)", test), ("todas (13 sem)", weeks)):
        days, T = r16.cell_tensor(cells, wk)
        wts = r16.boot_weights(len(days), a.boot, rng)
        for pn_i, pn in enumerate(r16.PRED):
            for W in LABEL_W:
                j = 1 + LABEL_W.index(W)
                for sname, tiers in SCHEMES.items():
                    for row in tier_stats(T, pn_i, j, tiers, wts):
                        out.append(dict(split=split, predictor=pn, W=W, scheme=sname, **row))
    df = pd.DataFrame(out)
    od = Path(a.out_dir) / "tables"
    od.mkdir(parents=True, exist_ok=True)
    df.to_csv(od / "scheme_tiers.csv", index=False)
    print(df[(df.predictor == "R") & (df.W == 300) & (df.split != "todas (13 sem)")].round(3).to_string())


if __name__ == "__main__":
    main()
