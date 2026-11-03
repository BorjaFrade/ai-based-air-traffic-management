#!/usr/bin/env python3
"""Stage 18: data-driven choice of the number of colours (K).

Model. Every kept pair-snapshot belongs to one of 36 (m_min, tau) cells (9 m bins x 4 tau bins) or to the
"far" cell (m >= 12). Under a null of no information, cell c has E_c expected labelled pairs (observed order
rate of far pairs in the same density stratum times the pairs in the cell). A K-colour scheme groups the cells
into K tiers; tier k has multiplicative risk lambda_k = O_k / E_k. Score (Poisson log-likelihood gain over the
null lambda = 1):  G = sum_k [ O_k log(lambda_k) - (lambda_k - 1) E_k ].

* 'free'   : cells are sorted by lift and the optimal contiguous grouping in that order is found by dynamic
             programming (upper bound on what any (m, tau) colouring can achieve).
* 'm-only' : tiers are contiguous intervals of the m axis (tau ignored): the simplest interpretable family.

K is chosen by leave-one-week-out cross-validation: partitions and lambdas are fitted on 12 weeks, G is
evaluated on the held-out week. K* = smallest K whose marginal held-out gain over K-1 is not significant
(paired mean / SE across the 13 folds < 2) -- and we also report the K with the maximal mean gain.

Outputs: data/processed/scat/18_optimal_k/tables/*.csv, figures/*.png
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
NM, NT = 9, 4
KMAX = 24


def cell_arrays(T, p, j):
    """T [day,pred,dens,m,tau,4] -> O, E over the 37 cells (36 kept + far), baseline from far pairs by density."""
    X = T[:, p].sum(0)  # dens,m,tau,4
    far_n, far_l = X[:, 9, 4, 0], X[:, 9, 4, j]
    r0 = np.where(far_n > 0, far_l / np.maximum(far_n, 1), 0.0)
    O, E, names = [], [], []
    for m in range(NM):
        for t in range(NT):
            O.append(X[:, m, t, j].sum()); E.append((X[:, m, t, 0] * r0).sum()); names.append((m, t))
    O.append(far_l.sum()); E.append((far_n * r0).sum()); names.append((9, 4))
    return np.array(O), np.array(E), names, r0


def seg_score(O, E):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where((O > 0) & (E > 0), O * np.log(O / E), 0.0)
    return v


def partition_dp(O, E, order, kmax):
    """Optimal contiguous partitions of cells in `order` into k groups, k = 1..kmax. Returns {k: list of index arrays}."""
    n = len(order)
    co, ce = np.concatenate([[0], np.cumsum(O[order])]), np.concatenate([[0], np.cumsum(E[order])])
    S = np.full((n + 1, n + 1), -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            S[i, j] = seg_score(np.array(co[j] - co[i]), np.array(ce[j] - ce[i]))
    kmax = min(kmax, n)
    dp = np.full((kmax + 1, n + 1), -np.inf); bk = np.zeros((kmax + 1, n + 1), dtype=int)
    dp[0, 0] = 0
    for k in range(1, kmax + 1):
        for j in range(k, n + 1):
            cand = dp[k - 1, k - 1:j] + S[k - 1:j, j]
            b = int(np.argmax(cand)); dp[k, j] = cand[b]; bk[k, j] = b + k - 1
    out = {}
    for k in range(1, kmax + 1):
        cuts, j = [], n
        for kk in range(k, 0, -1):
            i = bk[kk, j]; cuts.append(order[i:j]); j = i
        out[k] = cuts[::-1]
    return out


def heldout_gain(parts, O_tr, E_tr, O_te, E_te):
    g = 0.0
    for grp in parts:
        lam = O_tr[grp].sum() / max(E_tr[grp].sum(), 1e-12)
        o, e = O_te[grp].sum(), E_te[grp].sum()
        if lam <= 0:
            g += -1e9 if o > 0 else (e)  # impossible if an event occurs
        else:
            g += o * np.log(lam) - (lam - 1) * e
    return g


def families(O, E, names):
    idx = np.arange(len(O))
    lift = O / np.maximum(E, 1e-12)
    free_order = idx[np.argsort(lift, kind="stable")]
    # m-only: collapse tau -> 10 m groups (9 kept + far); order by m
    return {"free": (O, E, free_order)}


def m_only_arrays(O, E):
    Om = np.array([O[m * NT:(m + 1) * NT].sum() for m in range(NM)] + [O[-1]])
    Em = np.array([E[m * NT:(m + 1) * NT].sum() for m in range(NM)] + [E[-1]])
    return Om, Em, np.arange(NM + 1)


def run(cells, weeks, p, j, kmax=KMAX):
    tens = {w: r16.cell_tensor(cells, [w])[1] for w in weeks}
    rows = []
    full_T = r16.cell_tensor(cells, weeks)[1]
    for fold, w in enumerate(weeks):
        tr = [x for x in weeks if x != w]
        def collapse(T):
            return T.sum(0, keepdims=True)
        Ttr = sum(collapse(tens[x]) for x in tr)
        Tte = collapse(tens[w])
        O_tr, E_tr, names, r0 = cell_arrays(Ttr, p, j)
        # test expectations use TRAIN baseline rates (fair out-of-sample null)
        Xte = Tte[:, p].sum(0)
        O_te = np.array([Xte[:, m, t, j].sum() for m in range(NM) for t in range(NT)] + [Xte[:, 9, 4, j].sum()])
        E_te = np.array([(Xte[:, m, t, 0] * r0).sum() for m in range(NM) for t in range(NT)] + [(Xte[:, 9, 4, 0] * r0).sum()])
        lift_tr = O_tr / np.maximum(E_tr, 1e-12)
        fam = {"free": (O_tr, E_tr, O_te, E_te, np.argsort(lift_tr, kind="stable"))}
        Om_tr, Em_tr, om = m_only_arrays(O_tr, E_tr)
        Om_te, Em_te, _ = m_only_arrays(O_te, E_te)
        fam["m-only"] = (Om_tr, Em_tr, Om_te, Em_te, om)
        for fname, (Ot, Et, Oe, Ee, order) in fam.items():
            parts = partition_dp(Ot, Et, order, kmax if fname == "free" else len(Ot))
            for k, grp in parts.items():
                rows.append(dict(family=fname, fold=fold, week=w, K=k,
                                 gain=heldout_gain(grp, Ot, Et, Oe, Ee),
                                 train_gain=sum(Ot[g].sum() * np.log(max(Ot[g].sum(), 1e-12) / max(Et[g].sum(), 1e-12)) -
                                                (Ot[g].sum() / max(Et[g].sum(), 1e-12) - 1) * Et[g].sum() for g in grp)))
    return pd.DataFrame(rows), full_T


def summarise(cv: pd.DataFrame):
    out = []
    for fam, d in cv.groupby("family"):
        piv = d.pivot(index="fold", columns="K", values="gain")
        ks = list(piv.columns)
        best = piv.mean().idxmax()
        for k in ks:
            row = dict(family=fam, K=k, mean_gain=piv[k].mean(), se_gain=piv[k].std(ddof=1) / np.sqrt(len(piv)))
            if k > 1:
                diff = piv[k] - piv[k - 1]
                row["marg_mean"] = diff.mean(); row["marg_se"] = diff.std(ddof=1) / np.sqrt(len(diff))
                row["marg_t"] = row["marg_mean"] / max(row["marg_se"], 1e-12)
            diffb = piv[best] - piv[k]
            row["loss_vs_best"] = diffb.mean(); row["loss_vs_best_se"] = diffb.std(ddof=1) / np.sqrt(len(diffb))
            row["is_best"] = k == best
            out.append(row)
    s = pd.DataFrame(out)
    return s


def choose_k(s: pd.DataFrame, t_stop=2.0, frac=0.95):
    res = {}
    for fam, d in s.groupby("family"):
        d = d.sort_values("K")
        best = d.loc[d.mean_gain.idxmax()]
        # first K whose next step is not significant
        kstop = None
        for _, r in d.iterrows():
            nxt = d[d.K == r.K + 1]
            if len(nxt) and nxt.iloc[0].marg_t < t_stop:
                kstop = int(r.K); break
        k_one_se = int(d[d.loss_vs_best <= d.loss_vs_best_se].K.min())
        k_95 = int(d[d.mean_gain >= frac * best.mean_gain].K.min())
        res[fam] = dict(k_best=int(best.K), k_first_nonsignificant_step=kstop, k_one_se=k_one_se, k_95pct_of_best=k_95,
                        gain_best=float(best.mean_gain))
    return res


def final_partition(full_T, p, j, K, family):
    O, E, names, r0 = cell_arrays(full_T.sum(0, keepdims=True), p, j)
    if family == "free":
        parts = partition_dp(O, E, np.argsort(O / np.maximum(E, 1e-12), kind="stable"), K)[K]
        tiers = []
        for g in parts:
            tiers.append(dict(lift=float(O[g].sum() / E[g].sum()), cells=[names[i] for i in g], n_labelled=float(O[g].sum())))
        return sorted(tiers, key=lambda t: t["lift"])
    Om, Em, order = m_only_arrays(O, E)
    parts = partition_dp(Om, Em, order, K)[K]
    return [dict(lift=float(Om[g].sum() / Em[g].sum()), m_bins=[int(i) for i in g]) for g in parts]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="data/processed/scat/16_pair_hazard")
    ap.add_argument("--out-dir", default="data/processed/scat/18_optimal_k")
    a = ap.parse_args(argv)
    cells, _, _ = r16.load_tables(Path(a.base_dir) / "tables")
    weeks = sorted(cells.week.unique())
    od = Path(a.out_dir) / "tables"; od.mkdir(parents=True, exist_ok=True)
    allsum, choice, finals = [], {}, {}
    for pn_i, pn in enumerate(r16.PRED):
        for W in LABEL_W:
            j = 1 + LABEL_W.index(W)
            cv, full_T = run(cells, weeks, pn_i, j)
            s = summarise(cv); s["predictor"], s["W"] = pn, W
            allsum.append(s)
            ch = choose_k(s)
            choice[f"{pn}_{W}"] = ch
            if pn == "R" and W == 300:
                cv.to_csv(od / "cv_gain_R_300.csv", index=False)
                for fam in ("free", "m-only"):
                    k = ch[fam]["k_first_nonsignificant_step"] or ch[fam]["k_best"]
                    finals[fam] = dict(K=k, tiers=final_partition(full_T, pn_i, j, k, fam))
                finals["free_by_K"] = {k: final_partition(full_T, pn_i, j, k, "free") for k in range(2, 9)}
                finals["m_only_by_K"] = {k: final_partition(full_T, pn_i, j, k, "m-only") for k in range(2, 7)}
    pd.concat(allsum).to_csv(od / "cv_summary.csv", index=False)
    (od / "k_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    (od / "final_partitions.json").write_text(json.dumps(finals, indent=1, default=str) + "\n")
    print(json.dumps(choice, indent=1))
    s = pd.concat(allsum)
    q = s[(s.predictor == "R") & (s.W == 300)]
    print(q[q.K <= 12][["family", "K", "mean_gain", "se_gain", "marg_mean", "marg_t", "loss_vs_best"]].round(3).to_string())




# ---------------------------------------------------------------------------------------------- structured schemes
def cid(m, t):
    return m * NT + t


def group(pred):
    return np.array([cid(m, t) for m in range(NM) for t in range(NT) if pred(m, t)], dtype=int)


# m_idx: 0 <1 | 1 1-1.2 | 2 1.2-1.5 | 3 1.5-2 | 4+ >=2 ; t: 0 <2m | 1 2-5 | 2 5-10 | 3 10-20
def scheme_groups(name):
    red = lambda m, t: m == 0  # noqa: E731
    amber = lambda m, t: m in (2, 3)  # noqa: E731
    rest_ids = lambda used: np.array([i for i in range(NM * NT + 1) if i not in set(np.concatenate(used))], dtype=int)  # noqa: E731
    S = {}
    r, am = group(red), group(amber)
    S["3: <1 | 1.2-2 | resto"] = [r, am]
    r5a, r5b = group(lambda m, t: m == 0 and t <= 1), group(lambda m, t: m == 0 and t >= 2)
    S["4: <1 en <5min | <1 en 5-20min | 1.2-2 | resto"] = [r5a, r5b, am]
    a5a, a5b = group(lambda m, t: m in (2, 3) and t <= 1), group(lambda m, t: m in (2, 3) and t >= 2)
    S["4: <1 | 1.2-2 en <5min | 1.2-2 en 5-20min | resto"] = [r, a5a, a5b]
    S["5: ambos divididos en 5 min"] = [r5a, r5b, a5a, a5b]
    r2a, r2b, r2c = (group(lambda m, t: m == 0 and t == 0), group(lambda m, t: m == 0 and t == 1), r5b)
    S["5: rojo en <2 | 2-5 | 5-20 min; 1.2-2; resto"] = [r2a, r2b, r2c, am]
    a15a, a15b = group(lambda m, t: m == 2), group(lambda m, t: m == 3)
    S["4: <1 | 1.2-1.5 | 1.5-2 | resto"] = [r, a15a, a15b]
    S["4: <1 | 1-1.2 | 1.2-2 | resto"] = [r, group(lambda m, t: m == 1), am]
    S["3 (propuesta anterior): <1 | 1-2 | resto"] = [r, group(lambda m, t: m in (1, 2, 3))]
    S["4 (propuesta anterior): <1 <5min | <1 5-20min | 1-2 | resto"] = [r5a, r5b, group(lambda m, t: m in (1, 2, 3))]
    S["2: m<2 | resto"] = [group(lambda m, t: m <= 3)]
    S["2: m<1 | resto"] = [r]
    S["2: 1.2<=m<2 o m<1 | resto"] = [group(lambda m, t: m in (0, 2, 3))]
    S["6: 5 anterior + 1.5 partido"] = [r2a, r2b, r2c, a15a, a15b]
    out = {}
    for k, g in S.items():
        used = np.concatenate(g)
        out[k] = g + [np.array([i for i in range(NM * NT + 1) if i not in set(used.tolist())], dtype=int)]
    return out


def structured(cells, weeks, p, j):
    tens = {w: r16.cell_tensor(cells, [w])[1].sum(0, keepdims=True) for w in weeks}
    schemes = scheme_groups(None)
    rows = []
    for fold, w in enumerate(weeks):
        Ttr = sum(tens[x] for x in weeks if x != w)
        O_tr, E_tr, names, r0 = cell_arrays(Ttr, p, j)
        Xte = tens[w][:, p].sum(0)
        O_te = np.array([Xte[:, m, t, j].sum() for m in range(NM) for t in range(NT)] + [Xte[:, 9, 4, j].sum()])
        E_te = np.array([(Xte[:, m, t, 0] * r0).sum() for m in range(NM) for t in range(NT)] + [(Xte[:, 9, 4, 0] * r0).sum()])
        for nm, parts in schemes.items():
            rows.append(dict(scheme=nm, fold=fold, K=len(parts), gain=heldout_gain(parts, O_tr, E_tr, O_te, E_te)))
    return pd.DataFrame(rows)


def structured_main(base_dir="data/processed/scat/16_pair_hazard", out_dir="data/processed/scat/18_optimal_k", weeks=None):
    cells, _, _ = r16.load_tables(Path(base_dir) / "tables")
    weeks = weeks or sorted(cells.week.unique())
    od = Path(out_dir) / "tables"
    res = []
    for pn_i, pn in enumerate(r16.PRED):
        for W in LABEL_W:
            d = structured(cells, weeks, pn_i, 1 + LABEL_W.index(W))
            piv = d.pivot(index="fold", columns="scheme", values="gain")
            ref = "3: <1 | 1.2-2 | resto"
            for sc in piv.columns:
                diff = piv[sc] - piv[ref]
                res.append(dict(predictor=pn, W=W, scheme=sc, K=int(d[d.scheme == sc].K.iloc[0]), mean_gain=piv[sc].mean(),
                                vs3_mean=diff.mean(), vs3_se=diff.std(ddof=1) / np.sqrt(len(diff)),
                                vs3_t=diff.mean() / max(diff.std(ddof=1) / np.sqrt(len(diff)), 1e-12)))
    df = pd.DataFrame(res)
    df.to_csv(od / "structured_schemes.csv", index=False)
    return df


# ---------------------------------------------------------------------------------------------- region DP
def region_arrays(T, p, j, r0=None):
    """Cells of the 'm < 2' region (4 m bins x 4 tau bins = 16) + one aggregated green cell (m 2-12) ; far pairs excluded."""
    X = T[:, p].sum(0)
    if r0 is None:
        far_n, far_l = X[:, 9, 4, 0], X[:, 9, 4, j]
        r0 = np.where(far_n > 0, far_l / np.maximum(far_n, 1), 0.0)
    O, E = [], []
    for m in range(4):
        for t in range(NT):
            O.append(X[:, m, t, j].sum()); E.append((X[:, m, t, 0] * r0).sum())
    O.append(sum(X[:, m, t, j].sum() for m in range(4, NM) for t in range(NT)))
    E.append(sum((X[:, m, t, 0] * r0).sum() for m in range(4, NM) for t in range(NT)))
    return np.array(O), np.array(E), r0


def region_names():
    ml = ["<1", "1-1.2", "1.2-1.5", "1.5-2"]
    tl = ["<2min", "2-5min", "5-10min", "10-20min"]
    return [f"m{m} {t}" for m in ml for t in tl] + ["verde base (m 2-12)"]


def region_cv(cells, weeks, p, j, kmax=17):
    tens = {w: r16.cell_tensor(cells, [w])[1].sum(0, keepdims=True) for w in weeks}
    rows = []
    for fold, w in enumerate(weeks):
        Ttr = sum(tens[x] for x in weeks if x != w)
        Otr, Etr, r0 = region_arrays(Ttr, p, j)
        Ote, Ete, _ = region_arrays(tens[w], p, j, r0)
        parts = partition_dp(Otr, Etr, np.argsort(Otr / np.maximum(Etr, 1e-12), kind="stable"), kmax)
        for k, grp in parts.items():
            rows.append(dict(fold=fold, K=k, gain=heldout_gain(grp, Otr, Etr, Ote, Ete)))
    return pd.DataFrame(rows)


def region_main(base_dir="data/processed/scat/16_pair_hazard", out_dir="data/processed/scat/18_optimal_k", weeks=None):
    cells, _, _ = r16.load_tables(Path(base_dir) / "tables")
    weeks = weeks or sorted(cells.week.unique())
    od = Path(out_dir) / "tables"
    res, finals = [], {}
    for pn_i, pn in enumerate(r16.PRED):
        for W in LABEL_W:
            j = 1 + LABEL_W.index(W)
            cv = region_cv(cells, weeks, pn_i, j)
            piv = cv.pivot(index="fold", columns="K", values="gain")
            best = piv.mean().idxmax()
            for k in piv.columns:
                row = dict(predictor=pn, W=W, K=k, mean_gain=piv[k].mean(), frac_of_best=piv[k].mean() / piv[best].mean())
                if k > 1:
                    d = piv[k] - piv[k - 1]
                    row["marg_mean"], row["marg_t"] = d.mean(), d.mean() / max(d.std(ddof=1) / np.sqrt(len(d)), 1e-12)
                row["is_best"] = k == best
                res.append(row)
            if pn == "R":
                T = r16.cell_tensor(cells, weeks)[1].sum(0, keepdims=True)
                O, E, _ = region_arrays(T, 0, j)
                nm = region_names()
                finals[f"R_{W}"] = {k: [dict(lift=float(O[g].sum() / E[g].sum()), cells=[nm[i] for i in g]) for g in
                                       partition_dp(O, E, np.argsort(O / np.maximum(E, 1e-12), kind="stable"), 8)[k]] for k in range(2, 9)}
    df = pd.DataFrame(res)
    df.to_csv(od / "region_cv.csv", index=False)
    (od / "region_partitions.json").write_text(json.dumps(finals, indent=1) + "\n")
    return df


# ---------------------------------------------------------------------------------------------- clean named schemes on the region
def named_schemes():
    def g(f):
        return np.array([m * NT + t for m in range(4) for t in range(NT) if f(m, t)] , dtype=int)
    red, redA, redB = g(lambda m, t: m == 0), g(lambda m, t: m == 0 and t <= 1), g(lambda m, t: m == 0 and t >= 2)
    red2a, red2b = g(lambda m, t: m == 0 and t == 0), g(lambda m, t: m == 0 and t == 1)
    am12, am1 = g(lambda m, t: m in (2, 3)), g(lambda m, t: m in (1, 2, 3))
    am12a, am12b = g(lambda m, t: m in (2, 3) and t <= 1), g(lambda m, t: m in (2, 3) and t >= 2)
    a15a, a15b = g(lambda m, t: m == 2), g(lambda m, t: m == 3)
    return {
        "2 colores: alerta = m<1": [red],
        "2 colores: alerta = m<2": [g(lambda m, t: True)],
        "3 colores: rojo m<1 | amarillo 1.2<=m<2": [red, am12],
        "3 colores: rojo m<1 | amarillo 1<=m<2": [red, am1],
        "3 colores: rojo m<1 <5min | naranja m<1 5-20min": [redA, redB],
        "4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2": [redA, redB, am12],
        "4 colores: rojo <5min | naranja 5-20min | amarillo 1-2": [redA, redB, am1],
        "4 colores: rojo m<1 | amarillo 1.2-1.5 | amarillo 1.5-2": [red, a15a, a15b],
        "4 colores: rojo m<1 | amarillo 1.2-2 <5min | amarillo 1.2-2 5-20min": [red, am12a, am12b],
        "5 colores: 4 colores + amarillo partido en 5 min": [redA, redB, am12a, am12b],
        "5 colores: 4 colores + rojo partido en 2 min": [red2a, red2b, redB, am12],
        "5 colores: 4 colores + amarillo partido en 1.5": [redA, redB, a15a, a15b],
        "6 colores: rojo en 2 min + amarillo en 1.5": [red2a, red2b, redB, a15a, a15b],
    }


def named_cv(cells, weeks, p, j):
    tens = {w: r16.cell_tensor(cells, [w])[1].sum(0, keepdims=True) for w in weeks}
    sch = named_schemes()
    rows = []
    for fold, w in enumerate(weeks):
        Ttr = sum(tens[x] for x in weeks if x != w)
        Otr, Etr, r0 = region_arrays(Ttr, p, j)
        Ote, Ete, _ = region_arrays(tens[w], p, j, r0)
        allc = np.arange(len(Otr))
        parts_k1 = [allc]
        rows.append(dict(scheme="1 color", fold=fold, K=1, gain=heldout_gain(parts_k1, Otr, Etr, Ote, Ete)))
        for nm, grp in sch.items():
            used = set(np.concatenate(grp).tolist())
            green = np.array([i for i in allc if i not in used], dtype=int)
            rows.append(dict(scheme=nm, fold=fold, K=len(grp) + 1, gain=heldout_gain(grp + [green], Otr, Etr, Ote, Ete)))
    return pd.DataFrame(rows)


def named_main(base_dir="data/processed/scat/16_pair_hazard", out_dir="data/processed/scat/18_optimal_k", weeks=None):
    cells, _, _ = r16.load_tables(Path(base_dir) / "tables")
    weeks = weeks or sorted(cells.week.unique())
    od = Path(out_dir) / "tables"
    res = []
    for pn_i, pn in enumerate(r16.PRED):
        for W in LABEL_W:
            j = 1 + LABEL_W.index(W)
            d = named_cv(cells, weeks, pn_i, j)
            piv = d.pivot(index="fold", columns="scheme", values="gain")
            rc = region_cv(cells, weeks, pn_i, j, kmax=12)
            best = rc.pivot(index="fold", columns="K", values="gain").mean().max()
            g1 = piv["1 color"].mean()
            for sc in piv.columns:
                diff = piv[sc] - piv["1 color"]
                res.append(dict(predictor=pn, W=W, scheme=sc, K=int(d[d.scheme == sc].K.iloc[0]), mean_gain=piv[sc].mean(),
                                gain_over_1=diff.mean(), se=diff.std(ddof=1) / np.sqrt(len(diff)),
                                pct_of_achievable=100 * diff.mean() / (best - g1)))
    df = pd.DataFrame(res)
    df.to_csv(od / "named_schemes_cv.csv", index=False)
    inc = []
    steps = [("1 color", "2 colores: alerta = m<1", "1 -> 2 colores"),
             ("2 colores: alerta = m<1", "3 colores: rojo m<1 | amarillo 1.2<=m<2", "2 -> 3 colores"),
             ("3 colores: rojo m<1 | amarillo 1.2<=m<2", "4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2", "3 -> 4 colores (rojo/naranja por tiempo)"),
             ("3 colores: rojo m<1 | amarillo 1.2<=m<2", "4 colores: rojo m<1 | amarillo 1.2-1.5 | amarillo 1.5-2", "3 -> 4 colores (amarillo partido en 1.5)"),
             ("4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2", "5 colores: 4 colores + rojo partido en 2 min", "4 -> 5 colores (rojo en 2 min)"),
             ("4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2", "5 colores: 4 colores + amarillo partido en 5 min", "4 -> 5 colores (amarillo por tiempo)"),
             ("4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2", "5 colores: 4 colores + amarillo partido en 1.5", "4 -> 5 colores (amarillo en 1.5)"),
             ("4 colores: rojo <5min | naranja 5-20min | amarillo 1-2", "4 colores: rojo <5min | naranja 5-20min | amarillo 1.2-2", "amarillo desde 1.0 -> desde 1.2")]
    for pn_i, pn in enumerate(r16.PRED):
        for W in LABEL_W:
            d = named_cv(cells, weeks, pn_i, 1 + LABEL_W.index(W))
            piv = d.pivot(index="fold", columns="scheme", values="gain")
            for a_, b_, lab in steps:
                diff = piv[b_] - piv[a_]
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                inc.append(dict(predictor=pn, W=W, step=lab, gain=diff.mean(), se=se, t=diff.mean() / max(se, 1e-12),
                                folds_better=int((diff > 0).sum())))
    pd.DataFrame(inc).to_csv(od / "named_increments.csv", index=False)
    return df


if __name__ == "__main__":
    main()
    structured_main()
    region_main()
    named_main()
