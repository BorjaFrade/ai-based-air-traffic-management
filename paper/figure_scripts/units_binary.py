#!/usr/bin/env python3
"""Binary m_min thresholds at pair-instant and episode units on test weeks."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scat_analysis"))
import stage16_report as r16  # noqa: E402
sys.path.insert(0, str(ROOT / "paper/figure_scripts"))
import benchmark as bm  # noqa: E402

OUT = ROOT / "results/benchmarks_units"
THRESHOLDS = (1.0, 1.5, 2.0)
NS = 1_000_000_000


def aggregate(frame: pd.DataFrame, days: list[str], stratum: str, label: str,
              n_strata: int) -> tuple[np.ndarray, np.ndarray]:
    grouped = frame.groupby(["day", stratum], observed=True)[label].agg(["size", "sum"])
    full = pd.MultiIndex.from_product([days, range(n_strata)], names=["day", stratum])
    grouped = grouped.reindex(full, fill_value=0)
    return grouped["size"].to_numpy().reshape(len(days), n_strata), grouped["sum"].to_numpy().reshape(len(days), n_strata)


def pair_arrays(days: list[str]) -> dict[float, tuple[np.ndarray, ...]]:
    tab = ROOT / "data/processed/scat/16_pair_hazard/tables"
    cells = pd.concat((pd.read_parquet(p) for p in sorted(tab.glob("cells_*.parquet"))), ignore_index=True)
    cells = r16.nominal_days_only(cells)
    days_int = set(pd.to_datetime(days).map(lambda x: (x - pd.Timestamp("1970-01-01")).days))
    cells = cells.loc[cells.day.isin(days_int) & cells.predictor.eq("R")].copy()
    day_map = dict(zip(sorted(days_int), days))
    cells["day"] = cells.day.map(day_map)
    out = {}
    for c, maxbin in ((1.0, 0), (1.5, 2), (2.0, 3)):
        arrays = []
        for side in (False, True):
            x = cells.loc[cells.m_bin.le(maxbin) == side]
            z = x.groupby(["day", "dens_bin"], observed=True)[["n_pairs", "n_lab_300"]].sum()
            full = pd.MultiIndex.from_product([days, range(4)], names=["day", "dens_bin"])
            z = z.reindex(full, fill_value=0)
            arrays.extend([z.n_pairs.to_numpy().reshape(len(days), 4),
                           z.n_lab_300.to_numpy().reshape(len(days), 4)])
        out[c] = tuple(arrays)  # negative n/y, positive n/y
    return out


def common_episodes(days: list[str]) -> dict[float, dict[str, tuple[np.ndarray, ...]]]:
    cols = ["split", "day", "m_min", "traffic_stratum", "y_fixed", "y_exposure"]
    ep = pd.read_parquet(ROOT / "results/benchmarks_green_all/episodes.parquet", columns=cols)
    ep = ep.loc[ep.split.eq("test")]
    assert set(ep.day) == set(days)
    result = {}
    for c in THRESHOLDS:
        result[c] = {}
        for label, window in (("y_fixed", "episode_start"), ("y_exposure", "episode_any")):
            neg = ep.loc[ep.m_min.ge(c)]
            pos = ep.loc[ep.m_min.lt(c)]
            result[c][window] = (*aggregate(neg, days, "traffic_stratum", label, 3),
                                 *aggregate(pos, days, "traffic_stratum", label, 3))
    return result


def reconstructed_episodes(days: list[str]) -> dict[float, dict[str, tuple[np.ndarray, ...]]]:
    """Regroup all m<c observations; hold the common-cohort m>=c reference fixed."""
    common = common_episodes(days)
    audit = pd.read_json(ROOT / "data/processed/scat/19_calib_selection/tables/audit.json", typ="series")
    event_index = bm.events()
    frozen = pd.read_parquet(ROOT / "results/benchmarks_green_all/episodes.parquet",
                             columns=["week", "start_ns"])
    starts = {w: np.sort(g.start_ns.to_numpy(np.int64)) for w, g in frozen.groupby("week", sort=False)}
    edges = np.array([2012, 2939])
    chunks = {c: [] for c in THRESHOLDS}
    columns = ["T", "flight_key_a", "flight_key_b", "source_member_a", "source_member_b", "m_min"]
    for week in audit["test"]:
        path = ROOT / "results/stage20/pairs" / f"pairs_{week}.parquet"
        parts = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=columns):
            x = batch.to_pandas()
            x = x.loc[x.m_min.lt(2)].copy()
            if x.empty:
                continue
            wk = pd.Series(week, index=x.index)
            a = bm.entity(wk, x.flight_key_a, x.source_member_a).to_numpy(str)
            b = bm.entity(wk, x.flight_key_b, x.source_member_b).to_numpy(str)
            x["entity1"], x["entity2"] = np.where(a < b, a, b), np.where(a < b, b, a)
            x["time_ns"] = np.rint(x["T"].to_numpy(float) * NS).astype(np.int64)
            parts.append(x[["entity1", "entity2", "time_ns", "m_min"]])
        observations = pd.concat(parts, ignore_index=True)
        for c in THRESHOLDS:
            x = observations.loc[observations.m_min.lt(c)].copy()
            x.sort_values(["entity1", "entity2", "time_ns"], kind="mergesort", inplace=True)
            cut = (x.entity1.ne(x.entity1.shift()) | x.entity2.ne(x.entity2.shift()) |
                   x.time_ns.diff().gt(360 * NS))
            x["run"] = cut.cumsum()
            x["y_exposure"] = bm.any_order(x.entity1.to_numpy(str), x.entity2.to_numpy(str),
                                             x.time_ns.to_numpy(np.int64), event_index, 0, 300)
            g = x.groupby("run", sort=False).agg(entity1=("entity1", "first"),
                                                  entity2=("entity2", "first"),
                                                  start_ns=("time_ns", "first"),
                                                  y_exposure=("y_exposure", "max"))
            g["y_fixed"] = bm.any_order(g.entity1.to_numpy(str), g.entity2.to_numpy(str),
                                         g.start_ns.to_numpy(np.int64), event_index, 0, 300)
            tt = g.start_ns.to_numpy(np.int64)
            ordered = starts[week]
            density = (np.searchsorted(ordered, tt, side="right") -
                       np.searchsorted(ordered, tt - 3600 * NS, side="left"))
            g["traffic_stratum"] = np.searchsorted(edges, density, side="right")
            g["day"] = pd.to_datetime(tt, utc=True).strftime("%Y-%m-%d")
            chunks[c].append(g[["day", "traffic_stratum", "y_fixed", "y_exposure"]])
        print(f"Rebuilt binary runs: {week}", flush=True)
    result = {}
    for c in THRESHOLDS:
        ep = pd.concat(chunks[c], ignore_index=True)
        assert set(ep.day) <= set(days)
        result[c] = {}
        for label, window in (("y_fixed", "episode_start"), ("y_exposure", "episode_any")):
            n0, y0, _, _ = common[c][window]
            np_, yp = aggregate(ep, days, "traffic_stratum", label, 3)
            result[c][window] = n0, y0, np_, yp
    return result


def estimates(arrays: tuple[np.ndarray, ...], weights: np.ndarray) -> dict:
    n0, y0, n1, y1 = arrays
    n0b, y0b, n1b, y1b = (weights @ x for x in arrays)
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline = np.divide(y0b, n0b, out=np.zeros_like(y0b, dtype=float), where=n0b > 0)
        lifts = y1b.sum(axis=1) / (n1b * baseline).sum(axis=1)
        rates0 = y0b.sum(axis=1) / n0b.sum(axis=1)
        rates1 = y1b.sum(axis=1) / n1b.sum(axis=1)
    base = np.divide(y0.sum(axis=0), n0.sum(axis=0),
                     out=np.zeros(n0.shape[1]), where=n0.sum(axis=0) > 0)
    return {"n0": int(n0.sum()), "y0": int(y0.sum()), "n1": int(n1.sum()),
            "y1": int(y1.sum()), "rate0": y0.sum()/n0.sum(), "rate1": y1.sum()/n1.sum(),
            "lift": y1.sum()/(n1.sum(axis=0)*base).sum(),
            "lift_boot": lifts, "rate0_boot": rates0, "rate1_boot": rates1}


def main() -> None:
    start = time.perf_counter()
    days = []
    audit = pd.read_json(ROOT / "data/processed/scat/19_calib_selection/tables/audit.json", typ="series")
    for w in audit["test"]:
        days.extend(pd.date_range(w[:8], periods=7).strftime("%Y-%m-%d").tolist())
    days.sort()
    rng = np.random.default_rng(20261112)
    weights = rng.multinomial(len(days), np.ones(len(days))/len(days), size=400)
    pair = pair_arrays(days)
    common = common_episodes(days)
    rebuilt = reconstructed_episodes(days)
    rows, comparisons = [], []
    for c in THRESHOLDS:
        pp = estimates(pair[c], weights)
        if c == 1:
            assert abs(pp["rate1"] / pp["rate0"] - 2.49) < .01
        variants = {"common": common[c], "rebuilt_positive": rebuilt[c]}
        for mode, windows in variants.items():
            for window, arrays in windows.items():
                ep = estimates(arrays, weights)
                for unit, vals in (("pair_instant", pp), ("episode", ep)):
                    for cls, idx in (("negative", 0), ("positive", 1)):
                        rates = vals[f"rate{idx}_boot"]
                        rows.append(dict(threshold=c, episode_mode=mode, window=window,
                                         unit=unit, binary_class=cls, n=vals[f"n{idx}"],
                                         positive_labels=vals[f"y{idx}"], rate=vals[f"rate{idx}"],
                                         rate_lo=np.quantile(rates, .025), rate_hi=np.quantile(rates, .975),
                                         lift=1.0 if idx == 0 else vals["lift"],
                                         lift_lo=1.0 if idx == 0 else np.quantile(vals["lift_boot"], .025),
                                         lift_hi=1.0 if idx == 0 else np.quantile(vals["lift_boot"], .975)))
                ratio_boot = ep["lift_boot"] / pp["lift_boot"]
                log_boot = np.log(ep["lift_boot"]) - np.log(pp["lift_boot"])
                comparisons.append(dict(threshold=c, episode_mode=mode, window=window,
                                        pair_lift=pp["lift"], episode_lift=ep["lift"],
                                        lift_ratio=ep["lift"]/pp["lift"],
                                        ratio_lo=np.quantile(ratio_boot, .025),
                                        ratio_hi=np.quantile(ratio_boot, .975),
                                        log_lift_difference=np.log(ep["lift"])-np.log(pp["lift"]),
                                        log_diff_lo=np.quantile(log_boot, .025),
                                        log_diff_hi=np.quantile(log_boot, .975)))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "binary_rates_lifts.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "binary_lift_ratios.csv", index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"Runtime: {time.perf_counter()-start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
