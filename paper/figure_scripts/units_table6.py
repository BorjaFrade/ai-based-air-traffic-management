#!/usr/bin/env python3
"""Reproduce MDPI Table 6 point estimates from frozen Stage 16 and episode data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scat_analysis"))
import stage16_report as r16  # noqa: E402
import stage17_colour_schemes as r17  # noqa: E402

OUT = ROOT / "results/benchmarks_units"
COLORS = ("verde", "amarillo", "naranja", "rojo")
EXPECTED_RATE = np.array([[13.5, 20.7, 30.1, 42.2],
                          [15.8, 21.5, 27.7, 30.9],
                          [34.4, 30.5, 41.0, 45.9]])
EXPECTED_LIFT = np.array([[1.00, 1.53, 2.23, 3.11],
                          [1.00, 1.37, 1.75, 1.96],
                          [1.00, 0.89, 1.19, 1.33]])


def pair_instant_column(test_weeks: list[str]) -> pd.DataFrame:
    """Stage 17's exact density-stratified colour lift on nominal test days."""
    tab = ROOT / "data/processed/scat/16_pair_hazard/tables"
    cells = pd.concat((pd.read_parquet(p) for p in sorted(tab.glob("cells_*.parquet"))),
                      ignore_index=True)
    cells = r16.nominal_days_only(cells)
    _, tensor = r16.cell_tensor(cells, test_weeks)
    tiers = r17.SCHEMES["4 colores (final, amarillo 1.2-2)"]
    weights = r16.boot_weights(tensor.shape[0], 400, np.random.default_rng(20261112))
    rows = r17.tier_stats(tensor, 0, 2, tiers, weights)
    result = pd.DataFrame(rows).rename(columns={"tier": "color", "n_pairs": "n",
                                                "order_rate": "rate"})
    result["window"] = "pair_instant"
    result["positive"] = np.rint(result.n * result.rate).astype(int)
    return result[["window", "color", "n", "positive", "rate", "lift", "lift_lo", "lift_hi"]]


def episode_columns() -> pd.DataFrame:
    """Same episode traffic strata and both frozen 300-s labels as Table 6."""
    path = ROOT / "results/benchmarks_green_all/episodes.parquet"
    cols = ["split", "highest_color", "traffic_stratum", "y_fixed", "y_exposure"]
    ep = pd.read_parquet(path, columns=cols)
    ep = ep.loc[ep.split.eq("test")]
    rows = []
    for label, window in (("y_fixed", "episode_start"), ("y_exposure", "episode_any")):
        base = ep.loc[ep.highest_color.eq("verde")].groupby("traffic_stratum")[label].mean()
        for color in COLORS:
            group = ep.loc[ep.highest_color.eq(color)]
            n, positive = len(group), int(group[label].sum())
            expected = group.traffic_stratum.value_counts().mul(base).sum()
            rows.append(dict(window=window, color=color, n=n, positive=positive,
                             rate=positive / n, lift=positive / expected))
    return pd.DataFrame(rows)


def main() -> None:
    audit = pd.read_json(ROOT / "data/processed/scat/19_calib_selection/tables/audit.json", typ="series")
    frames = [pair_instant_column(audit["test"]), episode_columns()]
    result = pd.concat(frames, ignore_index=True)
    rate = (100 * result.rate.to_numpy()).reshape(3, 4)
    lift = result.lift.to_numpy().reshape(3, 4)
    if not (np.array_equal(np.round(rate, 1), EXPECTED_RATE)
            and np.array_equal(np.round(lift, 2), EXPECTED_LIFT)):
        print(result.to_string(index=False))
        raise AssertionError("MDPI Table 6 does not reproduce; stop before robustness analyses")
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "table6_reproduction.csv", index=False)
    print(result.to_string(index=False))
    print("PASS: all 24 published rates and lifts reproduce at displayed precision")


if __name__ == "__main__":
    main()
