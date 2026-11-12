#!/usr/bin/env python3
"""Re-score the same episodes using all extracted heading/level orders."""
from __future__ import annotations

import pandas as pd

import benchmark as b


def main():
    base = b.ROOT / "results/benchmarks"
    out = b.ROOT / "results/benchmarks_broad_orders"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(base / "episodes.parquet")
    index = b.events(broad=True)
    a, c = d.entity1.to_numpy(str), d.entity2.to_numpy(str)
    t = d.start_ns.to_numpy("int64")
    d["y_fixed"] = b.any_order(a, c, t, index, 0, 300)
    d["prior_600"] = b.any_order(a, c, t, index, 600, 0)
    d.to_parquet(out / "episodes_broad_labels.parquet", index=False)
    b.score(d, out)
    summary = (d.loc[d.split.eq("test")].groupby("highest_color").y_fixed
               .agg(["size", "mean"]).reset_index())
    summary.to_csv(out / "test_color_rates.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
