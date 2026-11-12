#!/usr/bin/env python3
"""Re-score prespecified order-window sensitivities on the frozen episode cohort."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import benchmark as b


def main():
    base = b.ROOT / "results/benchmarks"
    original = pd.read_parquet(base / "episodes.parquet")
    index = b.events()
    a = original.entity1.to_numpy(str)
    c = original.entity2.to_numpy(str)
    t = original.start_ns.to_numpy("int64")
    rows = []
    for later, prior in ((120, 600), (600, 600), (300, 300)):
        cohort = original.copy()
        cohort["y_fixed"] = b.any_order(a, c, t, index, 0, later)
        cohort["prior_600"] = b.any_order(a, c, t, index, prior, 0)
        out = b.ROOT / f"results/benchmarks_window_{later}_prior_{prior}"
        out.mkdir(parents=True, exist_ok=True)
        b.score(cohort, out)
        d = pd.read_csv(out / "metrics.csv")
        selected = json.loads((out / "decision.json").read_text())["episode"]
        m0 = d.loc[d.weight.eq("episode") & d.model.eq("M0")].iloc[0]
        m4 = d.loc[d.weight.eq("episode") & d.model.eq("M4")].iloc[0]
        rates = cohort.loc[cohort.split.eq("test")].groupby("highest_color").y_fixed.mean()
        rows.append({"later_sec": later, "prior_sec": prior,
                     "m0_brier": m0.brier, "m4_brier": m4.brier,
                     "m4_delta_brier": m4.delta_brier_vs_M0,
                     "monotone_color_rates": bool(rates.reindex(b.COLORS).is_monotonic_increasing),
                     "selected_continuous_model": selected["selected_continuous_model"],
                     "continuous_preferred": selected["continuous_preferred"]})
    pd.DataFrame(rows).to_csv(base / "window_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
