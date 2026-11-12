#!/usr/bin/env python3
"""Score the no-limit green cohort with the calibration-frozen primary models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import benchmark as b


def main():
    out = b.ROOT / "results/benchmarks_green_all"
    d = pd.read_parquet(out / "episodes.parquet")
    train, test = d.loc[d.split.eq("calibration")], d.loc[d.split.eq("test")]
    # M4 and its C=10 regularisation were selected on the approved primary
    # calibration cohort. This large sensitivity does not reselect on test.
    p0 = b._fit_predict(train, test, "M0", "episode")
    p4 = b._fit_predict(train, test, "M4", "episode", c=10.0)
    y = test.y_fixed.to_numpy(int)
    rows = [{"model": name, "brier": brier_score_loss(y, p)}
            for name, p in (("M0", p0), ("M4", p4))]
    pd.DataFrame(rows).to_csv(out / "frozen_model_scores.csv", index=False)
    rates = test.groupby("highest_color").y_fixed.agg(["size", "mean"]).reset_index()
    rates.to_csv(out / "test_color_rates.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(rates.to_string(index=False))


if __name__ == "__main__":
    main()
