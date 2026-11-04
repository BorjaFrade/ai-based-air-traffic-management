"""Stage 20 temporal and attribution invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scat_stage_io import to_ns  # noqa: E402
import stage20_audit_clearances as s20  # noqa: E402


def test_integral_and_fractional_epoch_seconds_share_nanosecond_unit():
    pair = to_ns(pd.Series([1_476_490_800.0]), "pair")
    order = to_ns(pd.Series([1_476_490_920.5]), "order")
    assert pair[0] == 1_476_490_800_000_000_000
    assert order[0] - pair[0] == 120_500_000_000


def test_next_order_window_is_open_at_snapshot_and_closed_at_300_seconds():
    base = 1_476_490_800_000_000_000
    key = "week\x1fflight\x1fmember"
    index = {key: (base + np.array([0, 300, 301], dtype=np.int64) * 1_000_000_000,
                   np.array([0, 1, 2], dtype=np.int32))}
    ids, _ = s20.nearest(np.array([key]), np.array([base]), index, 300, True)
    assert ids.tolist() == [1]
    counts = np.zeros((4, 3), dtype=np.int64)
    s20.count_all_events(np.array([key]), np.array([base]),
                         np.array(["rojo"]), index, 300, counts)
    assert counts[3].tolist() == [0, 1, 0]


def test_each_eligible_order_is_counted_once_for_a_pair_snapshot():
    base = 1_476_490_800_000_000_000
    a, b = "week\x1fa\x1fm", "week\x1fb\x1fm"
    index = {a: (base + np.array([120, 180], dtype=np.int64) * 1_000_000_000,
                 np.array([0, 1], dtype=np.int32)),
             b: (base + np.array([240], dtype=np.int64) * 1_000_000_000,
                 np.array([2], dtype=np.int32))}
    counts = np.zeros((4, 3), dtype=np.int64)
    for key in (a, b):
        s20.count_all_events(np.array([key]), np.array([base]),
                             np.array(["rojo"]), index, 300, counts)
    assert counts[3].tolist() == [1, 1, 1]
