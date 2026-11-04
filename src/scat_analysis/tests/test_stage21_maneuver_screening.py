"""Stage 21 observed-geometry and pre-order-information invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage21_maneuver_screening as s21  # noqa: E402


def test_observed_minimum_uses_horizontal_and_vertical_at_the_same_time():
    rules = dict(s21.RULES, horizon_min=1 / 3, step_sec=10, min_coverage=1.0)
    a = np.array([[0, 60, 15, 35000], [10, 60, 15, 35000], [20, 60, 15, 35000]], float)
    b = np.array([[0, 60 + 1 / 60, 15, 37000],
                  [10, 60 + 10 / 60, 15, 35000],
                  [20, 60 + 1 / 60, 15, 37000]], float)
    result, _ = s21.observed_pair(a, b, 0, rules)
    assert result["observed_evaluable"]
    assert result["m_min_observed"] > 1.9
    assert abs(result["m_min_observed"] - max(result["h_at_observed_min_nm"] / 5,
                                                result["v_at_observed_min_ft"] / 1000)) < 1e-9


def test_missing_future_points_are_not_extrapolated():
    rules = dict(s21.RULES, horizon_min=1, step_sec=10, min_coverage=.8)
    a = np.array([[0, 60, 15, 35000], [10, 60, 15, 35000]], float)
    b = np.array([[0, 60.1, 15, 35000], [10, 60.1, 15, 35000]], float)
    result, _ = s21.observed_pair(a, b, 0, rules)
    assert not result["observed_evaluable"]
    assert np.isnan(result["m_min_observed"])


def test_partial_coverage_is_not_complete_evidence_of_absence():
    rules = dict(s21.RULES, horizon_min=5, step_sec=10, min_coverage=.8)
    times = np.array([t for t in range(0, 301, 10) if t not in (50, 60, 70, 80)])
    a = np.column_stack([times, np.full(len(times), 60),
                         np.full(len(times), 15), np.full(len(times), 35000)])
    b = np.column_stack([times, np.full(len(times), 60.1),
                         np.full(len(times), 15), np.full(len(times), 35000)])
    result, _ = s21.observed_pair(a, b, 0, rules)
    assert result["observed_evaluable"]
    assert not result["full_coverage"]


def test_preorder_state_excludes_later_maneuver():
    track = np.array([[-60, 60, 15, 35000], [-10, 60, 15.1, 35000],
                      [5, 60.5, 15.1, 38000]], float)
    state = s21.pre_state(track, 0, s21.RULES)
    assert state is not None
    assert state["alt"] == 35000
    assert state["vz"] == 0
    assert state["vx"] > 0
