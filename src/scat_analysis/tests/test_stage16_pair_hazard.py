"""Stage 16 invariants: projection, route following, no-future prediction lookup, pair features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "acquisition"))
import stage16_pair_hazard as s16  # noqa: E402

RULES = s16.s15.load_rules()
GRID = np.arange(0, 1201, 10, dtype=float)


def test_projection_preserves_local_distances():
    x1, y1 = s16.to_plane(np.array([59.0]), np.array([15.0]))
    x2, y2 = s16.to_plane(np.array([60.0]), np.array([15.0]))
    assert np.hypot(x2 - x1, y2 - y1)[0] == pytest.approx(60.0, rel=0.005)
    # 1 degree of longitude at 60N is 30 NM
    x3, y3 = s16.to_plane(np.array([60.0]), np.array([16.0]))
    x4, y4 = s16.to_plane(np.array([60.0]), np.array([17.0]))
    assert np.hypot(x4 - x3, y4 - y3)[0] == pytest.approx(30.0, rel=0.01)
    # the centre maps to the origin
    cx, cy = s16.to_plane(np.array([62.0]), np.array([16.0]))
    assert abs(cx[0]) < 1e-9 and abs(cy[0]) < 1e-9


def test_plane_direction_follows_track():
    ux, uy = s16.plane_direction(np.array([62.0]), np.array([16.0]), np.array([90.0]))
    assert ux[0] == pytest.approx(1.0, abs=1e-3) and abs(uy[0]) < 1e-3
    ux, uy = s16.plane_direction(np.array([62.0]), np.array([16.0]), np.array([0.0]))
    assert uy[0] == pytest.approx(1.0, abs=1e-3)


def _route(x0, y0, z0, ux, uy, speed, pts):
    k = s16.MAX_PATH_FIXES
    px, py, pz = (np.full((1, k), np.nan) for _ in range(3))
    for i, (x, y, z) in enumerate(pts):
        px[0, i], py[0, i], pz[0, i] = x, y, z
    return s16.route_positions(np.array([x0]), np.array([y0]), np.array([z0]), np.array([ux]), np.array([uy]),
                               np.array([speed]), px, py, pz, GRID)


def test_route_following_turns_at_the_fix_and_follows_altitude_profile():
    speed = 480.0 / 3600.0  # 8 NM/min
    # east 80 NM then north; altitude 35000 -> 31000 between fixes
    xs, ys, zs = _route(0, 0, 35000.0, 1, 0, speed, [(80, 0, 35000.0), (80, 80, 31000.0)])
    t10 = np.searchsorted(GRID, 600)  # 600 s = 80 NM travelled = the first fix
    assert xs[0, t10] == pytest.approx(80, abs=1e-6) and ys[0, t10] == pytest.approx(0, abs=1e-6)
    t15 = np.searchsorted(GRID, 900)  # 40 NM into the second leg
    assert xs[0, t15] == pytest.approx(80, abs=1e-6) and ys[0, t15] == pytest.approx(40, abs=1e-6)
    assert zs[0, t15] == pytest.approx(33000.0, abs=1)
    # past the last fix: keeps going north at the same speed
    assert ys[0, -1] == pytest.approx(speed * 1200 - 80, abs=1e-6)


def test_route_without_fixes_equals_constant_velocity_horizontally():
    speed = 450.0 / 3600.0
    xs, ys, zs = _route(10, 5, 35000.0, 0.6, 0.8, speed, [])
    xc, yc, _ = s16.constant_velocity(np.array([10.0]), np.array([5.0]), np.array([35000.0]),
                                      np.array([0.6 * speed]), np.array([0.8 * speed]), np.array([0.0]), GRID)
    assert np.allclose(xs, xc) and np.allclose(ys, yc) and np.allclose(zs, 35000.0)


def test_route_and_cv_agree_on_a_straight_route():
    speed = 450.0 / 3600.0
    xs, ys, _ = _route(0, 0, 35000.0, 1, 0, speed, [(50, 0, 35000.0), (200, 0, 35000.0), (400, 0, 35000.0)])
    xc, yc, _ = s16.constant_velocity(np.array([0.0]), np.array([0.0]), np.array([35000.0]),
                                      np.array([speed]), np.array([0.0]), np.array([0.0]), GRID)
    assert np.allclose(xs, xc) and np.allclose(ys, yc)


def test_prediction_lookup_never_uses_a_future_update():
    index = pd.DataFrame({"code": [1, 1, 1, 2], "prediction_epoch": [100.0, 200.0, 400.0, 150.0], "rg": 0, "off": 0})
    t0 = 0.0
    index["key"] = index.code * s16.KEY_SCALE + (index.prediction_epoch - t0)
    index = index.sort_values("key").reset_index(drop=True)
    codes, T = np.array([1, 1, 1, 2, 2, 3]), np.array([99.0, 250.0, 400.0, 149.0, 160.0, 500.0])
    pos = s16.latest_prediction(index, codes, T, t0, max_age=900.0)
    got = [(-1 if p < 0 else float(index.prediction_epoch[p])) for p in pos]
    assert got == [-1, 200.0, 400.0, -1, 150.0, -1]
    assert all(index.prediction_epoch[p] <= t for p, t in zip(pos, T) if p >= 0)
    # a stale update is rejected
    assert s16.latest_prediction(index, np.array([1]), np.array([250.0]), t0, max_age=10.0)[0] == -1


def test_pair_features_head_on_and_diverging():
    speed = 480.0 / 3600.0
    n = 3
    x0 = np.array([0.0, 100.0, 0.0])
    y0 = np.array([0.0, 0.0, 300.0])
    vx = np.array([speed, -speed, speed])
    X = x0[:, None] + vx[:, None] * GRID[None, :]
    Y = np.repeat(y0[:, None], len(GRID), 1)
    Z = np.full_like(X, 35000.0)
    i0, i1 = np.triu_indices(n, 1)
    keep, feat = s16.pair_features(X, Y, Z, i0, i1, GRID, RULES, 6, 12.0)
    assert keep[0] and not keep[1] and not keep[2]  # only the head-on pair is kept (the others stay >= 60 NM apart)
    # 10 s sampling at 16 NM/min leaves at most 1.33 NM to the true minimum (m <= 0.27)
    assert feat["m_min"][0] < 0.3 and feat["state_1200"][0] == 3
    # gap 100 NM closing at 16 NM/min reaches 5 NM at 356 s
    assert 350 <= feat["t_first_red"][0] <= 370
    assert feat["state_60"][0] == 0  # not RED within the first minute


def test_empty_pair_selection_is_safe():
    X = np.array([[0.0] * len(GRID), [500.0] * len(GRID)])
    keep, feat = s16.pair_features(X, np.zeros_like(X), np.full_like(X, 35000.0), np.array([0]), np.array([1]),
                                   GRID, RULES, 6, 12.0)
    assert not keep.any() and len(feat["m_min"]) == 0 and len(feat["state_60"]) == 0


def test_order_labels_look_only_forward():
    orders = {"A": np.array([100.0, 500.0])}
    assert s16.order_labels("A", 90.0, orders) == [True, True, True]       # 100 within 120 s
    assert s16.order_labels("A", 100.0, orders) == [False, False, True]    # the order at T itself is not future
    assert s16.order_labels("A", 101.0, orders) == [False, False, True]    # next order at 500 s
    assert s16.order_labels("A", 501.0, orders) == [False, False, False]   # nothing after the last order
    assert s16.order_labels("B", 0.0, orders) == [False, False, False]


def test_parse_route_and_build_path_skip_passed_fixes():
    route = ('[{"lat":62.0,"lon":16.0,"afl_value":350.0,"afl_unit":"F","eto":"2017-01-01T00:00:00"},'
             '{"lat":62.0,"lon":17.0,"afl_value":350.0,"afl_unit":"F","eto":"2017-01-01T00:02:00"},'
             '{"lat":62.0,"lon":19.0,"afl_value":320.0,"afl_unit":"F","eto":"2017-01-01T00:07:00"}]')
    rec = s16.parse_route(route)
    assert rec is not None and np.allclose(rec[3], [35000.0, 35000.0, 32000.0])
    t_start = float(np.datetime64("2017-01-01T00:01:00").astype("datetime64[s]").astype(np.int64))
    x0, y0 = s16.to_plane(np.array([62.0]), np.array([16.6]))
    ux, uy = s16.plane_direction(np.array([62.0]), np.array([16.6]), np.array([90.0]))
    px, py, pz = s16.build_path(rec, t_start, x0[0], y0[0], ux[0], uy[0])
    assert len(px) == 2  # first fix has an ETO in the past (> 60 s old)
    # an aircraft that already flew past the 2nd fix (behind, within 40 NM) drops it
    x1, y1 = s16.to_plane(np.array([62.0]), np.array([17.4]))
    px, py, pz = s16.build_path(rec, t_start, x1[0], y1[0], ux[0], uy[0])
    assert len(px) == 1 and pz[0] == 32000.0
    assert s16.parse_route("not json") is None
