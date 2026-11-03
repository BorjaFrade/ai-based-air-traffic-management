"""Stage 15 invariants: rule equivalence, geometry, control selection, loader parity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "acquisition"))
import stage15_predicted_separation as s15  # noqa: E402

RULES = s15.load_rules()


def test_classifier_matches_reference_detector():
    safety = pytest.importorskip("opensky_safety")
    profile = json.loads(s15.PROFILE_PATH.read_text())["ENROUTE"]
    state_rules = safety.normalize_state_rules(profile["states"])
    rng = np.random.default_rng(1)
    n = 4000
    H = rng.uniform(0, 12, (n, 2))
    V = rng.choice([0, 300, 800, 999.5, 1000, 1001, 1100, 1250, 1450, 1600, 2500], (n, 2)) \
        + rng.uniform(-30, 30, (n, 2)) * rng.integers(0, 2, (n, 2))
    V = np.abs(V)
    got = s15.classify_states(H, V, RULES)[:, 0]
    names = {"NORMAL": 0, "PROXIMITY": 1, "HIGH_RISK": 2, "CONFLICT": 3}
    want = [names[safety.classify_pair_dynamic(h[0], v[0], next_horizontal_nm=h[1],
                                               next_vertical_ft=v[1], state_rules=state_rules)]
            for h, v in zip(H, V)]
    assert (got == np.array(want)).all()


def test_stable_rvsm_crossing_is_normal_and_closing_head_on_same_level_is_red():
    # Parallel, 0.3 NM apart, exactly 1000 ft: legal and stable -> NORMAL.
    H = np.full((1, 3), 0.3)
    V = np.full((1, 3), 1000.0)
    assert s15.classify_states(H, V, RULES)[0, 0] == 0
    # Same level, closing from 20 NM at 16 NM/min: first RED when H < 5 NM.
    t = np.arange(0, 125, 5, dtype=float)
    H = np.abs(20.0 - 16.0 / 60.0 * t)[None, :]
    V = np.zeros_like(H)
    S = s15.classify_states(H, V, RULES)[0]
    first_red = int(np.argmax(S == 3))
    assert H[0, first_red] < 5 <= H[0, first_red - 1]


def _focal(vx=0.0, vy=0.0, z=35000.0, vr=0.0):
    return {"vx": vx, "vy": vy, "z": z, "vr": vr, "z_raw": z, "vr_raw": vr}


def test_head_on_prediction_windows_and_first_red_time():
    step, windows = RULES["step"], s15.make_windows(RULES["step"])
    grid = np.arange(HORIZON := 600 // step + 2, dtype=float) * step
    speed = 480.0 / 3600.0
    # Neighbour 60 NM ahead, flying toward the focal aircraft, same level: closure 16 NM/min.
    out = s15.evaluate_pairs(np.array([0.0]), np.array([60.0]), np.array([0.0]), np.array([-speed]),
                             np.array([35000.0]), np.array([0.0]), _focal(vy=speed), RULES, grid,
                             windows, None, 30, 1500)
    # H(t) = 60 - 16/60*t -> 5 NM at t = 206.25 s -> first RED sample at 210 s.
    assert out["t_first_red_s"] == 210.0
    assert out["state_c0"] == 0 and out["state_c2"] == 0
    # The pair crosses at 225 s and diverges afterwards: the 5-10 min window alone is clear,
    # but the cumulative (worst-so-far) state keeps the RED.
    assert out["state_c5"] == 3 and out["state_w5"] == 3
    assert out["state_w10"] == 0 and out["state_c10"] == 3
    assert out["minnorm_c5"] < 1.0 and out["minnorm_c2"] > 1.0


def test_vertical_separation_removes_predicted_loss():
    step, windows = RULES["step"], s15.make_windows(RULES["step"])
    grid = np.arange(600 // step + 2, dtype=float) * step
    speed = 480.0 / 3600.0
    out = s15.evaluate_pairs(np.array([0.0]), np.array([60.0]), np.array([0.0]), np.array([-speed]),
                             np.array([35000.0 + 2000.0]), np.array([0.0]), _focal(vy=speed), RULES,
                             grid, windows, None, 30, 1500)
    assert out["state_c10"] == 0 and np.isnan(out["t_first_red_s"])


def test_post_order_level_change_resolves_or_creates_loss():
    step, windows = RULES["step"], s15.make_windows(RULES["step"])
    grid = np.arange(600 // step + 2, dtype=float) * step
    speed = 480.0 / 3600.0
    args = (np.array([0.0]), np.array([60.0]), np.array([0.0]), np.array([-speed]))
    # Same level conflict; order climbs the focal aircraft by 2000 ft -> post-order is clear.
    out = s15.evaluate_pairs(*args, np.array([35000.0]), np.array([0.0]), _focal(vy=speed), RULES,
                             grid, windows, 37000.0, 30, 1500)
    assert out["state_c10"] == 3 and out["post_state_c10"] == 0
    # Neighbour is 2000 ft above; order climbs the focal aircraft INTO its level.
    out = s15.evaluate_pairs(*args, np.array([37000.0]), np.array([0.0]), _focal(vy=speed), RULES,
                             grid, windows, 37000.0, 30, 1500)
    assert out["state_c10"] == 0 and out["post_state_c10"] == 3


def test_post_order_altitude_stops_at_target_and_respects_delay():
    grid = np.arange(0, 305, 5, dtype=float)
    z = s15.focal_post_order_altitude(35000.0, 0.0, 36000.0, grid, 30.0, 1500.0, 100.0)
    assert z[0] == 35000.0 and z[grid == 30][0] == 35000.0
    assert z.max() == 36000.0 and z[-1] == 36000.0
    assert z[grid == 60][0] == pytest.approx(35000.0 + 1500.0 * 30 / 60)
    z = s15.focal_post_order_altitude(35000.0, -50.0, 34000.0, grid, 30.0, 1500.0, 100.0)
    assert z[-1] == 34000.0 and np.all(np.diff(z) <= 1e-9)


def _states(rows):
    cols = ["archive_week", "flight_key", "source_member", "plot_index", "time_epoch", "latitude",
            "longitude", "altitude_ft", "flight_level", "ground_speed_kt", "track_deg",
            "vertical_rate_fpm", "on_ground"]
    return pd.DataFrame(rows, columns=cols)


def _synthetic_week(week="20170107_20170113", seed=3):
    rng = np.random.default_rng(seed)
    rows, plot = [], 0
    t0 = 1_483_747_200.0
    # Four flights flying east/west on the same parallel with a head-on pair.
    specs = [("A", 55.0, 10.0, 90.0, 35000), ("B", 55.0, 14.0, 270.0, 35000),
             ("C", 55.6, 11.0, 90.0, 37000), ("D", 54.4, 12.0, 90.0, 33000)]
    for name, lat, lon, track, alt in specs:
        for k in range(0, 1800, 5):
            speed = 450.0
            east = speed * np.sin(np.deg2rad(track)) * k / 3600.0 / 60.0 / np.cos(np.deg2rad(lat))
            rows.append([week, name, f"{name}.json", plot, t0 + k, lat, lon + east, float(alt),
                         alt / 100.0, speed, track, 0.0, False])
            plot += 1
    return _states(rows), t0


def test_lean_loader_matches_stage13_loader(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    a, _ = _synthetic_week("20170107_20170113")
    b, _ = _synthetic_week("20170215_20170221", seed=4)
    b["time_epoch"] += 5_000_000
    clash = b.loc[b.flight_key.eq("A")].copy().iloc[:40]
    clash["source_member"] = "A.other.json"
    bad = b.iloc[:50].copy()
    bad["flight_level"] = 150.0  # outside the band
    path = tmp_path / "states.parquet"
    pq.write_table(pa.Table.from_pandas(pd.concat([a, b, clash, bad], ignore_index=True)), path, row_group_size=700)
    parquet = pq.ParquetFile(path)
    lean, audit_l = s15.load_week_states_lean(parquet, "20170215_20170221")
    ref, audit_r = s15.s13.load_week_states(parquet, "20170215_20170221")
    for column in ("flight_key", "source_member", "entity_key"):
        lean[column] = lean[column].astype(object)
        ref[column] = ref[column].astype(object)
    pd.testing.assert_frame_equal(lean.reset_index(drop=True), ref.reset_index(drop=True))
    assert audit_l == audit_r


def _cfg(**kw):
    base = {"radius_nm": 150.0, "controls": 3, "seed": 15, "guard_s": 120.0,
            "class_threshold": 300.0, "delay_s": 30.0, "nominal_fpm": 1500.0}
    base.update(kw)
    return base


def test_controls_are_order_free_same_flight_and_reproducible(monkeypatch):
    states, t0 = _synthetic_week()
    states["entity_key"] = states.flight_key.astype(str)
    states = states.sort_values(["time_epoch", "plot_index"], kind="mergesort").reset_index(drop=True)
    arrays = s15.build_arrays(states, RULES["deadband"])
    ev = {"event_id": "LEVEL:x", "entity": "A", "clearance_epoch": t0 + 900, "matched_vertical_rate_fpm": 0.0}
    clearances = {"A": np.array([t0 + 900.0, t0 + 1000.0])}
    first = s15.pick_controls(ev, arrays, clearances, 3, 15, 120.0, 300.0)
    again = s15.pick_controls(ev, arrays, clearances, 3, 15, 120.0, 300.0)
    assert len(first) == 3 and (first == again).all()
    times = arrays["time_epoch"][first]
    assert all(arrays["entity_key"][i] == "A" for i in first)
    assert (np.abs(times - (t0 + 900)) > 120).all() and (np.abs(times - (t0 + 1000)) > 120).all()
    # A climbing event must not draw level controls.
    ev["matched_vertical_rate_fpm"] = 1800.0
    assert len(s15.pick_controls(ev, arrays, clearances, 3, 15, 120.0, 300.0)) == 0


def test_end_to_end_head_on_pair_is_flagged_and_controls_exist():
    states, t0 = _synthetic_week()
    states["entity_key"] = states.flight_key.astype(str)
    states = states.sort_values(["time_epoch", "plot_index"], kind="mergesort").reset_index(drop=True)
    s15._CLEARANCE_CACHE["table"] = pd.DataFrame(
        {"archive_week": ["20170107_20170113"], "flight_key": ["A"], "source_member": ["A.json"],
         "clearance_epoch": [t0 + 60.0]})
    ev = pd.DataFrame([{
        "event_id": "LEVEL:1", "order_type": "LEVEL", "clearance_id": "1", "archive_week": "20170107_20170113",
        "flight_key": "A", "source_member": "A.json", "clearance_epoch": t0 + 60.0,
        "matched_state_epoch": t0 + 60.0, "matched_latitude": 55.0, "matched_longitude": 10.0 +
        450 * 60 / 3600 / 60 / np.cos(np.deg2rad(55.0)), "matched_altitude_ft": 35000.0,
        "matched_flight_level": 350.0, "matched_ground_speed_kt": 450.0, "matched_track_deg": 90.0,
        "matched_vertical_rate_fpm": 0.0, "previous_cfl": 350.0, "new_cfl": 370.0, "direction": "climb",
        "absolute_heading_instruction_from_observed_deg": np.nan, "turn_direction": np.nan,
        "heading_event_kind": np.nan}])
    result = s15.analyse_week("20170107_20170113", ev, states, RULES, _cfg())
    event = result.loc[result.query_kind.eq("event")].iloc[0]
    assert event.valid_query and event.n_nb_100 >= 1
    assert event.rel_flight_key in {"B", "C", "D"}
    controls = result.loc[result.query_kind.eq("control")]
    assert len(controls) == 3 and (controls.parent_event_id == "LEVEL:1").all()
    assert controls.post_state_c10.isna().all() or "post_state_c10" not in controls
    # A and B (35 000 ft) fly toward each other 4 deg apart: RED must be predicted early on.
    assert event.state_c10 == 3 or event.state_c5 == 3 or event.state_c2 == 3


def test_level_hold_sensitivity_removes_all_vertical_motion():
    states, _ = _synthetic_week()
    states["entity_key"] = states.flight_key.astype(str)
    states["vertical_rate_fpm"] = 1800.0
    states = states.sort_values(["time_epoch", "plot_index"], kind="mergesort").reset_index(drop=True)
    assert (s15.build_arrays(states, RULES["deadband"], "rate")["vr_eff"] == 1800.0).all()
    assert (s15.build_arrays(states, RULES["deadband"], "level")["vr_eff"] == 0.0).all()
