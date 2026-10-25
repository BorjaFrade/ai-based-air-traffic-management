import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage12f_robustness.py"
SPEC = importlib.util.spec_from_file_location("stage12f_robustness", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_flight_vertex_split_has_no_flight_leakage_and_discards_cross_edges():
    d = pd.DataFrame({
        "a_flight_key": ["A", "A", "B", "C", "D", "E", "F", "G"],
        "b_flight_key": ["B", "C", "C", "D", "E", "F", "G", "A"],
    })
    train, test, dropped, train_flights, test_flights = MOD.flight_edge_split(d, .4, 9)
    assert train_flights.isdisjoint(test_flights)
    assert not (train & test).any()
    assert np.array_equal(train | test | dropped, np.ones(len(d), dtype=bool))
    train_vertices = set(d.loc[train, "a_flight_key"]) | set(d.loc[train, "b_flight_key"])
    test_vertices = set(d.loc[test, "a_flight_key"]) | set(d.loc[test, "b_flight_key"])
    assert train_vertices.isdisjoint(test_vertices)


def test_duration_rates_are_explicitly_normalized_in_minutes():
    d = pd.DataFrame({
        "log_min_h": [1., 2.], "log_simultaneous_v": [3., 4.], "start_closing": [1., 1.],
        "crossing_angle": [90., 90.], "fraction_converging": [.5, .5], "relative_speed": [2., 2.],
        "track_change": [60., 60.], "altitude_change": [6000., 6000.],
        "observed_duration_s": [60., 120.],
    })
    x = MOD.raw_matrix(d, "geometry_plus_rates")
    assert np.allclose(x.track_rate_deg_min, [60., 30.])
    assert np.allclose(x.altitude_rate_ft_min, [6000., 3000.])


def test_symmetric_phase_pair_is_order_invariant():
    d = pd.DataFrame({"a_phase": ["CLIMB", "CRUISE_OR_LEVEL"],
                      "b_phase": ["CRUISE_OR_LEVEL", "CLIMB"]})
    assert MOD.phase_pair(d).nunique() == 1
