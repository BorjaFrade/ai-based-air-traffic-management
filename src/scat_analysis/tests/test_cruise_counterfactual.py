import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cruise_counterfactual_kinematics import (
    angular_difference_deg, conservative_label, finite_cpa, haversine_nm,
    nearest_within_radius, next_clearance_censors, propagate, joint_separation_nm,
    interaction_agreement_for_clearance,
)


def state(flight, lat=55.0, lon=13.0, alt=35000.0, speed=360.0,
          track=90.0, vr=0.0, epoch=1000.0):
    return {"flight_key": flight, "latitude": lat, "longitude": lon,
            "altitude_ft": alt, "ground_speed_kt": speed,
            "track_deg": track, "vertical_rate_fpm": vr, "time_epoch": epoch}


class CruiseCounterfactualTests(unittest.TestCase):
    def test_straight_line_and_knots_to_nm(self):
        a = state("a")
        lat, lon, alt = propagate(a, np.array([0, 60]), 1000)
        self.assertAlmostEqual(float(haversine_nm(lat[0], lon[0], lat[1], lon[1])), 6.0, delta=.02)
        self.assertAlmostEqual(alt[1], 35000)

    def test_pre_anchor_age_is_propagated_but_future_initial_state_rejected(self):
        a = state("a", epoch=995)
        lat, lon, _ = propagate(a, np.array([0]), 1000)
        self.assertGreater(lon[0], 13.0)
        with self.assertRaises(ValueError):
            propagate(state("a", epoch=1001), np.array([0]), 1000)

    def test_parallel_aircraft_keep_separation(self):
        a = state("a")
        b = state("b", lat=55 + 2 / 60)
        result = finite_cpa(a, b, 1000, 120)
        self.assertAlmostEqual(result["cpa_nm"], result["current_horizontal_sep_nm"], delta=.01)

    def test_converging_head_on_cpa_within_horizon(self):
        a = state("a", speed=300, track=90)
        b = state("b", lon=13 + 10 / (60 * np.cos(np.deg2rad(55))), speed=300, track=270)
        result = finite_cpa(a, b, 1000, 120)
        self.assertLess(result["cpa_nm"], .05)
        self.assertAlmostEqual(result["tcpa_s"], 60, delta=1)

    def test_diverging_pair_raw_tcpa_negative(self):
        a = state("a", speed=300, track=270)
        b = state("b", lon=13 + 10 / (60 * np.cos(np.deg2rad(55))), speed=300, track=90)
        result = finite_cpa(a, b, 1000, 120)
        self.assertLess(result["raw_tcpa_s"], 0)
        self.assertEqual(result["tcpa_s"], 0)
        self.assertAlmostEqual(result["cpa_nm"], result["current_horizontal_sep_nm"], delta=.01)

    def test_crossing_geometry(self):
        a = state("a", track=0, speed=300)
        b = state("b", lat=55 + 5 / 60, lon=13 - 5 / (60 * np.cos(np.deg2rad(55))),
                  track=90, speed=300)
        result = finite_cpa(a, b, 1000, 120)
        self.assertLess(result["cpa_nm"], result["current_horizontal_sep_nm"])
        self.assertGreater(result["tcpa_s"], 0)

    def test_heading_wrap(self):
        self.assertEqual(float(angular_difference_deg(359, 1)), -2)
        self.assertEqual(float(angular_difference_deg(1, 359)), 2)

    def test_climb_descent_and_level_units(self):
        a = state("a", vr=1000)
        b = state("b", vr=-1000)
        c = state("c", vr=0)
        self.assertAlmostEqual(propagate(a, np.array([60]), 1000)[2][0], 36000)
        self.assertAlmostEqual(propagate(b, np.array([60]), 1000)[2][0], 34000)
        self.assertAlmostEqual(propagate(c, np.array([60]), 1000)[2][0], 35000)

    def test_missing_neighbour(self):
        self.assertEqual(nearest_within_radius(state("a"), [], 50), [])

    def test_joint_geometry_uses_feet_to_nautical_miles_without_safety_cutoff(self):
        self.assertAlmostEqual(float(joint_separation_nm(0, 6076.12)), 1.0)
        self.assertAlmostEqual(float(joint_separation_nm(3, 4 * 6076.12)), 5.0)

    def test_timing_only_corroboration_is_speed_specific(self):
        types = np.array(["LEVEL", "HEADING", "SPEED", "SPEED"])
        joint = np.array([False, True, False, True])
        timing = np.array([True, False, True, False])
        self.assertEqual(interaction_agreement_for_clearance(types, joint, timing).tolist(),
                         [False, True, True, True])

    def test_next_clearance_censoring(self):
        self.assertFalse(next_clearance_censors(1100, 1000, 60))
        self.assertTrue(next_clearance_censors(1100, 1000, 120))
        self.assertFalse(next_clearance_censors(np.nan, 1000, 120))

    def test_horizon_clipping(self):
        a = state("a", speed=300, track=90)
        b = state("b", lon=13 + 10 / (60 * np.cos(np.deg2rad(55))), speed=300, track=270)
        shorter = finite_cpa(a, b, 1000, 30)
        longer = finite_cpa(a, b, 1000, 120)
        self.assertEqual(shorter["tcpa_s"], 30)
        self.assertGreater(shorter["cpa_nm"], longer["cpa_nm"])

    def test_30_and_50_nm_candidate_selection(self):
        a = state("a")
        near = state("b", lat=55 + 20 / 60)
        middle = state("c", lat=55 + 40 / 60)
        far = state("d", lat=55 + 60 / 60)
        self.assertEqual([r["flight_key"] for r in nearest_within_radius(a, [near, middle, far], 30)], ["b"])
        self.assertEqual({r["flight_key"] for r in nearest_within_radius(a, [near, middle, far], 50)}, {"b", "c"})

    def test_unknown_when_evidence_insufficient_or_scenarios_disagree(self):
        self.assertEqual(conservative_label(False, "YES", True, True, True, False, 2, True), "UNKNOWN")
        self.assertEqual(conservative_label(True, "YES", True, True, True, False, 2, False), "UNKNOWN")
        self.assertEqual(conservative_label(True, "YES", True, True, True, False, 1, True), "UNKNOWN")
        self.assertEqual(conservative_label(True, "YES", True, True, True, False, 2, True),
                         "LIKELY_INTERACTION_RELATED")


if __name__ == "__main__":
    unittest.main()
