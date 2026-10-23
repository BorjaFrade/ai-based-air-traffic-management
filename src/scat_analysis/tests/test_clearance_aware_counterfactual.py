import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cruise_clearance_aware_kinematics import (
    assigned_level_ft, classify_previous_level_intent,
    comparable_speed_delta, has_same_partner_across_horizons,
    previous_heading_track, previous_level_altitudes,
    scenarios_not_opposed, speed_trend_path,
)
from cruise_counterfactual_kinematics import next_clearance_censors

stage09b = importlib.import_module("09b_clearance_aware_counterfactual_cruise")


class ClearanceAwareCounterfactualTests(unittest.TestCase):
    def test_level_at_previous_cfl_holds_current_altitude_without_teleporting(self):
        state, reason = classify_previous_level_intent(35000, 500, 34975, 0, 0)
        self.assertEqual((state, reason), ("LEVEL_AT_PREVIOUS_CFL", "OK"))
        np.testing.assert_allclose(previous_level_altitudes(34975, 35000, 0, state, [0, 60, 120]),
                                   [34975, 34975, 34975])

    def test_climbing_toward_previous_cfl_stops_at_target(self):
        state, reason = classify_previous_level_intent(35000, 500, 33000, 1000, 900)
        self.assertEqual((state, reason), ("CLIMBING_TOWARD_PREVIOUS_CFL", "OK"))
        np.testing.assert_allclose(previous_level_altitudes(33000, 35000, 1000, state,
                                                             [0, 60, 120, 180]),
                                   [33000, 34000, 35000, 35000])

    def test_descending_toward_previous_cfl_stops_at_target(self):
        state, reason = classify_previous_level_intent(33000, 500, 35000, -1000, -900)
        self.assertEqual((state, reason), ("DESCENDING_TOWARD_PREVIOUS_CFL", "OK"))
        np.testing.assert_allclose(previous_level_altitudes(35000, 33000, -1000, state,
                                                             [0, 60, 120, 180]),
                                   [35000, 34000, 33000, 33000])

    def test_new_cfl_above_and_below_previous(self):
        up = stage09b.clearance_components({"cfl": 350, "cfl_unit": "F"},
                                           {"cfl": 370, "cfl_unit": "F"})
        down = stage09b.clearance_components({"cfl": 350, "cfl_unit": "F"},
                                             {"cfl": 330, "cfl_unit": "F"})
        self.assertEqual(up["delta_CFL_ft"], 2000)
        self.assertEqual(down["delta_CFL_ft"], -2000)
        self.assertEqual(up["transition_class"], "LEVEL_CHANGE")

    def test_previous_cfl_missing(self):
        self.assertTrue(np.isnan(assigned_level_ft(None, "F")))
        state, reason = classify_previous_level_intent(np.nan, 100, 35000, 0, 0)
        self.assertEqual(reason, "MISSING_PREVIOUS_CFL")
        self.assertEqual(state, "PREVIOUS_LEVEL_INTENT_UNKNOWN")

    def test_previous_instruction_too_old(self):
        state, reason = classify_previous_level_intent(35000, 1801, 35000, 0, 0)
        self.assertEqual(reason, "PREVIOUS_INSTRUCTION_TOO_OLD")

    def test_heading_wrap_and_inactive_previous_heading(self):
        self.assertAlmostEqual(previous_heading_track(359, 1, 2, 100), 0)
        self.assertTrue(np.isnan(previous_heading_track(180, 1, 2, 100)))

    def test_speed_unit_incompatibility(self):
        self.assertTrue(np.isnan(comparable_speed_delta(.78, "MACH", 280, "KNOT")))
        self.assertEqual(comparable_speed_delta(260, "KNOT", 280, "KNOT"), 20)
        np.testing.assert_allclose(speed_trend_path(300, .1, [0, 60]), [300, 306])
        self.assertTrue(np.isnan(speed_trend_path(300, 1, [0, 60])).all())

    def test_multi_action_transition(self):
        old = {"cfl": 350, "cfl_unit": "F", "assigned_speed_val": 270,
               "assigned_speed_unit": "KNOT"}
        new = {"cfl": 370, "cfl_unit": "F", "assigned_speed_val": 280,
               "assigned_speed_unit": "KNOT"}
        self.assertEqual(stage09b.clearance_components(old, new)["transition_class"], "MULTI_ACTION")

    def test_subsequent_clearance_before_60_s(self):
        self.assertTrue(next_clearance_censors(1059, 1000, 60))
        self.assertTrue(next_clearance_censors(1059, 1000, 120))

    def test_subsequent_clearance_between_60_and_120_s(self):
        self.assertFalse(next_clearance_censors(1070, 1000, 60))
        self.assertTrue(next_clearance_censors(1070, 1000, 120))

    def test_partner_switch_is_not_consistency(self):
        self.assertFalse(has_same_partner_across_horizons([(60, "A"), (120, "B")]))
        self.assertTrue(has_same_partner_across_horizons([(60, "A"), (120, "A")]))

    def test_strong_kinematic_disagreement_rejected(self):
        self.assertFalse(scenarios_not_opposed(-1.1, .5))
        self.assertTrue(scenarios_not_opposed(-.2, .5))

    def test_flight_level_and_altitude_units(self):
        self.assertEqual(assigned_level_ft(350, "F"), 35000)
        self.assertEqual(assigned_level_ft(35000, "A"), 35000)
        self.assertTrue(np.isnan(assigned_level_ft(350, "UNKNOWN")))


if __name__ == "__main__":
    unittest.main()

