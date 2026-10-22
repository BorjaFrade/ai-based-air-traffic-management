import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from pilot_scope import PILOT_WEEK, SCOPE_ERROR, assert_pilot_weeks


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AltitudeBandPilotTests(unittest.TestCase):
    def test_nonoverlapping_band_boundaries(self):
        band_of = load_script("05c_audit_window_features_v2.py").band_of
        cases = [(-1001, "UNKNOWN_INVALID"), (-1000, "VERY_LOW"),
                 (1999.99, "VERY_LOW"), (2000, "LOW"), (7999.99, "LOW"),
                 (8000, "MID"), (28999.99, "MID"), (29000, "CRUISE_FL290_410"),
                 (41000, "CRUISE_FL290_410"), (41000.01, "ABOVE_FL410"),
                 (60001, "UNKNOWN_INVALID")]
        for altitude, expected in cases:
            with self.subTest(altitude=altitude):
                self.assertEqual(band_of(altitude), expected)

    def test_pilot_week_guard_rejects_any_other_week(self):
        assert_pilot_weeks([PILOT_WEEK])
        with self.assertRaises(SystemExit) as error:
            assert_pilot_weeks([PILOT_WEEK, "20161022_20161028"])
        self.assertIn(SCOPE_ERROR, str(error.exception))

    def test_cruise_matching_uses_exact_context_and_unique_controls(self):
        matching = load_script("04c_cruise_matching_v2.py")
        common = {"flight_phase": "CRUISE_OR_LEVEL", "centre": "1", "hour_utc": 12,
                  "analysis_altitude_ft": 35000, "local_traffic_density": 1,
                  "aircraft_type": "A320"}
        interventions = pd.DataFrame([
            {**common, "window_id": "I:a", "focal_flight_key": "a", "event_epoch": 100.0, "ground_speed_at_anchor_kt": 400.0},
            {**common, "window_id": "I:b", "focal_flight_key": "b", "event_epoch": 130.0, "ground_speed_at_anchor_kt": 410.0},
        ])
        controls = pd.DataFrame([
            {**common, "window_id": "C:x", "focal_flight_key": "x", "event_epoch": 110.0, "ground_speed_at_anchor_kt": 401.0, "source_clearance_id": "old-x"},
            {**common, "window_id": "C:y", "focal_flight_key": "y", "event_epoch": 120.0, "ground_speed_at_anchor_kt": 411.0, "source_clearance_id": "old-y"},
            {**common, "flight_phase": "CLIMB", "window_id": "C:z", "focal_flight_key": "z", "event_epoch": 110.0,
             "ground_speed_at_anchor_kt": 400.0, "source_clearance_id": "old-z"},
        ])
        config = {"name": "test", "speed_kt": 10, "altitude_ft": 500, "density": 0, "time_s": 60}
        pairs = matching.caliper_match(interventions, controls, config)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(pairs.control_window_id.duplicated().any())
        self.assertTrue((pairs.intervention_flight_phase == pairs.control_flight_phase).all())
        self.assertTrue((pairs.intervention_centre == pairs.control_centre).all())
        self.assertTrue((pairs.intervention_hour_utc == pairs.control_hour_utc).all())
        self.assertTrue(pairs.control_time_difference_s.le(60).all())

    def test_cruise_robustness_excludes_matched_context_and_clipped_cpa(self):
        analysis = load_script("06d_validate_cruise_matching_v2.py")
        v1 = {"SMD": -.20}
        v2 = {"SMD": -.18, "intervention_count": 90, "intervention_missing": 10,
              "control_count": 90, "control_missing": 10,
              "bootstrap_conservative_SMD_p025": -.23, "bootstrap_conservative_SMD_p975": -.12}
        self.assertEqual(analysis.feature_classification("min_future_cpa_ge_1s_within_30nm", v1, v2)[0], "ROBUST")
        self.assertEqual(analysis.feature_classification("min_cpa_within_30nm", v1, v2)[0], "PARTIALLY_ROBUST")
        self.assertEqual(analysis.feature_classification("ground_speed_at_anchor_kt", v1, v2)[0], "UNSTABLE")
        self.assertTrue(np.isclose(np.sin(np.deg2rad(359)), np.sin(np.deg2rad(-1))))


if __name__ == "__main__":
    unittest.main()
