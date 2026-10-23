"""Stage 10 semantic and same-pair kinematic checks."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("stage10_rule_geometry", SCRIPT_DIR / "10_compare_rule_geometry_with_atc_intervention_cruise.py")
stage10 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage10)


class Stage10RulesTest(unittest.TestCase):
    def test_any_neighbour_rule_is_not_nearest_neighbour_rule(self):
        rows = [
            # The closest pair has legal vertical separation; another pair
            # is inside both dimensions, so the WINDOW is rule-positive.
            ("a", -10, 3.0, 2000.0, 2.0, 2.0, 2000.0),
            ("a", -10, 4.0, 500.0, 2.0, 2.0, 500.0),
            # Older rule-positive state must not contaminate the later anchor.
            ("b", -20, 4.0, 500.0, 2.0, 2.0, 500.0),
            ("b", -10, 10.0, 2000.0, 8.0, 2.0, 2000.0),
        ]
        columns = ["window_id", "relative_seconds", "horizontal_separation_nm",
                   "vertical_separation_ft", "cpa_nm", "tcpa_min", "predicted_min_vertical_sep_ft"]
        frame = pd.DataFrame(rows, columns=columns)
        frame["predictor_eligible"] = True
        frame["horizontal_closing_rate_nm_min"] = 1.0
        frame["vertical_closing_rate_fpm"] = 0.0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame.to_parquet(root / "native.parquet", index=False)
            result = stage10.anchor_pairs({"a", "b"}, root / "native.parquet", root / "progress.log").set_index("window_id")
        self.assertEqual(result.loc["a", "anchor_horizontal_separation_nm"], 3.0)
        self.assertTrue(result.loc["a", "anchor_any_5NM_1000FT"])
        self.assertFalse(result.loc["b", "anchor_any_5NM_1000FT"])

    def test_rule_boundary_and_missing_are_not_positive_or_negative(self):
        h = pd.Series([4.9, 5.0, np.nan, 4.0])
        v = pd.Series([999.0, 999.0, 500.0, np.nan])
        self.assertEqual(stage10.rule_status(h, v, 5, 1000).tolist(),
                         ["RULE_POSITIVE", "RULE_NEGATIVE", "RULE_UNKNOWN", "RULE_UNKNOWN"])

    def test_future_rule_requires_same_pair_quality_and_finite_tcpa(self):
        frame = pd.DataFrame({
            "anchor_horizontal_separation_nm": [10, 10, 31, 10, np.nan],
            "anchor_cpa_nm": [4, 4, 4, 4, 4],
            "anchor_tcpa_min": [2, 0.001, 2, 8, 2],
            "anchor_predicted_min_vertical_sep_ft": [900, 900, 900, 900, 900],
        })
        self.assertEqual(stage10.future_rule(frame).tolist(),
                         ["RULE_POSITIVE", "RULE_UNKNOWN", "RULE_UNKNOWN", "RULE_UNKNOWN", "RULE_UNKNOWN"])

    def test_time_to_rule_region_separates_entry_no_entry_and_unknown(self):
        frame = pd.DataFrame({
            "window_id": ["enters", "stays_vertical", "missing"],
            "ATC_INTERVENTION": [1, 0, 1],
            "anchor_horizontal_separation_nm": [10, 10, 10],
            "anchor_cpa_nm": [1, 1, 1],
            "anchor_tcpa_min": [2, 2, np.nan],
            "anchor_vertical_separation_ft": [500, 2000, 500],
            "anchor_vertical_closing_rate_fpm": [0, 0, 0],
        })
        result = stage10.time_to_region(frame, 5, 1000).set_index("window_id")
        self.assertEqual(result.loc["enters", "time_to_rule_region_s"], 65)
        self.assertTrue(result.loc["stays_vertical", "model_evaluable"])
        self.assertFalse(result.loc["stays_vertical", "enters_within_horizon"])
        self.assertFalse(result.loc["missing", "model_evaluable"])

    def test_pair_overlap_keeps_unknown_separate(self):
        frame = pd.DataFrame({"matched_pair_id": [0, 0, 1, 1],
                              "ATC_INTERVENTION": [1, 0, 1, 0],
                              "rule_5NM_1000FT": ["RULE_POSITIVE", "RULE_NEGATIVE",
                                                  "RULE_UNKNOWN", "RULE_NEGATIVE"]})
        _, pairs = stage10.overlap(frame, "5NM_1000FT")
        self.assertEqual(pairs.set_index("matched_pair_id").category.to_dict(),
                         {0: "RULE_I1_C0", 1: "RULE_UNKNOWN_PAIR"})


if __name__ == "__main__":
    unittest.main()
