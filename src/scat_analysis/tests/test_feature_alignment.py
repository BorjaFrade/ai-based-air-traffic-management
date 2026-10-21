import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("stage05", SCRIPT_DIR / "05_build_interaction_features.py")
stage05 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stage05)


class FeatureAlignmentTests(unittest.TestCase):
    def test_trends_follow_sorted_group_positions(self):
        features = pd.DataFrame(
            [
                {"window_id": "B", "focal_flight_key": "fb", "other_flight_key": "ob", "time_epoch": 60, "horizontal_separation_nm": 18.0},
                {"window_id": "A", "focal_flight_key": "fa", "other_flight_key": "oa", "time_epoch": 0, "horizontal_separation_nm": 10.0},
                {"window_id": "B", "focal_flight_key": "fb", "other_flight_key": "ob", "time_epoch": 0, "horizontal_separation_nm": 20.0},
                {"window_id": "A", "focal_flight_key": "fa", "other_flight_key": "oa", "time_epoch": 60, "horizontal_separation_nm": 7.0},
            ],
            index=[100, 500, 900, 1300],
        )
        result = stage05.add_trends(features)
        at_60 = result.loc[result["time_epoch"] == 60].set_index("window_id")
        self.assertAlmostEqual(at_60.loc["A", "separation_trend_60s"], -3.0)
        self.assertAlmostEqual(at_60.loc["B", "separation_trend_60s"], -2.0)


if __name__ == "__main__":
    unittest.main()
