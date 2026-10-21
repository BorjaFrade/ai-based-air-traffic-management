import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from common import latest_record, latest_record_with_age


def load_script(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TemporalSafetyTests(unittest.TestCase):
    def test_latest_record_is_as_of_only(self):
        records = [(100.0, {"id": "future"}), (200.0, {"id": "later"})]
        self.assertEqual(latest_record(records, 99.0), {})
        self.assertEqual(latest_record(records, 100.0)["id"], "future")
        record, age = latest_record_with_age(records, 150.0)
        self.assertEqual(record["id"], "future")
        self.assertEqual(age, 50.0)

    def test_week_split_is_flight_disjoint_and_embargoed(self):
        module = load_script("07_explain_interventions.py")
        events = pd.DataFrame(
            [
                {"window_id": "w1", "event_epoch": 1_000, "archive_week": "01", "focal_flight_key": "f1", "ATC_INTERVENTION": 1},
                {"window_id": "w2", "event_epoch": 2_000, "archive_week": "02", "focal_flight_key": "f2", "ATC_INTERVENTION": 0},
                {"window_id": "w3", "event_epoch": 3_000, "archive_week": "03", "focal_flight_key": "f3", "ATC_INTERVENTION": 1},
                {"window_id": "w4", "event_epoch": 4_000, "archive_week": "04", "focal_flight_key": "f4", "ATC_INTERVENTION": 0},
            ]
        )
        train, test = module.temporal_split(events, embargo_seconds=100)
        self.assertTrue(set(train["focal_flight_key"]).isdisjoint(set(test["focal_flight_key"])))
        self.assertTrue(set(train["window_id"]).isdisjoint(set(test["window_id"])))
        self.assertGreaterEqual(test["event_epoch"].min(), train["event_epoch"].max() + 100)


if __name__ == "__main__":
    unittest.main()
