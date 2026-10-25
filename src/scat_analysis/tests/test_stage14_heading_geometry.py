"""Stage 14 heading-change angle and event-selection invariants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage14_heading_change_geometry as stage14


def _clearance(clearance_id, clearance_type, previous, new, *, fl=300):
    return {
        "clearance_id": clearance_id,
        "archive_week": "20170107_20170113",
        "flight_key": f"flight-{clearance_id}",
        "source_member": f"{clearance_id}.json",
        "clearance_epoch": 1000.0,
        "clearance_timestamp": "2017-01-07T00:16:40Z",
        "clearance_type": clearance_type,
        "previous_value": json.dumps(previous),
        "new_value": json.dumps(new),
        "previous_record_json": "{}",
        "clearance_cfl_unit": "F",
        "surveillance_match_valid": True,
        "surveillance_match_age_s": 1.0,
        "matched_state_epoch": "999",
        "matched_latitude": "50.0",
        "matched_longitude": "0.0",
        "matched_altitude_ft": "30000",
        "matched_flight_level": str(fl),
        "matched_ground_speed_kt": "450",
        "matched_track_deg": "90",
        "matched_heading_deg": "250",
        "matched_vertical_rate_fpm": "0",
    }


def test_shortest_signed_heading_change_wraps_at_north():
    previous = pd.Series([350.0, 10.0, 0.0, 360.0])
    new = pd.Series([10.0, 350.0, 180.0, 0.0])
    assert stage14.shortest_signed_heading_change(previous, new).tolist() == [20, -20, -180, 0]


def test_prepare_events_excludes_initial_cancel_beacon_and_equivalent_heading():
    source = pd.DataFrame([
        _clearance("right-wrap", "HEADING", {"assigned_heading_val": 350},
                   {"assigned_heading_val": 10}),
        _clearance("left-wrap", "HEADING", {"assigned_heading_val": 10},
                   {"assigned_heading_val": 350}),
        _clearance("initial", "HEADING", {"assigned_heading_val": None},
                   {"assigned_heading_val": 90}),
        _clearance("cancel", "HEADING", {"assigned_heading_val": 90},
                   {"assigned_heading_val": None}),
        _clearance("beacon", "HEADING", {"assign_heading_beacon": "ABC"},
                   {"assign_heading_beacon": "DEF"}),
        _clearance("equivalent", "HEADING", {"assigned_heading_val": 0},
                   {"assigned_heading_val": 360}),
        _clearance("outside-band", "HEADING", {"assigned_heading_val": 90},
                   {"assigned_heading_val": 100}, fl=420),
        _clearance("combined", "HEADING+SPEED", {"assigned_heading_val": 280},
                   {"assigned_heading_val": 300}),
    ])

    events, audit = stage14.prepare_events(source)

    assert audit["heading_records"] == 8
    assert audit["heading_only_records"] == 7
    assert audit["combined_heading_records"] == 1
    assert audit["both_assigned_headings_numeric"] == 5
    assert audit["actual_numeric_heading_changes"] == 4
    assert audit["new_numeric_heading_instructions"] == 5
    assert audit["eligible_clearance_events"] == 4
    assert set(events.clearance_id) == {"right-wrap", "left-wrap", "initial", "combined"}
    actual = events.set_index("clearance_id")
    assert actual.loc["right-wrap", "signed_heading_change_deg"] == 20
    assert actual.loc["left-wrap", "signed_heading_change_deg"] == -20
    assert actual.loc["combined", "turn_direction"] == "right"
    assert pd.isna(actual.loc["initial", "absolute_heading_change_deg"])
    assert actual.loc["initial", "absolute_heading_instruction_from_observed_deg"] == 160
