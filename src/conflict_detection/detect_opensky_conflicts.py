#!/usr/bin/env python3
"""
Evaluate 1-, 2- and 5-minute separation alerts from preprocessed OpenSky
10-second aircraft-state CSVs. Keep in project_root/src/conflict_detection/.

Dependencies and separation_profiles.json remain in src/acquisition/.
The raw Trino archives are not reread by this detector.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from pyproj import Geod


# =============================================================================
# LOCAL DEPENDENCIES
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ACQUISITION_DIR = SRC_DIR / "acquisition"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if not ACQUISITION_DIR.exists():
    raise FileNotFoundError(
        "OpenSky acquisition dependency directory not found:\n"
        f"{ACQUISITION_DIR}"
    )

if str(ACQUISITION_DIR) not in sys.path:
    sys.path.insert(0, str(ACQUISITION_DIR))

from area_presets import (  # noqa: E402
    filter_dataframe_to_area,
    load_area_presets,
    resolve_area_preset,
)

from opensky_safety import (  # noqa: E402
    STATE_NORMAL,
    STATE_PROXIMITY,
    STATE_HIGH_RISK,
    STATE_CONFLICT,
    STATE_PRIORITY,
    classify_pair,
    classify_pair_dynamic,
    pair_separation_at_time,
    load_safety_profile,
    enabled_states,
    profile_vertical_rate_deadband_fpm,
    vertical_separation_for_rule,
)


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "opensky_10s"
EVENT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "opensky_separation"
SNAPSHOT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "opensky_snapshots"

BBOX_PRESETS_FILE = ACQUISITION_DIR / "bbox_presets.json"
SEPARATION_PROFILES_FILE = ACQUISITION_DIR / "separation_profiles.json"

DEFAULT_SECTOR = "LECMSAN_ENVELOPE"
DEFAULT_ANALYSIS_AREA = "LECMSAN"
SEPARATION_PROFILE = "ENROUTE"

# Observation grid.
SNAPSHOT_STEP_SECONDS = 10

# Prediction grid.
LOOKAHEAD_MINUTES = (1, 2, 5)
LOOKAHEAD_SECONDS = int(max(LOOKAHEAD_MINUTES) * 60)
PREDICTION_STEP_SECONDS = 5

# Reconstruction safeguards.
MAX_INTERPOLATION_GAP_SECONDS = 20.0
MAX_EXTRAPOLATION_SECONDS = 10.0
MAX_STATE_AGE_SECONDS = 15.0
EXACT_POSITION_TOLERANCE_SECONDS = 0.75

# Cruise experiment.
FILTER_CRUISE_FL_BAND = True
CRUISE_FL_MIN = 290
CRUISE_FL_MAX = 410
CHECK_GROUND_AIRCRAFT = False

# Input streaming.
READ_CHUNK_ROWS = 250_000
REQUIRE_CHRONOLOGICAL_INPUT = True

# Pair-screening switches.
ENABLE_DISTANCE_PREFILTER = True
ENABLE_CPA_PREFILTER = True
CPA_PREFILTER_MARGIN_NM = 1.0

# Save regularised 10-second traffic states by default.
SAVE_REGULARIZED_SNAPSHOTS = True

MPS_TO_KT = 1.9438444924406
MPS_TO_FPM = 196.8503937008
GEOD = Geod(ellps="WGS84")

OUTPUT_SUFFIXES = {
    STATE_PROXIMITY: "separation_proximity_events",
    STATE_HIGH_RISK: "separation_high_risk_events",
    STATE_CONFLICT: "loss_of_separation_events",
}

STATE_COLOURS = {
    STATE_PROXIMITY: "YELLOW",
    STATE_HIGH_RISK: "ORANGE",
    STATE_CONFLICT: "RED",
}


# =============================================================================
# SAFETY PROFILE
# =============================================================================

ACTIVE_SAFETY_PROFILE = load_safety_profile(
    SEPARATION_PROFILES_FILE,
    SEPARATION_PROFILE,
)

SAFETY_STATE_RULES = ACTIVE_SAFETY_PROFILE["states"]
ENABLED_SAFETY_STATES = enabled_states(SAFETY_STATE_RULES)

RED_RULE = SAFETY_STATE_RULES[STATE_CONFLICT]
MIN_HORIZONTAL_SEPARATION_NM = float(RED_RULE["horizontal_nm"])
MIN_VERTICAL_SEPARATION_FT = float(RED_RULE["vertical_ft"])
VERTICAL_SEPARATION_TOLERANCE_FT = float(
    RED_RULE.get("vertical_tolerance_ft", 0.0)
)
VERTICAL_RATE_DEADBAND_FPM = profile_vertical_rate_deadband_fpm(
    SAFETY_STATE_RULES
)

ACTIVE_EVENT_STATES = tuple(
    state
    for state in (
        STATE_PROXIMITY,
        STATE_HIGH_RISK,
        STATE_CONFLICT,
    )
    if state in ENABLED_SAFETY_STATES
)


def active_warning_horizontal_nm() -> float:
    values = []
    for state, rule in SAFETY_STATE_RULES.items():
        if state == STATE_NORMAL or state not in ENABLED_SAFETY_STATES:
            continue
        try:
            value = float(rule.get("horizontal_nm"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return max(values) if values else MIN_HORIZONTAL_SEPARATION_NM * 1.5


def active_warning_vertical_ft() -> float:
    values = []
    for state, rule in SAFETY_STATE_RULES.items():
        if state == STATE_NORMAL or state not in ENABLED_SAFETY_STATES:
            continue
        try:
            value = float(rule.get("vertical_ft"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return max(values) if values else MIN_VERTICAL_SEPARATION_FT * 1.5


def classify_pair_active(
    horizontal_nm: float,
    vertical_ft: float | None,
    *,
    next_horizontal_nm: float | None = None,
    next_vertical_ft: float | None = None,
) -> str:
    """
    Use the dynamic active-separator rules when enabled by the profile.

    Falls back to the legacy static classifier for older profiles.
    """
    dynamic_enabled = (
        SAFETY_STATE_RULES.get(
            STATE_PROXIMITY,
            {},
        ).get(
            "requires_active_separator_deterioration",
            False,
        )
        or SAFETY_STATE_RULES.get(
            STATE_HIGH_RISK,
            {},
        ).get(
            "requires_active_separator_deterioration",
            False,
        )
    )

    if dynamic_enabled:
        return classify_pair_dynamic(
            horizontal_nm,
            vertical_ft,
            next_horizontal_nm=(
                next_horizontal_nm
            ),
            next_vertical_ft=(
                next_vertical_ft
            ),
            state_rules=(
                SAFETY_STATE_RULES
            ),
        )

    kwargs = dict(
        min_horizontal_nm=(
            MIN_HORIZONTAL_SEPARATION_NM
        ),
        min_vertical_ft=(
            MIN_VERTICAL_SEPARATION_FT
        ),
        warning_horizontal_nm=(
            active_warning_horizontal_nm()
        ),
        warning_vertical_ft=(
            active_warning_vertical_ft()
        ),
    )

    try:
        return classify_pair(
            horizontal_nm,
            vertical_ft,
            state_rules=(
                SAFETY_STATE_RULES
            ),
            **kwargs,
        )
    except TypeError:
        return classify_pair(
            horizontal_nm,
            vertical_ft,
            **kwargs,
        )


def combine_state(a: str, b: str) -> str:
    return b if STATE_PRIORITY.get(b, -1) > STATE_PRIORITY.get(a, -1) else a


# =============================================================================
# GENERIC HELPERS
# =============================================================================

MONTHLY_FILE_RE = re.compile(r"^(?P<name>.+)_(?P<year>\d{4})-(?P<month>\d{2})\.csv$")


def parse_day_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return numeric
        magnitude = float(finite.abs().median())
        if magnitude > 1e17:
            return numeric / 1_000_000_000.0
        if magnitude > 1e14:
            return numeric / 1_000_000.0
        if magnitude > 1e11:
            return numeric / 1_000.0
        return numeric

    values = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return values.sub(epoch).dt.total_seconds().astype("float64")


def row_numeric(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = row.get(name, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def ground_speed_ms(row: pd.Series) -> float | None:
    value = row_numeric(row, "velocity")
    if value is not None:
        return value
    value_kt = row_numeric(row, "velocity_kts")
    return None if value_kt is None else value_kt / MPS_TO_KT


def track_deg(row: pd.Series) -> float | None:
    return row_numeric(row, "true_track", "heading")


def vertical_rate_ms(row: pd.Series) -> float | None:
    value = row_numeric(row, "vertical_rate", "vertrate")
    if value is not None:
        return value
    value_fpm = row_numeric(row, "vertical_rate_fpm")
    return None if value_fpm is None else value_fpm / MPS_TO_FPM


def altitude_ft(row: pd.Series) -> float | None:
    value = row_numeric(row, "baroaltitude")
    return None if value is None else value * 3.2808398950131


def callsign_of(row: pd.Series) -> str:
    value = str(row.get("callsign", "") or "").strip()
    return value if value else str(row.get("icao24", "") or "").upper()


def is_true_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def load_analysis_area(area_key: str | None) -> dict | None:
    if not area_key:
        return None
    presets = load_area_presets(BBOX_PRESETS_FILE)
    return resolve_area_preset(area_key, presets)


# =============================================================================
# FAST PAIR SCREENING / DETAILED LOOK-AHEAD
# =============================================================================



def continuous_tcpa_seconds_local(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
    max_seconds: float,
) -> float | None:
    lat_a = row_numeric(aircraft_a, "lat", "latitude")
    lon_a = row_numeric(aircraft_a, "lon", "longitude")
    lat_b = row_numeric(aircraft_b, "lat", "latitude")
    lon_b = row_numeric(aircraft_b, "lon", "longitude")
    speed_a = ground_speed_ms(aircraft_a)
    speed_b = ground_speed_ms(aircraft_b)
    track_a = track_deg(aircraft_a)
    track_b = track_deg(aircraft_b)

    if None in (lat_a, lon_a, lat_b, lon_b, speed_a, speed_b, track_a, track_b):
        return None

    azimuth_ab, _, distance_m = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    az = math.radians(azimuth_ab)
    rx = distance_m * math.sin(az)
    ry = distance_m * math.cos(az)

    ta = math.radians(track_a)
    tb = math.radians(track_b)
    vax = speed_a * math.sin(ta)
    vay = speed_a * math.cos(ta)
    vbx = speed_b * math.sin(tb)
    vby = speed_b * math.cos(tb)
    rvx = vbx - vax
    rvy = vby - vay

    denom = rvx * rvx + rvy * rvy
    if denom <= 1e-9:
        return 0.0

    tcpa = -((rx * rvx) + (ry * rvy)) / denom
    return float(min(max(tcpa, 0.0), float(max_seconds)))


def snapshot_unique(snapshot: pd.DataFrame) -> pd.DataFrame:
    snapshot = snapshot.copy()
    if "lastcontact" in snapshot.columns:
        snapshot["_lastcontact_numeric"] = pd.to_numeric(
            snapshot["lastcontact"], errors="coerce"
        )
        snapshot.sort_values(
            ["icao24", "_lastcontact_numeric"],
            inplace=True,
            kind="mergesort",
        )
    snapshot.drop_duplicates(subset=["icao24"], keep="last", inplace=True)
    snapshot.drop(columns=["_lastcontact_numeric"], errors="ignore", inplace=True)
    snapshot.reset_index(drop=True, inplace=True)
    return snapshot


def evaluate_candidate_pair(
    index_a: int,
    aircraft_a: pd.Series,
    index_b: int,
    aircraft_b: pd.Series,
    current_h_nm: float,
    current_v_ft: float | None,
    current_state: str,
    tcpa_s: float | None,
    cpa_nm: float,
    cpa_v_ft: float | None,
) -> dict | None:
    intervals = [
        (0.0, 60.0),
        (60.0, 120.0),
        (120.0, 300.0),
    ]

    interval_states = [
        STATE_NORMAL,
        STATE_NORMAL,
        STATE_NORMAL,
    ]

    min_h = [
        math.inf,
        math.inf,
        math.inf,
    ]

    min_v = [
        math.inf,
        math.inf,
        math.inf,
    ]

    min_h_time = [
        None,
        None,
        None,
    ]

    min_v_time = [
        None,
        None,
        None,
    ]

    first_state_time = {
        STATE_PROXIMITY: None,
        STATE_HIGH_RISK: None,
        STATE_CONFLICT: None,
    }

    for time_s in np.arange(
        0.0,
        LOOKAHEAD_SECONDS
        + PREDICTION_STEP_SECONDS
        * 0.5,
        PREDICTION_STEP_SECONDS,
        dtype=float,
    ):
        if time_s == 0.0:
            h_nm = (
                current_h_nm
            )

            v_ft = (
                current_v_ft
            )

            state = (
                current_state
            )

        else:
            (
                h_nm,
                v_ft,
            ) = pair_separation_at_time(
                aircraft_a,
                aircraft_b,
                float(
                    time_s
                ),
                vertical_rate_deadband_fpm=(
                    VERTICAL_RATE_DEADBAND_FPM
                ),
            )

            next_time_s = min(
                float(
                    time_s
                )
                + float(
                    PREDICTION_STEP_SECONDS
                ),
                float(
                    LOOKAHEAD_SECONDS
                ),
            )

            if next_time_s <= float(
                time_s
            ):
                next_h_nm = (
                    h_nm
                )

                next_v_ft = (
                    v_ft
                )
            else:
                (
                    next_h_nm,
                    next_v_ft,
                ) = pair_separation_at_time(
                    aircraft_a,
                    aircraft_b,
                    next_time_s,
                    vertical_rate_deadband_fpm=(
                        VERTICAL_RATE_DEADBAND_FPM
                    ),
                )

            state = classify_pair_active(
                h_nm,
                v_ft,
                next_horizontal_nm=(
                    next_h_nm
                ),
                next_vertical_ft=(
                    next_v_ft
                ),
            )

        if (
            state
            in first_state_time
            and first_state_time[
                state
            ]
            is None
        ):
            first_state_time[
                state
            ] = float(
                time_s
            )

        for k, (
            start_s,
            end_s,
        ) in enumerate(
            intervals
        ):
            if (
                start_s
                <= time_s
                <= end_s
            ):
                interval_states[
                    k
                ] = combine_state(
                    interval_states[
                        k
                    ],
                    state,
                )

                if h_nm < min_h[k]:
                    min_h[
                        k
                    ] = float(
                        h_nm
                    )

                    min_h_time[
                        k
                    ] = float(
                        time_s
                    )

                if (
                    v_ft is not None
                    and np.isfinite(
                        v_ft
                    )
                    and v_ft < min_v[k]
                ):
                    min_v[
                        k
                    ] = float(
                        v_ft
                    )

                    min_v_time[
                        k
                    ] = float(
                        time_s
                    )

    worst = (
        current_state
    )

    for state in interval_states:
        worst = combine_state(
            worst,
            state,
        )

    if worst == STATE_NORMAL:
        return None

    vertical_rule_ft = (
        vertical_separation_for_rule(
            current_v_ft,
            red_vertical_ft=(
                MIN_VERTICAL_SEPARATION_FT
            ),
            tolerance_ft=(
                VERTICAL_SEPARATION_TOLERANCE_FT
            ),
        )
    )

    return {
        "index_a": index_a,
        "index_b": index_b,
        "event_state": worst,
        "state_current": current_state,
        "worst_status_0_1min": (
            interval_states[0]
        ),
        "worst_status_1_2min": (
            interval_states[1]
        ),
        "worst_status_2_5min": (
            interval_states[2]
        ),
        "horizontal_sep_nm": (
            current_h_nm
        ),
        "vertical_sep_ft": (
            current_v_ft
        ),
        "vertical_sep_rule_ft": (
            vertical_rule_ft
        ),
        "cpa_nm": cpa_nm,
        "tcpa_s": (
            0.0
            if tcpa_s is None
            else tcpa_s
        ),
        "vertical_sep_at_cpa_ft": (
            cpa_v_ft
        ),
        "min_horizontal_0_1min_nm": (
            None
            if math.isinf(
                min_h[0]
            )
            else min_h[0]
        ),
        "time_to_min_horizontal_0_1min_s": (
            min_h_time[0]
        ),
        "min_vertical_0_1min_ft": (
            None
            if math.isinf(
                min_v[0]
            )
            else min_v[0]
        ),
        "time_to_min_vertical_0_1min_s": (
            min_v_time[0]
        ),
        "min_horizontal_1_2min_nm": (
            None
            if math.isinf(
                min_h[1]
            )
            else min_h[1]
        ),
        "time_to_min_horizontal_1_2min_s": (
            min_h_time[1]
        ),
        "min_vertical_1_2min_ft": (
            None
            if math.isinf(
                min_v[1]
            )
            else min_v[1]
        ),
        "time_to_min_vertical_1_2min_s": (
            min_v_time[1]
        ),
        "min_horizontal_2_5min_nm": (
            None
            if math.isinf(
                min_h[2]
            )
            else min_h[2]
        ),
        "time_to_min_horizontal_2_5min_s": (
            min_h_time[2]
        ),
        "min_vertical_2_5min_ft": (
            None
            if math.isinf(
                min_v[2]
            )
            else min_v[2]
        ),
        "time_to_min_vertical_2_5min_s": (
            min_v_time[2]
        ),
        "first_proximity_s": (
            first_state_time[
                STATE_PROXIMITY
            ]
        ),
        "first_high_risk_s": (
            first_state_time[
                STATE_HIGH_RISK
            ]
        ),
        "first_conflict_s": (
            first_state_time[
                STATE_CONFLICT
            ]
        ),
        "predicted_conflict": any(
            state == STATE_CONFLICT
            for state
            in interval_states
        ),
    }


def analyse_snapshot_fast(snapshot: pd.DataFrame) -> tuple[list[dict], dict]:
    snapshot = snapshot_unique(snapshot)
    rows = list(snapshot.iterrows())
    warning_h = active_warning_horizontal_nm()

    stats = {
        "pairs_total": 0,
        "pairs_distance_pruned": 0,
        "pairs_cpa_pruned": 0,
        "pairs_detailed": 0,
        "pairs_missing_vertical": 0,
    }
    events = []

    for i in range(len(rows)):
        index_a, aircraft_a = rows[i]
        for j in range(i + 1, len(rows)):
            index_b, aircraft_b = rows[j]
            stats["pairs_total"] += 1

            h_nm, v_ft = pair_separation_at_time(
                aircraft_a,
                aircraft_b,
                0.0,
                vertical_rate_deadband_fpm=(
                    VERTICAL_RATE_DEADBAND_FPM
                ),
            )

            if v_ft is None:
                stats["pairs_missing_vertical"] += 1

            (
                next_h_nm,
                next_v_ft,
            ) = pair_separation_at_time(
                aircraft_a,
                aircraft_b,
                float(
                    PREDICTION_STEP_SECONDS
                ),
                vertical_rate_deadband_fpm=(
                    VERTICAL_RATE_DEADBAND_FPM
                ),
            )

            current_state = classify_pair_active(
                h_nm,
                v_ft,
                next_horizontal_nm=(
                    next_h_nm
                ),
                next_vertical_ft=(
                    next_v_ft
                ),
            )

            speed_a_ms = ground_speed_ms(aircraft_a)
            speed_b_ms = ground_speed_ms(aircraft_b)

            # Conservative bound: if even two aircraft turning directly toward
            # one another could not enter the outer warning radius within five
            # minutes, no detailed prediction is required.
            if (
                ENABLE_DISTANCE_PREFILTER
                and current_state == STATE_NORMAL
                and speed_a_ms is not None
                and speed_b_ms is not None
            ):
                max_closure_nm = (
                    (speed_a_ms + speed_b_ms)
                    * MPS_TO_KT
                    * (LOOKAHEAD_SECONDS / 3600.0)
                )
                if h_nm > warning_h + max_closure_nm:
                    stats["pairs_distance_pruned"] += 1
                    continue

            tcpa = continuous_tcpa_seconds_local(
                aircraft_a,
                aircraft_b,
                LOOKAHEAD_SECONDS,
            )
            cpa_nm = h_nm
            cpa_v_ft = v_ft
            if tcpa is not None:
                cpa_nm, cpa_v_ft = pair_separation_at_time(
                    aircraft_a,
                    aircraft_b,
                    tcpa,
                    vertical_rate_deadband_fpm=(
                        VERTICAL_RATE_DEADBAND_FPM
                    ),
                )

            # CPA is based on a local tangent-plane model. The extra 1 NM
            # margin prevents borderline pairs from being discarded cheaply.
            if (
                ENABLE_CPA_PREFILTER
                and current_state == STATE_NORMAL
                and h_nm > warning_h
                and tcpa is not None
                and cpa_nm > warning_h + CPA_PREFILTER_MARGIN_NM
            ):
                stats["pairs_cpa_pruned"] += 1
                continue

            stats["pairs_detailed"] += 1
            event = evaluate_candidate_pair(
                index_a,
                aircraft_a,
                index_b,
                aircraft_b,
                h_nm,
                v_ft,
                current_state,
                tcpa,
                cpa_nm,
                cpa_v_ft,
            )
            if event is not None:
                events.append(event)

    return events, stats


# =============================================================================
# OUTPUTS
# =============================================================================

EVENT_COLUMNS = [
    "sector",
    "timestamp",
    "timestamp_utc",
    "event_state",
    "event_colour",
    "state_current",
    "worst_status_0_1min",
    "worst_status_1_2min",
    "worst_status_2_5min",
    "icao24_a",
    "callsign_a",
    "lat_a",
    "lon_a",
    "baro_altitude_ft_a",
    "ground_speed_kt_a",
    "true_track_deg_a",
    "vertical_rate_fpm_a",
    "state_source_a",
    "exact_grid_row_a",
    "state_age_s_a",
    "position_age_s_a",
    "icao24_b",
    "callsign_b",
    "lat_b",
    "lon_b",
    "baro_altitude_ft_b",
    "ground_speed_kt_b",
    "true_track_deg_b",
    "vertical_rate_fpm_b",
    "state_source_b",
    "exact_grid_row_b",
    "state_age_s_b",
    "position_age_s_b",
    "horizontal_sep_nm",
    "vertical_sep_ft",
    "vertical_sep_rule_ft",
    "cpa_nm",
    "tcpa_s",
    "vertical_sep_at_cpa_ft",
    "min_horizontal_0_1min_nm",
    "time_to_min_horizontal_0_1min_s",
    "min_vertical_0_1min_ft",
    "time_to_min_vertical_0_1min_s",
    "min_horizontal_1_2min_nm",
    "time_to_min_horizontal_1_2min_s",
    "min_vertical_1_2min_ft",
    "time_to_min_vertical_1_2min_s",
    "min_horizontal_2_5min_nm",
    "time_to_min_horizontal_2_5min_s",
    "min_vertical_2_5min_ft",
    "time_to_min_vertical_2_5min_s",
    "first_proximity_s",
    "first_high_risk_s",
    "first_conflict_s",
    "predicted_conflict",
]


def build_event_row(
    area_key: str,
    timestamp: float,
    snapshot: pd.DataFrame,
    event: dict,
) -> dict:
    a = snapshot.loc[event["index_a"]]
    b = snapshot.loc[event["index_b"]]

    speed_a = ground_speed_ms(a)
    speed_b = ground_speed_ms(b)
    vr_a = vertical_rate_ms(a)
    vr_b = vertical_rate_ms(b)

    row = {
        "sector": area_key,
        "timestamp": timestamp,
        "timestamp_utc": pd.to_datetime(timestamp, unit="s", utc=True).isoformat(),
        "event_state": event["event_state"],
        "event_colour": STATE_COLOURS[event["event_state"]],
        "icao24_a": str(a.get("icao24", "") or ""),
        "callsign_a": callsign_of(a),
        "lat_a": row_numeric(a, "lat"),
        "lon_a": row_numeric(a, "lon"),
        "baro_altitude_ft_a": altitude_ft(a),
        "ground_speed_kt_a": None if speed_a is None else speed_a * MPS_TO_KT,
        "true_track_deg_a": track_deg(a),
        "vertical_rate_fpm_a": None if vr_a is None else vr_a * MPS_TO_FPM,
        "state_source_a": a.get("state_source", ""),
        "exact_grid_row_a": a.get("exact_grid_row", ""),
        "state_age_s_a": row_numeric(a, "state_age_s"),
        "position_age_s_a": row_numeric(a, "position_age_s"),
        "icao24_b": str(b.get("icao24", "") or ""),
        "callsign_b": callsign_of(b),
        "lat_b": row_numeric(b, "lat"),
        "lon_b": row_numeric(b, "lon"),
        "baro_altitude_ft_b": altitude_ft(b),
        "ground_speed_kt_b": None if speed_b is None else speed_b * MPS_TO_KT,
        "true_track_deg_b": track_deg(b),
        "vertical_rate_fpm_b": None if vr_b is None else vr_b * MPS_TO_FPM,
        "state_source_b": b.get("state_source", ""),
        "exact_grid_row_b": b.get("exact_grid_row", ""),
        "state_age_s_b": row_numeric(b, "state_age_s"),
        "position_age_s_b": row_numeric(b, "position_age_s"),
    }
    row.update({key: value for key, value in event.items() if not key.startswith("index_")})
    return row



# =============================================================================
# PROCESSED DATASET INPUT
# =============================================================================

PROCESSED_FILE_RE = re.compile(r"^.+_(\d{4})-(\d{2})_10s\.csv$")


def parse_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> set[tuple[int, int]]:
    months = set()
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while current <= end_month:
        months.add((current.year, current.month))
        current = current + pd.DateOffset(months=1)
    return months


def input_files(area: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    directory = INPUT_ROOT / area
    if not directory.exists():
        raise FileNotFoundError(f"Missing processed data directory: {directory}")
    requested = months_between(start, end)
    files = []
    for path in directory.rglob("*_10s.csv"):
        if "_daily_cache" in path.parts:
            continue
        match = PROCESSED_FILE_RE.match(path.name)
        if match and (int(match[1]), int(match[2])) in requested:
            files.append(path)
    files.sort()
    missing = requested - {
        (int(PROCESSED_FILE_RE.match(path.name)[1]),
         int(PROCESSED_FILE_RE.match(path.name)[2]))
        for path in files
    }
    if missing:
        raise FileNotFoundError(
            "Missing processed monthly CSVs for: " +
            ", ".join(f"{y:04d}-{m:02d}" for y, m in sorted(missing))
        )
    return files


def iter_10s_snapshots(files: list[Path], start: float, end: float,
                       chunksize: int) -> Iterator[tuple[float, pd.DataFrame]]:
    """Stream chronological monthly CSVs, carrying the last timestamp between chunks."""
    previous_timestamp = float('-inf')
    for path in files:
        carry = pd.DataFrame()
        for raw in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            if "time" not in raw.columns or "icao24" not in raw.columns:
                raise ValueError(f"Missing time/icao24 fields in {path}")
            raw = raw.copy()
            raw["time"] = to_epoch_seconds(raw["time"])
            # Read all rows within one hour in roughly chronological order;
            # the preprocessor always serialises each day by time.
            raw = raw.loc[raw["time"].notna() & np.isfinite(raw["time"])].copy()
            if not carry.empty:
                raw = pd.concat([carry, raw], ignore_index=True)
            if raw.empty:
                carry = pd.DataFrame()
                continue
            raw.sort_values(["time", "icao24"], inplace=True, kind="mergesort")
            end_of_chunk = float(raw["time"].max())
            complete = raw.loc[raw["time"] < end_of_chunk]
            carry = raw.loc[raw["time"] == end_of_chunk].copy()
            if not complete.empty:
                for stamp, snapshot in complete.groupby("time", sort=True):
                    stamp = float(stamp)
                    if stamp < previous_timestamp:
                        raise ValueError(f"Processed input is not chronological: {path}")
                    previous_timestamp = stamp
                    if start <= stamp <= end:
                        yield stamp, snapshot_unique(snapshot)
        if not carry.empty:
            for stamp, snapshot in carry.groupby("time", sort=True):
                stamp = float(stamp)
                if stamp < previous_timestamp:
                    raise ValueError(f"Processed input is not chronological: {path}")
                previous_timestamp = stamp
                if start <= stamp <= end:
                    yield stamp, snapshot_unique(snapshot)


def run_output_paths(area: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    directory = (EVENT_OUTPUT_ROOT / area / "runs" /
                 f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}")
    files = {
        STATE_PROXIMITY: directory / "yellow_proximity_events.csv",
        STATE_HIGH_RISK: directory / "orange_high_risk_events.csv",
        STATE_CONFLICT: directory / "red_loss_of_separation_events.csv",
    }
    return {"directory": directory, **files}


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSky separation detector on reusable 10-second CSVs")
    parser.add_argument("--area", default="LECMSAN")
    parser.add_argument("--start", required=True, help="UTC inclusive, e.g. '2019-01-01 00:00:00'")
    parser.add_argument("--end", required=True, help="UTC inclusive, e.g. '2019-01-15 23:59:50'")
    parser.add_argument("--chunksize", type=int, default=READ_CHUNK_ROWS)
    args = parser.parse_args()
    start, end = parse_utc(args.start), parse_utc(args.end)
    if start > end or args.chunksize <= 0:
        parser.error("End must follow start, and chunksize must be positive")
    files = input_files(args.area, start, end)
    paths = run_output_paths(args.area, start, end)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    total = dict(snapshots=0, pairs_total=0, pairs_distance_pruned=0,
                 pairs_cpa_pruned=0, pairs_detailed=0,
                 **{STATE_PROXIMITY: 0, STATE_HIGH_RISK: 0, STATE_CONFLICT: 0})
    day = None
    today = {state: 0 for state in ACTIVE_EVENT_STATES}
    daily_log = []
    handles, writers = {}, {}
    began = time.monotonic()
    print("=" * 78)
    print("OpenSky separation detector - processed 10-second dataset")
    print(f"Area: {args.area} | Start UTC: {start} | End UTC: {end} (inclusive)")
    print(f"Input files: {len(files)} | Output: {paths['directory']}")
    print(f"Prediction horizons: 1, 2, 5 min | Step: {PREDICTION_STEP_SECONDS}s")
    print(
        f"Safety: H < {MIN_HORIZONTAL_SEPARATION_NM} NM AND "
        f"V < {MIN_VERTICAL_SEPARATION_FT} ft "
        f"(vertical tolerance +/-{VERTICAL_SEPARATION_TOLERANCE_FT} ft)"
    )
    print(
        f"Vertical-rate deadband: +/-{VERTICAL_RATE_DEADBAND_FPM} fpm"
    )
    print("=" * 78, flush=True)
    try:
        for state in ACTIVE_EVENT_STATES:
            handle = paths[state].open("w", encoding="utf-8", newline="")
            handles[state] = handle
            writers[state] = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS,
                                            extrasaction="ignore")
            writers[state].writeheader()
        for timestamp, snapshot in iter_10s_snapshots(
            files, start.timestamp(), end.timestamp(), args.chunksize
        ):
            dt = pd.to_datetime(timestamp, unit="s", utc=True)
            today_label = dt.strftime("%Y-%m-%d")
            if day != today_label:
                if day is not None:
                    print(f"DAY COMPLETE: {day} | " + " | ".join(
                        f"{STATE_COLOURS[s].lower()}={today.get(s, 0):,}"
                        for s in ACTIVE_EVENT_STATES), flush=True)
                    daily_log.append({"day": day, **{STATE_COLOURS[s].lower(): today.get(s,0)
                                                      for s in ACTIVE_EVENT_STATES}})
                day = today_label
                today = {state: 0 for state in ACTIVE_EVENT_STATES}
                print(f"DAY START: {day} UTC", flush=True)

            total["snapshots"] += 1
            if len(snapshot) >= 2:
                events, stats = analyse_snapshot_fast(snapshot)
                for key in ("pairs_total", "pairs_distance_pruned",
                            "pairs_cpa_pruned", "pairs_detailed"):
                    total[key] += stats[key]
                for event in events:
                    state = event["event_state"]
                    if state in writers:
                        writers[state].writerow(build_event_row(args.area, timestamp,
                                                                  snapshot, event))
                        total[state] += 1
                        today[state] += 1
            if dt.minute == 0 and dt.second == 0:
                print(
                    f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                    f"snapshots={total['snapshots']:,} | pairs={total['pairs_total']:,} | "
                    f"screened={total['pairs_distance_pruned'] + total['pairs_cpa_pruned']:,} | "
                    f"detailed={total['pairs_detailed']:,} | "
                    f"day yellow={today.get(STATE_PROXIMITY, 0):,} "
                    f"orange={today.get(STATE_HIGH_RISK, 0):,} "
                    f"red={today.get(STATE_CONFLICT, 0):,} | "
                    f"total yellow={total[STATE_PROXIMITY]:,} "
                    f"orange={total[STATE_HIGH_RISK]:,} "
                    f"red={total[STATE_CONFLICT]:,}",
                    flush=True,
                )
        if day is not None:
            print(f"DAY COMPLETE: {day}", flush=True)
            daily_log.append({"day": day, **{STATE_COLOURS[s].lower(): today.get(s,0)
                                              for s in ACTIVE_EVENT_STATES}})
    finally:
        for handle in handles.values():
            handle.close()
    elapsed = time.monotonic() - began
    summary = {
        "area": args.area, "start_utc": start.isoformat(),
        "end_utc_inclusive": end.isoformat(), "elapsed_s": elapsed, **total
    }
    pd.DataFrame([summary]).to_csv(paths["directory"] / "run_summary.csv", index=False)
    pd.DataFrame(daily_log).to_csv(paths["directory"] / "daily_summary.csv", index=False)
    print("=" * 78)
    print(f"RUN COMPLETE | elapsed={elapsed:.1f}s | snapshots={total['snapshots']:,}")
    print(f"Yellow: {total[STATE_PROXIMITY]:,} | Orange: {total[STATE_HIGH_RISK]:,} "
          f"| Red: {total[STATE_CONFLICT]:,}")
    print(f"Output: {paths['directory']}", flush=True)


if __name__ == "__main__":
    main()
