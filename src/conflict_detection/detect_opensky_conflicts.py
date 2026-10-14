#!/usr/bin/env python3
"""
Detect OpenSky separation events in archived sector CSV files.

The script reads monthly state-vector CSV files created under:

    data/raw/opensky_sectors/<BBOX_PRESET>/year=YYYY/month=MM/

and analyses every timestamp/snapshot using the same four-state safety logic
implemented in opensky_safety.py:

    NORMAL      -> green
    PROXIMITY   -> yellow
    HIGH_RISK   -> orange
    CONFLICT    -> red / loss of separation

For every monthly input CSV, three derived CSV files are written under:

    data/processed/opensky_separation/<BBOX_PRESET>/year=YYYY/month=MM/

with one row per relevant aircraft pair / timestamp:

    *_separation_proximity_events.csv
    *_separation_high_risk_events.csv
    *_loss_of_separation_events.csv

The output category is the WORST state reached by the pair either now or in
the configured look-ahead intervals (default: 0-1, 1-2 and 2-5 minutes).

The output row still contains the complete state timeline so it is possible to
distinguish:
    - a current problem,
    - a problem predicted in 0-1 min,
    - a problem predicted in 1-2 min,
    - a problem predicted in 2-5 min.

The analysis uses barometric altitude for vertical separation.

Master's Thesis:
AI-Based Air Traffic Management
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from opensky_safety import (
    STATE_NORMAL,
    STATE_PROXIMITY,
    STATE_HIGH_RISK,
    STATE_CONFLICT,
    STATE_PRIORITY,
    analyse_aircraft_snapshot,
    classify_pair,
    pair_separation_at_time,
    load_safety_profile,
    enabled_states,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "opensky_sectors"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opensky_separation"
)

SEPARATION_PROFILES_FILE = Path(__file__).with_name(
    "separation_profiles.json"
)


# =============================================================================
# DEFAULT ANALYSIS CONFIGURATION
# =============================================================================

DEFAULT_SECTOR = "GALICIA"

SEPARATION_PROFILE = "ENROUTE"

# Experimental warning envelope used by the current thesis configuration.
WARNING_FACTOR = 1.50

# Three prediction intervals:
#   0-1 min
#   1-2 min
#   2-5 min
LOOKAHEAD_MINUTES = (1, 2, 5)

# Evaluate the complete look-ahead trajectory every 5 s.
PREDICTION_STEP_SECONDS = 5

# Ground aircraft are not part of en-route separation analysis.
CHECK_GROUND_AIRCRAFT = False

# Defensive cruise-band filter. The archive downloader can already apply this,
# but keeping it here guarantees that this detector analyses the intended band.
FILTER_CRUISE_FL_BAND = True
CRUISE_FL_MIN = 290
CRUISE_FL_MAX = 410

# CSV streaming. Increase on machines with more RAM.
READ_CHUNK_ROWS = 250_000

# Re-running the detector is safe:
# False -> skip a month if all three output files already exist.
# True  -> regenerate that month's outputs.
OVERWRITE_EXISTING = False

# A monthly archive produced by download_opensky_sector_archive.py is expected
# to be chronological by query window. Rows inside each read chunk are sorted
# before grouping. If a later chunk contains timestamps older than an already
# emitted snapshot, the script raises an error rather than silently producing
# incomplete pairwise comparisons.
REQUIRE_CHRONOLOGICAL_INPUT = True


# =============================================================================
# OUTPUT NAMES
# =============================================================================

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
# OUTPUT COLUMNS
# =============================================================================

OUTPUT_COLUMNS = [
    "sector",
    "timestamp",
    "timestamp_utc",

    "event_state",
    "event_colour",
    "problem_description",
    "event_timeline",
    "first_alert_window",
    "first_worst_state_window",

    "alert_current",
    "alert_0_1min",
    "alert_1_2min",
    "alert_2_5min",

    "state_current",
    "worst_status_0_1min",
    "worst_status_1_2min",
    "worst_status_2_5min",

    "icao24_a",
    "callsign_a",
    "lat_a",
    "lon_a",
    "baro_altitude_ft_a",
    "ground_speed_ms_a",
    "ground_speed_kt_a",
    "true_track_deg_a",
    "vertical_rate_ms_a",
    "vertical_rate_fpm_a",

    "icao24_b",
    "callsign_b",
    "lat_b",
    "lon_b",
    "baro_altitude_ft_b",
    "ground_speed_ms_b",
    "ground_speed_kt_b",
    "true_track_deg_b",
    "vertical_rate_ms_b",
    "vertical_rate_fpm_b",

    "relative_horizontal_speed_kt",
    "horizontal_closing_rate_kt",
    "vertical_closing_rate_fpm",

    "horizontal_sep_nm",
    "vertical_sep_ft",

    "cpa_nm",
    "tcpa_s",
    "vertical_sep_at_cpa_ft",

    "min_horizontal_0_1min_nm",
    "time_to_min_horizontal_0_1min_s",
    "vertical_sep_at_min_horizontal_0_1min_ft",
    "min_vertical_0_1min_ft",
    "time_to_min_vertical_0_1min_s",
    "horizontal_sep_at_min_vertical_0_1min_nm",

    "min_horizontal_1_2min_nm",
    "time_to_min_horizontal_1_2min_s",
    "vertical_sep_at_min_horizontal_1_2min_ft",
    "min_vertical_1_2min_ft",
    "time_to_min_vertical_1_2min_s",
    "horizontal_sep_at_min_vertical_1_2min_nm",

    "min_horizontal_2_5min_nm",
    "time_to_min_horizontal_2_5min_s",
    "vertical_sep_at_min_horizontal_2_5min_ft",
    "min_vertical_2_5min_ft",
    "time_to_min_vertical_2_5min_s",
    "horizontal_sep_at_min_vertical_2_5min_nm",

    "min_horizontal_0_5min_nm",
    "time_to_min_horizontal_0_5min_s",
    "timestamp_min_horizontal_0_5min_utc",
    "vertical_sep_at_min_horizontal_0_5min_ft",

    "min_vertical_0_5min_ft",
    "time_to_min_vertical_0_5min_s",
    "timestamp_min_vertical_0_5min_utc",
    "horizontal_sep_at_min_vertical_0_5min_nm",

    "first_proximity_s",
    "first_proximity_timestamp_utc",
    "first_high_risk_s",
    "first_high_risk_timestamp_utc",
    "first_conflict_s",
    "first_conflict_timestamp_utc",

    "predicted_conflict",
]



# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

ACTIVE_SAFETY_PROFILE = load_safety_profile(
    SEPARATION_PROFILES_FILE,
    SEPARATION_PROFILE,
)

SAFETY_STATE_RULES = ACTIVE_SAFETY_PROFILE[
    "states"
]

ENABLED_SAFETY_STATES = enabled_states(
    SAFETY_STATE_RULES
)

RED_SEPARATION_RULE = SAFETY_STATE_RULES[
    STATE_CONFLICT
]

MIN_HORIZONTAL_SEPARATION_NM = float(
    RED_SEPARATION_RULE[
        "horizontal_nm"
    ]
)

MIN_VERTICAL_SEPARATION_FT = float(
    RED_SEPARATION_RULE[
        "vertical_ft"
    ]
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


# =============================================================================
# FILE DISCOVERY
# =============================================================================

MONTHLY_FILE_RE = re.compile(
    r"^(?P<name>.+)_(?P<year>\d{4})-(?P<month>\d{2})\.csv$"
)


def parse_month_key(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    match = re.fullmatch(
        r"(\d{4})-(\d{2})",
        value.strip(),
    )

    if not match:
        raise ValueError(
            "Month must use YYYY-MM format"
        )

    year = int(match.group(1))
    month = int(match.group(2))

    if not 1 <= month <= 12:
        raise ValueError(
            f"Invalid month: {month}"
        )

    return year, month


def month_tuple_to_int(value: tuple[int, int]) -> int:
    year, month = value
    return year * 12 + month


def discover_monthly_csvs(
    sector: str,
    start_month: tuple[int, int] | None,
    end_month: tuple[int, int] | None,
) -> list[tuple[Path, int, int]]:
    sector_root = INPUT_ROOT / sector

    if not sector_root.exists():
        raise FileNotFoundError(
            f"Sector archive folder not found:\n{sector_root}"
        )

    output = []

    start_value = (
        None
        if start_month is None
        else month_tuple_to_int(start_month)
    )

    end_value = (
        None
        if end_month is None
        else month_tuple_to_int(end_month)
    )

    for csv_file in sector_root.rglob("*.csv"):

        if "_window_cache" in csv_file.parts:
            continue

        match = MONTHLY_FILE_RE.match(
            csv_file.name
        )

        if not match:
            continue

        year = int(match.group("year"))
        month = int(match.group("month"))

        month_value = month_tuple_to_int(
            (year, month)
        )

        if (
            start_value is not None
            and month_value < start_value
        ):
            continue

        if (
            end_value is not None
            and month_value > end_value
        ):
            continue

        output.append(
            (csv_file, year, month)
        )

    output.sort(
        key=lambda item: (
            item[1],
            item[2],
            str(item[0]),
        )
    )

    return output


def build_output_paths(
    sector: str,
    year: int,
    month: int,
) -> dict[str, Path]:

    directory = (
        OUTPUT_ROOT
        / sector
        / f"year={year:04d}"
        / f"month={month:02d}"
    )

    base = (
        f"{sector.lower()}_"
        f"{year:04d}-{month:02d}"
    )

    return {
        "directory": directory,

        STATE_PROXIMITY:
            directory
            / f"{base}_separation_proximity_events.csv",

        STATE_HIGH_RISK:
            directory
            / f"{base}_separation_high_risk_events.csv",

        STATE_CONFLICT:
            directory
            / f"{base}_loss_of_separation_events.csv",
    }


# =============================================================================
# INPUT NORMALISATION
# =============================================================================

def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(
            np.nan,
            index=dataframe.index,
            dtype="float64",
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def prepare_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the subset needed for snapshot analysis.
    """
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.copy()

    required_columns = {
        "time",
        "icao24",
        "lat",
        "lon",
        "baroaltitude",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    dataframe["time"] = pd.to_numeric(
        dataframe["time"],
        errors="coerce",
    )

    dataframe["lat"] = pd.to_numeric(
        dataframe["lat"],
        errors="coerce",
    )

    dataframe["lon"] = pd.to_numeric(
        dataframe["lon"],
        errors="coerce",
    )

    dataframe["baroaltitude"] = pd.to_numeric(
        dataframe["baroaltitude"],
        errors="coerce",
    )

    valid = (
        dataframe["time"].notna()
        & np.isfinite(dataframe["time"])
        & dataframe["lat"].notna()
        & np.isfinite(dataframe["lat"])
        & dataframe["lon"].notna()
        & np.isfinite(dataframe["lon"])
    )

    dataframe = dataframe.loc[
        valid
    ].copy()

    if FILTER_CRUISE_FL_BAND:

        lower_m = (
            CRUISE_FL_MIN
            * 100.0
            * 0.3048
        )

        upper_m = (
            CRUISE_FL_MAX
            * 100.0
            * 0.3048
        )

        dataframe = dataframe.loc[
            dataframe["baroaltitude"].between(
                lower_m,
                upper_m,
                inclusive="both",
            )
        ].copy()

    if (
        not CHECK_GROUND_AIRCRAFT
        and "onground" in dataframe.columns
    ):
        onground = (
            dataframe["onground"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "1.0",
                    "yes",
                }
            )
        )

        dataframe = dataframe.loc[
            ~onground
        ].copy()

    dataframe.sort_values(
        ["time", "icao24"],
        inplace=True,
        kind="mergesort",
    )

    return dataframe


def prepare_snapshot(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure one state-vector record per aircraft at a timestamp.

    If duplicates exist for one icao24/time combination, keep the record with
    the newest last-contact value when available.
    """
    snapshot = snapshot.copy()

    if snapshot.empty:
        return snapshot

    if "lastcontact" in snapshot.columns:
        snapshot["_lastcontact_numeric"] = pd.to_numeric(
            snapshot["lastcontact"],
            errors="coerce",
        )

        snapshot.sort_values(
            [
                "icao24",
                "_lastcontact_numeric",
            ],
            inplace=True,
            kind="mergesort",
        )

    snapshot.drop_duplicates(
        subset=["icao24"],
        keep="last",
        inplace=True,
    )

    snapshot.drop(
        columns=["_lastcontact_numeric"],
        errors="ignore",
        inplace=True,
    )

    snapshot.reset_index(
        drop=True,
        inplace=True,
    )

    return snapshot


def iter_snapshots_from_month(
    csv_file: Path,
    chunksize: int,
) -> Iterator[tuple[float, pd.DataFrame]]:
    """
    Stream a large monthly CSV while preserving complete OpenSky hour partitions.

    The archive downloader appends Trino windows chronologically, but rows inside
    an individual Trino result are not assumed to be timestamp-sorted. Therefore
    this iterator buffers the final HOUR present in each pandas chunk, combines it
    with the next chunk, sorts the completed hour by timestamp, and only then
    emits complete snapshots.

    This is safer than carrying only the final timestamp because it cannot split
    one timestamp merely because Trino returned rows within an hour out of order.
    """
    carry = pd.DataFrame()
    last_emitted_hour: int | None = None

    for raw_chunk in pd.read_csv(
        csv_file,
        chunksize=chunksize,
        low_memory=False,
    ):

        chunk = prepare_rows(
            raw_chunk
        )

        if not carry.empty:
            chunk = pd.concat(
                [
                    carry,
                    chunk,
                ],
                ignore_index=True,
            )

        if chunk.empty:
            carry = pd.DataFrame()
            continue

        # Derive the OpenSky partition hour directly from Unix time. This avoids
        # depending on how pandas serialised the Trino `hour` column.
        chunk["_hour_key"] = (
            np.floor(
                chunk["time"].astype("float64")
                / 3600.0
            )
            .astype("int64")
        )

        minimum_hour = int(
            chunk["_hour_key"].min()
        )

        if (
            REQUIRE_CHRONOLOGICAL_INPUT
            and last_emitted_hour is not None
            and minimum_hour < last_emitted_hour
        ):
            raise ValueError(
                "The monthly archive moved backwards to an already processed "
                "OpenSky hour partition. Rebuild/sort the monthly archive before "
                "running the detector."
            )

        last_hour = int(
            chunk["_hour_key"].max()
        )

        complete = chunk.loc[
            chunk["_hour_key"] < last_hour
        ].copy()

        carry = chunk.loc[
            chunk["_hour_key"] == last_hour
        ].copy()

        if not complete.empty:
            complete.sort_values(
                ["time", "icao24"],
                inplace=True,
                kind="mergesort",
            )

            for hour_key, hour_frame in complete.groupby(
                "_hour_key",
                sort=True,
            ):
                hour_key = int(hour_key)

                if (
                    REQUIRE_CHRONOLOGICAL_INPUT
                    and last_emitted_hour is not None
                    and hour_key < last_emitted_hour
                ):
                    raise ValueError(
                        "OpenSky hour partitions are not chronological."
                    )

                for timestamp, snapshot in hour_frame.groupby(
                    "time",
                    sort=True,
                ):
                    snapshot = prepare_snapshot(
                        snapshot.drop(
                            columns=["_hour_key"],
                            errors="ignore",
                        )
                    )

                    yield float(timestamp), snapshot

                last_emitted_hour = hour_key

    if not carry.empty:
        carry.sort_values(
            ["time", "icao24"],
            inplace=True,
            kind="mergesort",
        )

        for hour_key, hour_frame in carry.groupby(
            "_hour_key",
            sort=True,
        ):
            hour_key = int(hour_key)

            if (
                REQUIRE_CHRONOLOGICAL_INPUT
                and last_emitted_hour is not None
                and hour_key < last_emitted_hour
            ):
                raise ValueError(
                    "OpenSky hour partitions are not chronological."
                )

            for timestamp, snapshot in hour_frame.groupby(
                "time",
                sort=True,
            ):
                snapshot = prepare_snapshot(
                    snapshot.drop(
                        columns=["_hour_key"],
                        errors="ignore",
                    )
                )

                yield float(timestamp), snapshot

            last_emitted_hour = hour_key



# =============================================================================
# PAIR KINEMATICS / LOOK-AHEAD METRICS
# =============================================================================

MPS_TO_KT = 1.9438444924406
MPS_TO_FPM = 196.8503937008


def row_numeric(
    row: pd.Series,
    *names: str,
) -> float | None:

    for name in names:
        value = row.get(
            name,
            np.nan,
        )

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if np.isfinite(
            value
        ):
            return value

    return None


def ground_speed_ms(
    row: pd.Series,
) -> float | None:

    value = row_numeric(
        row,
        "velocity",
    )

    if value is not None:
        return value

    value_kt = row_numeric(
        row,
        "velocity_kts",
    )

    if value_kt is not None:
        return (
            value_kt
            / MPS_TO_KT
        )

    return None


def track_deg(
    row: pd.Series,
) -> float | None:

    return row_numeric(
        row,
        "true_track",
        "heading",
    )


def vertical_rate_ms(
    row: pd.Series,
) -> float | None:

    value = row_numeric(
        row,
        "vertical_rate",
        "vertrate",
    )

    if value is not None:
        return value

    value_fpm = row_numeric(
        row,
        "vertical_rate_fpm",
    )

    if value_fpm is not None:
        return (
            value_fpm
            / MPS_TO_FPM
        )

    return None


def horizontal_relative_metrics(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
) -> tuple[float | None, float | None]:
    """
    Return:
        relative horizontal speed [kt]
        current horizontal closing rate [kt]

    Positive closing rate means the current horizontal distance is decreasing.
    Negative means the pair is opening.
    """
    lat_a = row_numeric(
        aircraft_a,
        "latitude",
        "lat",
    )
    lon_a = row_numeric(
        aircraft_a,
        "longitude",
        "lon",
    )
    lat_b = row_numeric(
        aircraft_b,
        "latitude",
        "lat",
    )
    lon_b = row_numeric(
        aircraft_b,
        "longitude",
        "lon",
    )

    speed_a = ground_speed_ms(
        aircraft_a
    )
    speed_b = ground_speed_ms(
        aircraft_b
    )

    track_a = track_deg(
        aircraft_a
    )
    track_b = track_deg(
        aircraft_b
    )

    if None in (
        lat_a,
        lon_a,
        lat_b,
        lon_b,
        speed_a,
        speed_b,
        track_a,
        track_b,
    ):
        return None, None

    from opensky_safety import GEOD

    azimuth_ab, _, distance_m = (
        GEOD.inv(
            lon_a,
            lat_a,
            lon_b,
            lat_b,
        )
    )

    azimuth_rad = math.radians(
        azimuth_ab
    )

    if distance_m <= 1e-9:
        unit_east = 0.0
        unit_north = 0.0
    else:
        unit_east = math.sin(
            azimuth_rad
        )
        unit_north = math.cos(
            azimuth_rad
        )

    track_a_rad = math.radians(
        track_a
    )
    track_b_rad = math.radians(
        track_b
    )

    velocity_a_east = (
        speed_a
        * math.sin(
            track_a_rad
        )
    )
    velocity_a_north = (
        speed_a
        * math.cos(
            track_a_rad
        )
    )

    velocity_b_east = (
        speed_b
        * math.sin(
            track_b_rad
        )
    )
    velocity_b_north = (
        speed_b
        * math.cos(
            track_b_rad
        )
    )

    relative_east = (
        velocity_b_east
        - velocity_a_east
    )
    relative_north = (
        velocity_b_north
        - velocity_a_north
    )

    relative_speed_ms = math.hypot(
        relative_east,
        relative_north,
    )

    range_rate_ms = (
        relative_east
        * unit_east
        +
        relative_north
        * unit_north
    )

    closing_rate_ms = (
        -range_rate_ms
    )

    return (
        relative_speed_ms
        * MPS_TO_KT,
        closing_rate_ms
        * MPS_TO_KT,
    )


def vertical_closing_rate_fpm(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
) -> float | None:
    """
    Positive means absolute vertical separation is decreasing.
    Negative means it is increasing.
    """
    altitude_a = row_altitude_ft(
        aircraft_a
    )
    altitude_b = row_altitude_ft(
        aircraft_b
    )

    rate_a_ms = vertical_rate_ms(
        aircraft_a
    )
    rate_b_ms = vertical_rate_ms(
        aircraft_b
    )

    if None in (
        altitude_a,
        altitude_b,
        rate_a_ms,
        rate_b_ms,
    ):
        return None

    delta_altitude_ft = (
        altitude_b
        - altitude_a
    )

    relative_vertical_fpm = (
        rate_b_ms
        - rate_a_ms
    ) * MPS_TO_FPM

    if abs(
        delta_altitude_ft
    ) <= 1e-9:
        return (
            -abs(
                relative_vertical_fpm
            )
        )

    derivative_absolute_separation = (
        math.copysign(
            1.0,
            delta_altitude_ft,
        )
        * relative_vertical_fpm
    )

    return (
        -derivative_absolute_separation
    )


def timestamp_from_offset(
    timestamp: float,
    offset_seconds: float | None,
) -> str:

    if (
        offset_seconds is None
        or not np.isfinite(
            offset_seconds
        )
    ):
        return ""

    return (
        pd.to_datetime(
            timestamp
            + float(
                offset_seconds
            ),
            unit="s",
            utc=True,
        )
        .isoformat()
    )


def prediction_sample_times(
    start_s: float,
    end_s: float,
) -> np.ndarray:

    times = np.arange(
        float(
            start_s
        ),
        float(
            end_s
        )
        + PREDICTION_STEP_SECONDS
        * 0.5,
        PREDICTION_STEP_SECONDS,
        dtype=float,
    )

    if (
        times.size == 0
        or times[-1] < end_s
    ):
        times = np.append(
            times,
            float(
                end_s
            ),
        )

    return times


def pair_lookahead_metrics(
    aircraft_a: pd.Series,
    aircraft_b: pd.Series,
    timestamp: float,
) -> dict:
    """
    Compute minimum-separation metrics with the same constant-kinematics
    projection and the same 5-second sampling used by the safety analysis.

    Minimum horizontal and minimum vertical separation are independent:
    they can occur at different times. Therefore the separation in the other
    dimension is recorded at each minimum.
    """
    intervals = [
        ("0_1min", 0.0, 60.0),
        ("1_2min", 60.0, 120.0),
        ("2_5min", 120.0, 300.0),
    ]

    warning_horizontal_nm = (
        MIN_HORIZONTAL_SEPARATION_NM
        * WARNING_FACTOR
    )

    warning_vertical_ft = (
        MIN_VERTICAL_SEPARATION_FT
        * WARNING_FACTOR
    )

    output = {}
    all_samples = []

    for (
        label,
        start_s,
        end_s,
    ) in intervals:

        samples = []

        for time_s in prediction_sample_times(
            start_s,
            end_s,
        ):

            horizontal_nm, vertical_ft = (
                pair_separation_at_time(
                    aircraft_a,
                    aircraft_b,
                    float(
                        time_s
                    ),
                )
            )

            state = classify_pair(
                horizontal_nm,
                vertical_ft,
                min_horizontal_nm=
                    MIN_HORIZONTAL_SEPARATION_NM,
                min_vertical_ft=
                    MIN_VERTICAL_SEPARATION_FT,
                warning_horizontal_nm=
                    warning_horizontal_nm,
                warning_vertical_ft=
                    warning_vertical_ft,
                state_rules=
                    SAFETY_STATE_RULES,
            )

            sample = {
                "time_s":
                    float(
                        time_s
                    ),
                "horizontal_nm":
                    float(
                        horizontal_nm
                    ),
                "vertical_ft":
                    (
                        None
                        if vertical_ft is None
                        else float(
                            vertical_ft
                        )
                    ),
                "state":
                    state,
            }

            samples.append(
                sample
            )
            all_samples.append(
                sample
            )

        horizontal_min_sample = min(
            samples,
            key=lambda item:
                item[
                    "horizontal_nm"
                ],
        )

        valid_vertical_samples = [
            item
            for item in samples
            if (
                item[
                    "vertical_ft"
                ]
                is not None
                and np.isfinite(
                    item[
                        "vertical_ft"
                    ]
                )
            )
        ]

        vertical_min_sample = (
            min(
                valid_vertical_samples,
                key=lambda item:
                    item[
                        "vertical_ft"
                    ],
            )
            if valid_vertical_samples
            else None
        )

        output[
            f"min_horizontal_{label}_nm"
        ] = (
            horizontal_min_sample[
                "horizontal_nm"
            ]
        )

        output[
            f"time_to_min_horizontal_{label}_s"
        ] = (
            horizontal_min_sample[
                "time_s"
            ]
        )

        output[
            f"vertical_sep_at_min_horizontal_{label}_ft"
        ] = (
            horizontal_min_sample[
                "vertical_ft"
            ]
        )

        output[
            f"min_vertical_{label}_ft"
        ] = (
            None
            if vertical_min_sample is None
            else vertical_min_sample[
                "vertical_ft"
            ]
        )

        output[
            f"time_to_min_vertical_{label}_s"
        ] = (
            None
            if vertical_min_sample is None
            else vertical_min_sample[
                "time_s"
            ]
        )

        output[
            f"horizontal_sep_at_min_vertical_{label}_nm"
        ] = (
            None
            if vertical_min_sample is None
            else vertical_min_sample[
                "horizontal_nm"
            ]
        )

    unique_samples = {
        sample["time_s"]:
            sample
        for sample in all_samples
    }

    ordered_samples = [
        unique_samples[
            time_s
        ]
        for time_s in sorted(
            unique_samples
        )
    ]

    horizontal_global = min(
        ordered_samples,
        key=lambda item:
            item[
                "horizontal_nm"
            ],
    )

    valid_vertical_global = [
        item
        for item in ordered_samples
        if (
            item[
                "vertical_ft"
            ]
            is not None
            and np.isfinite(
                item[
                    "vertical_ft"
                ]
            )
        )
    ]

    vertical_global = (
        min(
            valid_vertical_global,
            key=lambda item:
                item[
                    "vertical_ft"
                ],
        )
        if valid_vertical_global
        else None
    )

    output[
        "min_horizontal_0_5min_nm"
    ] = (
        horizontal_global[
            "horizontal_nm"
        ]
    )

    output[
        "time_to_min_horizontal_0_5min_s"
    ] = (
        horizontal_global[
            "time_s"
        ]
    )

    output[
        "timestamp_min_horizontal_0_5min_utc"
    ] = timestamp_from_offset(
        timestamp,
        horizontal_global[
            "time_s"
        ],
    )

    output[
        "vertical_sep_at_min_horizontal_0_5min_ft"
    ] = (
        horizontal_global[
            "vertical_ft"
        ]
    )

    output[
        "min_vertical_0_5min_ft"
    ] = (
        None
        if vertical_global is None
        else vertical_global[
            "vertical_ft"
        ]
    )

    output[
        "time_to_min_vertical_0_5min_s"
    ] = (
        None
        if vertical_global is None
        else vertical_global[
            "time_s"
        ]
    )

    output[
        "timestamp_min_vertical_0_5min_utc"
    ] = timestamp_from_offset(
        timestamp,
        (
            None
            if vertical_global is None
            else vertical_global[
                "time_s"
            ]
        ),
    )

    output[
        "horizontal_sep_at_min_vertical_0_5min_nm"
    ] = (
        None
        if vertical_global is None
        else vertical_global[
            "horizontal_nm"
        ]
    )

    for (
        target_state,
        field_prefix,
    ) in (
        (
            STATE_PROXIMITY,
            "first_proximity",
        ),
        (
            STATE_HIGH_RISK,
            "first_high_risk",
        ),
        (
            STATE_CONFLICT,
            "first_conflict",
        ),
    ):

        first_sample = next(
            (
                sample
                for sample in ordered_samples
                if sample[
                    "state"
                ]
                == target_state
            ),
            None,
        )

        first_time = (
            None
            if first_sample is None
            else first_sample[
                "time_s"
            ]
        )

        output[
            f"{field_prefix}_s"
        ] = first_time

        output[
            f"{field_prefix}_timestamp_utc"
        ] = timestamp_from_offset(
            timestamp,
            first_time,
        )

    return output


# =============================================================================
# EVENT HELPERS
# =============================================================================

def row_altitude_ft(
    row: pd.Series,
) -> float | None:

    value = row.get(
        "baroaltitude",
        np.nan,
    )

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return (
        value
        * 3.2808398950131
    )


def safe_float(
    value,
) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def pair_worst_state(
    pair: dict,
) -> str:

    states = [
        pair.get(
            "state_current",
            STATE_NORMAL,
        ),
        pair.get(
            "worst_status_0_1min",
            STATE_NORMAL,
        ),
        pair.get(
            "worst_status_1_2min",
            STATE_NORMAL,
        ),
        pair.get(
            "worst_status_2_5min",
            STATE_NORMAL,
        ),
    ]

    return max(
        states,
        key=lambda state:
            STATE_PRIORITY.get(
                state,
                -1,
            ),
    )


def non_normal_windows(
    pair: dict,
) -> list[tuple[str, str]]:

    ordered = [
        (
            "CURRENT",
            pair.get(
                "state_current",
                STATE_NORMAL,
            ),
        ),
        (
            "0-1min",
            pair.get(
                "worst_status_0_1min",
                STATE_NORMAL,
            ),
        ),
        (
            "1-2min",
            pair.get(
                "worst_status_1_2min",
                STATE_NORMAL,
            ),
        ),
        (
            "2-5min",
            pair.get(
                "worst_status_2_5min",
                STATE_NORMAL,
            ),
        ),
    ]

    return [
        (window, state)
        for window, state in ordered
        if state != STATE_NORMAL
    ]


def first_worst_window(
    pair: dict,
    worst_state: str,
) -> str:

    ordered = [
        (
            "CURRENT",
            pair.get(
                "state_current",
                STATE_NORMAL,
            ),
        ),
        (
            "0-1min",
            pair.get(
                "worst_status_0_1min",
                STATE_NORMAL,
            ),
        ),
        (
            "1-2min",
            pair.get(
                "worst_status_1_2min",
                STATE_NORMAL,
            ),
        ),
        (
            "2-5min",
            pair.get(
                "worst_status_2_5min",
                STATE_NORMAL,
            ),
        ),
    ]

    for window, state in ordered:
        if state == worst_state:
            return window

    return ""


def problem_description(
    state: str,
) -> str:

    if state == STATE_PROXIMITY:
        return (
            "Inside the warning envelope; regulatory minimum separation "
            "is not simultaneously infringed."
        )

    if state == STATE_HIGH_RISK:
        return (
            "High-risk separation state: one minimum is infringed while "
            "the other dimension is also inside the warning envelope."
        )

    if state == STATE_CONFLICT:
        return (
            f"Loss of separation: horizontal < "
            f"{MIN_HORIZONTAL_SEPARATION_NM:g} NM AND vertical < "
            f"{MIN_VERTICAL_SEPARATION_FT:g} ft."
        )

    return "Normal separation."


def callsign_of(
    row: pd.Series,
) -> str:

    callsign = str(
        row.get(
            "callsign",
            "",
        )
        or ""
    ).strip()

    if callsign:
        return callsign

    return str(
        row.get(
            "icao24",
            "",
        )
        or ""
    ).upper()


def build_event_row(
    sector: str,
    timestamp: float,
    snapshot: pd.DataFrame,
    pair: dict,
    event_state: str,
) -> dict:

    aircraft_a = snapshot.loc[
        pair["index_a"]
    ]

    aircraft_b = snapshot.loc[
        pair["index_b"]
    ]

    speed_a_ms = ground_speed_ms(
        aircraft_a
    )

    speed_b_ms = ground_speed_ms(
        aircraft_b
    )

    vertical_a_ms = vertical_rate_ms(
        aircraft_a
    )

    vertical_b_ms = vertical_rate_ms(
        aircraft_b
    )

    (
        relative_horizontal_speed_kt,
        horizontal_closing_rate_kt,
    ) = horizontal_relative_metrics(
        aircraft_a,
        aircraft_b,
    )

    pair_vertical_closing_rate_fpm = (
        vertical_closing_rate_fpm(
            aircraft_a,
            aircraft_b,
        )
    )

    lookahead_metrics = (
        pair_lookahead_metrics(
            aircraft_a,
            aircraft_b,
            timestamp,
        )
    )

    windows = non_normal_windows(
        pair
    )

    timeline = ";".join(
        f"{window}={state}"
        for window, state in windows
    )

    first_alert = (
        windows[0][0]
        if windows
        else ""
    )

    timestamp_utc = (
        pd.to_datetime(
            timestamp,
            unit="s",
            utc=True,
        )
        .isoformat()
    )

    return {
        "sector":
            sector,

        "timestamp":
            timestamp,

        "timestamp_utc":
            timestamp_utc,

        "event_state":
            event_state,

        "event_colour":
            STATE_COLOURS[
                event_state
            ],

        "problem_description":
            problem_description(
                event_state
            ),

        "event_timeline":
            timeline,

        "first_alert_window":
            first_alert,

        "first_worst_state_window":
            first_worst_window(
                pair,
                event_state,
            ),

        "alert_current":
            pair.get(
                "state_current",
                STATE_NORMAL,
            )
            != STATE_NORMAL,

        "alert_0_1min":
            pair.get(
                "worst_status_0_1min",
                STATE_NORMAL,
            )
            != STATE_NORMAL,

        "alert_1_2min":
            pair.get(
                "worst_status_1_2min",
                STATE_NORMAL,
            )
            != STATE_NORMAL,

        "alert_2_5min":
            pair.get(
                "worst_status_2_5min",
                STATE_NORMAL,
            )
            != STATE_NORMAL,

        "state_current":
            pair.get(
                "state_current",
                STATE_NORMAL,
            ),

        "worst_status_0_1min":
            pair.get(
                "worst_status_0_1min",
                STATE_NORMAL,
            ),

        "worst_status_1_2min":
            pair.get(
                "worst_status_1_2min",
                STATE_NORMAL,
            ),

        "worst_status_2_5min":
            pair.get(
                "worst_status_2_5min",
                STATE_NORMAL,
            ),

        "icao24_a":
            str(
                aircraft_a.get(
                    "icao24",
                    "",
                )
                or ""
            ),

        "callsign_a":
            callsign_of(
                aircraft_a
            ),

        "lat_a":
            safe_float(
                aircraft_a.get(
                    "lat",
                    np.nan,
                )
            ),

        "lon_a":
            safe_float(
                aircraft_a.get(
                    "lon",
                    np.nan,
                )
            ),

        "baro_altitude_ft_a":
            row_altitude_ft(
                aircraft_a
            ),

        "ground_speed_ms_a":
            speed_a_ms,

        "ground_speed_kt_a":
            (
                None
                if speed_a_ms is None
                else speed_a_ms
                * MPS_TO_KT
            ),

        "true_track_deg_a":
            track_deg(
                aircraft_a
            ),

        "vertical_rate_ms_a":
            vertical_a_ms,

        "vertical_rate_fpm_a":
            (
                None
                if vertical_a_ms is None
                else vertical_a_ms
                * MPS_TO_FPM
            ),

        "icao24_b":
            str(
                aircraft_b.get(
                    "icao24",
                    "",
                )
                or ""
            ),

        "callsign_b":
            callsign_of(
                aircraft_b
            ),

        "lat_b":
            safe_float(
                aircraft_b.get(
                    "lat",
                    np.nan,
                )
            ),

        "lon_b":
            safe_float(
                aircraft_b.get(
                    "lon",
                    np.nan,
                )
            ),

        "baro_altitude_ft_b":
            row_altitude_ft(
                aircraft_b
            ),

        "ground_speed_ms_b":
            speed_b_ms,

        "ground_speed_kt_b":
            (
                None
                if speed_b_ms is None
                else speed_b_ms
                * MPS_TO_KT
            ),

        "true_track_deg_b":
            track_deg(
                aircraft_b
            ),

        "vertical_rate_ms_b":
            vertical_b_ms,

        "vertical_rate_fpm_b":
            (
                None
                if vertical_b_ms is None
                else vertical_b_ms
                * MPS_TO_FPM
            ),

        "relative_horizontal_speed_kt":
            relative_horizontal_speed_kt,

        "horizontal_closing_rate_kt":
            horizontal_closing_rate_kt,

        "vertical_closing_rate_fpm":
            pair_vertical_closing_rate_fpm,

        "horizontal_sep_nm":
            safe_float(
                pair.get(
                    "horizontal_sep_nm"
                )
            ),

        "vertical_sep_ft":
            safe_float(
                pair.get(
                    "vertical_sep_ft"
                )
            ),

        "cpa_nm":
            safe_float(
                pair.get(
                    "cpa_nm"
                )
            ),

        "tcpa_s":
            safe_float(
                pair.get(
                    "tcpa_s"
                )
            ),

        "vertical_sep_at_cpa_ft":
            safe_float(
                pair.get(
                    "vertical_sep_at_cpa_ft"
                )
            ),

        **lookahead_metrics,

        "predicted_conflict":
            bool(
                pair.get(
                    "predicted_conflict",
                    False,
                )
            ),
    }


# =============================================================================
# CSV WRITERS
# =============================================================================

def open_event_writers(
    paths: dict[str, Path],
):

    paths["directory"].mkdir(
        parents=True,
        exist_ok=True,
    )

    handles = {}
    writers = {}

    for state in ACTIVE_EVENT_STATES:

        path = paths[state]

        handle = path.open(
            "w",
            encoding="utf-8",
            newline="",
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()

        handles[state] = handle
        writers[state] = writer

    return handles, writers


def close_handles(
    handles: dict,
) -> None:

    for handle in handles.values():
        try:
            handle.close()
        except Exception:
            pass


# =============================================================================
# MONTH ANALYSIS
# =============================================================================

def month_outputs_complete(
    paths: dict[str, Path],
) -> bool:

    return all(
        paths[state].exists()
        for state in ACTIVE_EVENT_STATES
    )


def analyse_month(
    sector: str,
    input_csv: Path,
    year: int,
    month: int,
    chunksize: int,
    overwrite: bool,
) -> dict:

    paths = build_output_paths(
        sector,
        year,
        month,
    )

    if (
        month_outputs_complete(
            paths
        )
        and not overwrite
    ):
        print(
            "SKIP completed outputs:",
            input_csv,
        )

        return {
            "snapshots": 0,
            STATE_PROXIMITY: 0,
            STATE_HIGH_RISK: 0,
            STATE_CONFLICT: 0,
            "skipped": True,
        }

    handles = {}
    writers = {}

    counts = {
        "snapshots": 0,
        "pairs_evaluated": 0,
        "pairs_missing_vertical": 0,
        STATE_PROXIMITY: 0,
        STATE_HIGH_RISK: 0,
        STATE_CONFLICT: 0,
        "skipped": False,
    }

    try:
        handles, writers = (
            open_event_writers(
                paths
            )
        )

        for timestamp, snapshot in iter_snapshots_from_month(
            input_csv,
            chunksize,
        ):

            counts["snapshots"] += 1

            if len(snapshot) < 2:
                continue

            (
                pair_results,
                _aircraft_states,
                _interval_states,
                skipped_pairs,
            ) = analyse_aircraft_snapshot(
                snapshot,
                min_horizontal_nm=
                    MIN_HORIZONTAL_SEPARATION_NM,
                min_vertical_ft=
                    MIN_VERTICAL_SEPARATION_FT,
                warning_factor=
                    WARNING_FACTOR,
                state_rules=
                    SAFETY_STATE_RULES,
                lookahead_minutes=
                    LOOKAHEAD_MINUTES,
                prediction_step_seconds=
                    PREDICTION_STEP_SECONDS,
                check_ground_aircraft=
                    CHECK_GROUND_AIRCRAFT,
            )

            counts[
                "pairs_evaluated"
            ] += len(
                pair_results
            )

            counts[
                "pairs_missing_vertical"
            ] += skipped_pairs

            for pair in pair_results:

                event_state = (
                    pair_worst_state(
                        pair
                    )
                )

                if (
                    event_state
                    == STATE_NORMAL
                ):
                    continue

                row = build_event_row(
                    sector,
                    timestamp,
                    snapshot,
                    pair,
                    event_state,
                )

                writers[
                    event_state
                ].writerow(
                    row
                )

                counts[
                    event_state
                ] += 1

            if (
                counts["snapshots"]
                % 1000
                == 0
            ):
                print(
                    f"  snapshots={counts['snapshots']:,} | "
                    f"yellow={counts[STATE_PROXIMITY]:,} | "
                    f"orange={counts[STATE_HIGH_RISK]:,} | "
                    f"red={counts[STATE_CONFLICT]:,}"
                )

    except Exception:

        close_handles(
            handles
        )

        # Remove partially written final CSVs so a failed month is never
        # mistaken for a completed analysis on the next run.
        for state in (
            STATE_PROXIMITY,
            STATE_HIGH_RISK,
            STATE_CONFLICT,
        ):
            paths[state].unlink(
                missing_ok=True
            )

        raise

    finally:
        close_handles(
            handles
        )

    print()
    print(
        f"MONTH COMPLETE: "
        f"{year:04d}-{month:02d}"
    )

    print(
        f"  snapshots analysed : "
        f"{counts['snapshots']:,}"
    )

    print(
        f"  pair evaluations   : "
        f"{counts['pairs_evaluated']:,}"
    )

    monthly_labels = {
        STATE_PROXIMITY: "proximity warning",
        STATE_HIGH_RISK: "high-risk separation",
        STATE_CONFLICT: "loss of separation",
    }

    for state in ACTIVE_EVENT_STATES:
        print(
            f"  {monthly_labels[state]:<21}: "
            f"{counts[state]:,}"
        )

    print(
        f"  output directory   : "
        f"{paths['directory']}"
    )

    return counts


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Detect current and predicted pairwise separation events in "
            "monthly OpenSky sector archive CSV files."
        )
    )

    parser.add_argument(
        "--sector",
        default=DEFAULT_SECTOR,
        help=(
            "BBOX preset / sector folder name under "
            "data/raw/opensky_sectors/"
        ),
    )

    parser.add_argument(
        "--start-month",
        help=(
            "First month to process, YYYY-MM. "
            "Default: first available month."
        ),
    )

    parser.add_argument(
        "--end-month",
        help=(
            "Last month to process, YYYY-MM, inclusive. "
            "Default: last available month."
        ),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=READ_CHUNK_ROWS,
        help=(
            f"Rows per pandas read chunk "
            f"(default: {READ_CHUNK_ROWS:,})."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate monthly output CSVs even if all three already exist."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    arguments = (
        parse_arguments()
    )

    if arguments.chunksize <= 0:
        raise ValueError(
            "--chunksize must be positive"
        )

    sector = (
        arguments.sector
    )

    start_month = (
        parse_month_key(
            arguments.start_month
        )
    )

    end_month = (
        parse_month_key(
            arguments.end_month
        )
    )

    if (
        start_month is not None
        and end_month is not None
        and month_tuple_to_int(
            end_month
        )
        <
        month_tuple_to_int(
            start_month
        )
    ):
        raise ValueError(
            "--end-month must be >= --start-month"
        )

    monthly_files = (
        discover_monthly_csvs(
            sector,
            start_month,
            end_month,
        )
    )

    if not monthly_files:
        raise FileNotFoundError(
            "No monthly sector CSV files matched the requested selection."
        )

    print()
    print("=" * 78)
    print(
        "OpenSky separation-event detector"
    )
    print("=" * 78)
    print(
        f"Sector: {sector}"
    )
    print(
        f"Input root: {INPUT_ROOT / sector}"
    )
    print(
        f"Output root: {OUTPUT_ROOT / sector}"
    )
    print(
        f"Monthly files: {len(monthly_files)}"
    )
    print(
        f"Separation profile: {SEPARATION_PROFILE}"
    )
    print(
        f"Minimum separation: "
        f"{MIN_HORIZONTAL_SEPARATION_NM:g} NM / "
        f"{MIN_VERTICAL_SEPARATION_FT:g} ft"
    )
    print(
        "Enabled safety states: "
        + ", ".join(
            sorted(
                ENABLED_SAFETY_STATES,
                key=lambda state:
                    STATE_PRIORITY[state],
            )
        )
    )
    print(
        f"Warning envelope: "
        f"{MIN_HORIZONTAL_SEPARATION_NM * WARNING_FACTOR:g} NM / "
        f"{MIN_VERTICAL_SEPARATION_FT * WARNING_FACTOR:g} ft"
    )
    print(
        f"Look-ahead intervals: "
        f"0-1, 1-2, 2-5 min"
    )
    print(
        f"Prediction step: "
        f"{PREDICTION_STEP_SECONDS} s"
    )
    print(
        f"Cruise filter: "
        + (
            f"FL{CRUISE_FL_MIN}-FL{CRUISE_FL_MAX}"
            if FILTER_CRUISE_FL_BAND
            else "disabled"
        )
    )
    print("=" * 78)

    total = {
        "snapshots": 0,
        "pairs_evaluated": 0,
        STATE_PROXIMITY: 0,
        STATE_HIGH_RISK: 0,
        STATE_CONFLICT: 0,
        "months_skipped": 0,
    }

    for (
        input_csv,
        year,
        month,
    ) in monthly_files:

        print()
        print("=" * 78)
        print(
            f"INPUT: {input_csv}"
        )
        print("=" * 78)

        counts = (
            analyse_month(
                sector=sector,
                input_csv=input_csv,
                year=year,
                month=month,
                chunksize=
                    arguments.chunksize,
                overwrite=(
                    arguments.overwrite
                    or OVERWRITE_EXISTING
                ),
            )
        )

        if counts.get(
            "skipped",
            False,
        ):
            total[
                "months_skipped"
            ] += 1

            continue

        total[
            "snapshots"
        ] += counts[
            "snapshots"
        ]

        total[
            "pairs_evaluated"
        ] += counts[
            "pairs_evaluated"
        ]

        for state in (
            STATE_PROXIMITY,
            STATE_HIGH_RISK,
            STATE_CONFLICT,
        ):
            total[state] += (
                counts[state]
            )

    print()
    print("=" * 78)
    print(
        "DETECTION FINISHED"
    )
    print("=" * 78)
    print(
        f"Snapshots analysed: "
        f"{total['snapshots']:,}"
    )
    print(
        f"Pair evaluations: "
        f"{total['pairs_evaluated']:,}"
    )
    final_labels = {
        STATE_PROXIMITY: "Proximity-warning events",
        STATE_HIGH_RISK: "High-risk events",
        STATE_CONFLICT: "Loss-of-separation events",
    }

    for state in ACTIVE_EVENT_STATES:
        print(
            f"{final_labels[state]}: "
            f"{total[state]:,}"
        )
    print(
        f"Months already complete/skipped: "
        f"{total['months_skipped']:,}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
