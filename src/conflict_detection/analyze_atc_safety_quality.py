#!/usr/bin/env python3
"""
ATC safety-performance analysis for OpenSky separation runs.

Purpose
-------
Generate thesis-ready metrics and plots that quantify how often the traffic
operates in GREEN / YELLOW / ORANGE / RED conditions, while keeping two
different concepts separate:

1) CURRENT STATE
   What is happening now.

2) EVENT STATE
   Worst state found in:
       NOW + 0-1 min + 1-2 min + 2-5 min

This distinction is essential:
a pair can be GREEN now but belong to a RED event because a loss of separation
is predicted later in the 5-minute horizon.

The script also groups consecutive pair observations into encounter "episodes"
so that one conflict lasting several minutes is not counted as dozens of
independent cases.

Outputs
-------
CSV:
    quality_metrics_summary.csv
    state_distribution_pair_current.csv
    state_distribution_pair_event.csv
    state_distribution_snapshot_current.csv
    state_distribution_snapshot_event.csv
    episodes.csv
    daily_metrics.csv
    aircraft_state_distribution_current.csv   (if traffic scan is enabled)

Figures (PNG + PDF):
    01_pair_current_state_distribution
    02_pair_current_non_green_rate
    03_pair_event_state_distribution
    04_snapshot_current_time_distribution
    05_snapshot_event_time_distribution
    06_daily_event_episodes
    07_episode_duration_histogram
    08_aircraft_current_state_distribution    (if traffic scan is enabled)

The CSV summaries are deliberately machine-readable so that a later
"manual vs automated controller" comparison can use exactly the same KPIs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_AREA = "LECMSAN"
DEFAULT_RUN_NAME = "20190101T000000_20190228T235950"
DEFAULT_LABEL = "Manual control"

DEFAULT_EPISODE_GAP_S = 60
DEFAULT_SNAPSHOT_STEP_S = 10
READ_CHUNK_ROWS = 300_000

SEPARATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opensky_separation"
)

PROCESSED_10S_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opensky_10s"
)


def discover_acquisition_dir() -> Path:
    candidates = (
        SRC_DIR / "acquisition",
        PROJECT_ROOT / "acquisition",
    )

    for candidate in candidates:
        if (
            (candidate / "separation_profiles.json").exists()
            and (candidate / "bbox_presets.json").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate acquisition directory. Expected either:\n"
        f"  {SRC_DIR / 'acquisition'}\n"
        "or:\n"
        f"  {PROJECT_ROOT / 'acquisition'}"
    )


ACQUISITION_DIR = discover_acquisition_dir()

SEPARATION_PROFILES_FILE = (
    ACQUISITION_DIR
    / "separation_profiles.json"
)


# =============================================================================
# STATES
# =============================================================================

STATE_NORMAL = "NORMAL"
STATE_PROXIMITY = "PROXIMITY"
STATE_HIGH_RISK = "HIGH_RISK"
STATE_CONFLICT = "CONFLICT"

STATES = (
    STATE_NORMAL,
    STATE_PROXIMITY,
    STATE_HIGH_RISK,
    STATE_CONFLICT,
)

STATE_PRIORITY = {
    STATE_NORMAL: 0,
    STATE_PROXIMITY: 1,
    STATE_HIGH_RISK: 2,
    STATE_CONFLICT: 3,
}

PRIORITY_STATE = {
    value: key
    for key, value in STATE_PRIORITY.items()
}

STATE_SHORT = {
    STATE_NORMAL: "Green",
    STATE_PROXIMITY: "Yellow",
    STATE_HIGH_RISK: "Orange",
    STATE_CONFLICT: "Red",
}

EVENT_FILES = (
    "yellow_proximity_events.csv",
    "orange_high_risk_events.csv",
    "red_loss_of_separation_events.csv",
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "run"


def normalize_state(value) -> str:
    text = str(value).strip().upper()

    aliases = {
        "GREEN": STATE_NORMAL,
        "NORMAL": STATE_NORMAL,
        "YELLOW": STATE_PROXIMITY,
        "PROXIMITY": STATE_PROXIMITY,
        "ORANGE": STATE_HIGH_RISK,
        "HIGH_RISK": STATE_HIGH_RISK,
        "HIGH-RISK": STATE_HIGH_RISK,
        "RED": STATE_CONFLICT,
        "CONFLICT": STATE_CONFLICT,
    }

    return aliases.get(text, text)


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_original = int(series.notna().sum())
    valid_numeric = int(numeric.notna().sum())

    if (
        valid_original > 0
        and valid_numeric >= 0.90 * valid_original
    ):
        numeric = numeric.astype("float64")
        finite = numeric[np.isfinite(numeric)]

        if finite.empty:
            return numeric

        magnitude = float(
            finite.abs().median()
        )

        if magnitude > 1e17:
            return numeric / 1_000_000_000.0
        if magnitude > 1e14:
            return numeric / 1_000_000.0
        if magnitude > 1e11:
            return numeric / 1_000.0

        return numeric

    values = pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
        format="mixed",
    )

    epoch = pd.Timestamp(
        "1970-01-01",
        tz="UTC",
    )

    return (
        values
        .sub(epoch)
        .dt
        .total_seconds()
        .astype("float64")
    )


def canonical_pair_series(
    icao_a: pd.Series,
    icao_b: pd.Series,
) -> pd.Series:
    a = (
        icao_a
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    b = (
        icao_b
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    first = np.where(
        a <= b,
        a,
        b,
    )

    second = np.where(
        a <= b,
        b,
        a,
    )

    return (
        pd.Series(
            first,
            index=icao_a.index,
        )
        + "|"
        + pd.Series(
            second,
            index=icao_a.index,
        )
    )


# =============================================================================
# PROFILE COLOURS
# =============================================================================

def load_state_colours(
    profile_name: str,
) -> dict[str, str]:
    with SEPARATION_PROFILES_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        profiles = json.load(handle)

    if profile_name not in profiles:
        raise ValueError(
            f"Unknown separation profile: {profile_name}"
        )

    profile = profiles[
        profile_name
    ]

    colours = {
        STATE_NORMAL: "#00A651",
        STATE_PROXIMITY: "#FFD400",
        STATE_HIGH_RISK: "#FF6F00",
        STATE_CONFLICT: "#D7191C",
    }

    for rule in profile.get(
        "states",
        {},
    ).values():
        state = normalize_state(
            rule.get(
                "state",
                "",
            )
        )

        colour = rule.get(
            "colour"
        )

        if (
            state in STATE_PRIORITY
            and colour
        ):
            colours[state] = colour

    return colours


# =============================================================================
# LOAD RUN
# =============================================================================

def load_run_summary(
    run_dir: Path,
) -> pd.Series:
    path = (
        run_dir
        / "run_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing run summary:\n{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"Empty run summary:\n{path}"
        )

    return dataframe.iloc[0]


def load_events(
    run_dir: Path,
) -> pd.DataFrame:
    frames = []

    for filename in EVENT_FILES:
        path = (
            run_dir
            / filename
        )

        if not path.exists():
            print(
                f"WARNING: missing {filename}"
            )
            continue

        try:
            dataframe = pd.read_csv(
                path,
                low_memory=False,
            )
        except pd.errors.EmptyDataError:
            continue

        if dataframe.empty:
            continue

        dataframe = (
            dataframe.copy()
        )

        dataframe[
            "source_file"
        ] = filename

        frames.append(
            dataframe
        )

    if not frames:
        columns = [
            "timestamp",
            "timestamp_utc",
            "event_state",
            "state_current",
            "icao24_a",
            "icao24_b",
        ]
        return pd.DataFrame(
            columns=columns
        )

    events = pd.concat(
        frames,
        ignore_index=True,
    )

    required = {
        "timestamp",
        "event_state",
        "state_current",
        "icao24_a",
        "icao24_b",
    }

    missing = (
        required
        - set(
            events.columns
        )
    )

    if missing:
        raise ValueError(
            "Event CSVs are missing required columns: "
            f"{sorted(missing)}"
        )

    events[
        "timestamp"
    ] = (
        to_epoch_seconds(
            events[
                "timestamp"
            ]
        )
        .round()
        .astype(
            "Int64"
        )
    )

    events = events.loc[
        events[
            "timestamp"
        ].notna()
    ].copy()

    events[
        "timestamp"
    ] = events[
        "timestamp"
    ].astype(
        "int64"
    )

    events[
        "event_state"
    ] = events[
        "event_state"
    ].map(
        normalize_state
    )

    events[
        "state_current"
    ] = events[
        "state_current"
    ].map(
        normalize_state
    )

    events[
        "pair_key"
    ] = canonical_pair_series(
        events[
            "icao24_a"
        ],
        events[
            "icao24_b"
        ],
    )

    events[
        "timestamp_dt"
    ] = pd.to_datetime(
        events[
            "timestamp"
        ],
        unit="s",
        utc=True,
    )

    events[
        "day"
    ] = (
        events[
            "timestamp_dt"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # One detector row per pair and timestamp is expected.
    # If duplicates exist, keep the most severe event.
    events[
        "_event_priority"
    ] = events[
        "event_state"
    ].map(
        STATE_PRIORITY
    )

    events.sort_values(
        "_event_priority",
        inplace=True,
    )

    events.drop_duplicates(
        subset=[
            "timestamp",
            "pair_key",
        ],
        keep="last",
        inplace=True,
    )

    events.reset_index(
        drop=True,
        inplace=True,
    )

    return events


# =============================================================================
# DISTRIBUTION HELPERS
# =============================================================================

def distribution_frame(
    counts: dict[str, int],
    total: int,
) -> pd.DataFrame:
    rows = []

    for state in STATES:
        count = int(
            counts.get(
                state,
                0,
            )
        )

        percentage = (
            100.0
            * count
            / total
            if total > 0
            else np.nan
        )

        rows.append(
            {
                "state": state,
                "display_state": STATE_SHORT[state],
                "count": count,
                "percentage": percentage,
            }
        )

    return pd.DataFrame(
        rows
    )


def pair_current_distribution(
    events: pd.DataFrame,
    pairs_total: int,
) -> pd.DataFrame:
    non_green = (
        events.loc[
            events[
                "state_current"
            ]
            != STATE_NORMAL,
            "state_current",
        ]
        .value_counts()
        .to_dict()
    )

    non_green_total = int(
        sum(
            non_green.values()
        )
    )

    counts = {
        STATE_NORMAL:
            max(
                0,
                pairs_total
                - non_green_total,
            ),

        STATE_PROXIMITY:
            int(
                non_green.get(
                    STATE_PROXIMITY,
                    0,
                )
            ),

        STATE_HIGH_RISK:
            int(
                non_green.get(
                    STATE_HIGH_RISK,
                    0,
                )
            ),

        STATE_CONFLICT:
            int(
                non_green.get(
                    STATE_CONFLICT,
                    0,
                )
            ),
    }

    return distribution_frame(
        counts,
        pairs_total,
    )


def pair_event_distribution(
    events: pd.DataFrame,
    pairs_total: int,
) -> pd.DataFrame:
    event_counts = (
        events[
            "event_state"
        ]
        .value_counts()
        .to_dict()
    )

    event_total = int(
        len(
            events
        )
    )

    counts = {
        STATE_NORMAL:
            max(
                0,
                pairs_total
                - event_total,
            ),

        STATE_PROXIMITY:
            int(
                event_counts.get(
                    STATE_PROXIMITY,
                    0,
                )
            ),

        STATE_HIGH_RISK:
            int(
                event_counts.get(
                    STATE_HIGH_RISK,
                    0,
                )
            ),

        STATE_CONFLICT:
            int(
                event_counts.get(
                    STATE_CONFLICT,
                    0,
                )
            ),
    }

    return distribution_frame(
        counts,
        pairs_total,
    )


def worst_state_by_timestamp(
    events: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "state",
                "priority",
            ]
        )

    temp = events[
        [
            "timestamp",
            column,
        ]
    ].copy()

    temp[
        "priority"
    ] = temp[
        column
    ].map(
        STATE_PRIORITY
    )

    worst = (
        temp
        .groupby(
            "timestamp",
            as_index=False,
        )[
            "priority"
        ]
        .max()
    )

    worst[
        "state"
    ] = worst[
        "priority"
    ].map(
        PRIORITY_STATE
    )

    return worst


def snapshot_distribution(
    events: pd.DataFrame,
    snapshots_total: int,
    column: str,
) -> pd.DataFrame:
    worst = worst_state_by_timestamp(
        events,
        column,
    )

    non_green = (
        worst.loc[
            worst[
                "state"
            ]
            != STATE_NORMAL,
            "state",
        ]
        .value_counts()
        .to_dict()
    )

    non_green_total = int(
        sum(
            non_green.values()
        )
    )

    counts = {
        STATE_NORMAL:
            max(
                0,
                snapshots_total
                - non_green_total,
            ),

        STATE_PROXIMITY:
            int(
                non_green.get(
                    STATE_PROXIMITY,
                    0,
                )
            ),

        STATE_HIGH_RISK:
            int(
                non_green.get(
                    STATE_HIGH_RISK,
                    0,
                )
            ),

        STATE_CONFLICT:
            int(
                non_green.get(
                    STATE_CONFLICT,
                    0,
                )
            ),
    }

    return distribution_frame(
        counts,
        snapshots_total,
    )


# =============================================================================
# EPISODES
# =============================================================================

def build_episodes(
    events: pd.DataFrame,
    gap_s: int,
    snapshot_step_s: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "pair_key",
                "episode_number",
                "start_timestamp",
                "end_timestamp",
                "start_utc",
                "end_utc",
                "duration_s",
                "rows",
                "max_event_state",
                "max_current_state",
                "start_day",
            ]
        )

    work = (
        events.copy()
    )

    work.sort_values(
        [
            "pair_key",
            "timestamp",
        ],
        inplace=True,
    )

    work[
        "_gap_s"
    ] = (
        work
        .groupby(
            "pair_key"
        )[
            "timestamp"
        ]
        .diff()
    )

    work[
        "_new_episode"
    ] = (
        work[
            "_gap_s"
        ].isna()
        |
        (
            work[
                "_gap_s"
            ]
            > gap_s
        )
    ).astype(int)

    work[
        "episode_number"
    ] = (
        work
        .groupby(
            "pair_key"
        )[
            "_new_episode"
        ]
        .cumsum()
    )

    work[
        "_event_priority"
    ] = work[
        "event_state"
    ].map(
        STATE_PRIORITY
    )

    work[
        "_current_priority"
    ] = work[
        "state_current"
    ].map(
        STATE_PRIORITY
    )

    grouped = (
        work.groupby(
            [
                "pair_key",
                "episode_number",
            ],
            as_index=False,
        )
        .agg(
            start_timestamp=(
                "timestamp",
                "min",
            ),
            end_timestamp=(
                "timestamp",
                "max",
            ),
            rows=(
                "timestamp",
                "size",
            ),
            max_event_priority=(
                "_event_priority",
                "max",
            ),
            max_current_priority=(
                "_current_priority",
                "max",
            ),
        )
    )

    grouped[
        "duration_s"
    ] = (
        grouped[
            "end_timestamp"
        ]
        - grouped[
            "start_timestamp"
        ]
        + snapshot_step_s
    )

    grouped[
        "max_event_state"
    ] = grouped[
        "max_event_priority"
    ].map(
        PRIORITY_STATE
    )

    grouped[
        "max_current_state"
    ] = grouped[
        "max_current_priority"
    ].map(
        PRIORITY_STATE
    )

    grouped[
        "start_utc"
    ] = pd.to_datetime(
        grouped[
            "start_timestamp"
        ],
        unit="s",
        utc=True,
    )

    grouped[
        "end_utc"
    ] = pd.to_datetime(
        grouped[
            "end_timestamp"
        ],
        unit="s",
        utc=True,
    )

    grouped[
        "start_day"
    ] = (
        grouped[
            "start_utc"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    return grouped[
        [
            "pair_key",
            "episode_number",
            "start_timestamp",
            "end_timestamp",
            "start_utc",
            "end_utc",
            "duration_s",
            "rows",
            "max_event_state",
            "max_current_state",
            "start_day",
        ]
    ]


# =============================================================================
# TRAFFIC SCAN
# =============================================================================

def month_strings_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[str]:
    start_month = (
        start
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
    )

    end_month = (
        end
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
    )

    return [
        str(period)
        for period
        in pd.period_range(
            start_month,
            end_month,
            freq="M",
        )
    ]


def find_traffic_files(
    area: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    directory = (
        PROCESSED_10S_ROOT
        / area
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Processed traffic directory not found:\n{directory}"
        )

    files = []

    for month in month_strings_between(
        start,
        end,
    ):
        files.extend(
            sorted(
                path
                for path
                in directory.rglob(
                    f"*_{month}_10s.csv"
                )
                if "_daily_cache"
                not in path.parts
            )
        )

    # Preserve order while removing duplicates.
    unique = []
    seen = set()

    for path in files:
        resolved = path.resolve()

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        unique.append(
            path
        )

    if not unique:
        raise FileNotFoundError(
            "No processed *_10s.csv traffic files found "
            f"for {area} between {start} and {end}"
        )

    return unique


def scan_traffic(
    area: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunksize: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load only time + icao24 from processed 10-second traffic files.

    Returns:
        traffic_observations:
            unique timestamp-aircraft rows

        snapshot_stats:
            one row per timestamp with:
                aircraft_count
                pair_count
                day
    """

    files = find_traffic_files(
        area,
        start,
        end,
    )

    start_s = int(
        start.timestamp()
    )

    end_s = int(
        end.timestamp()
    )

    pieces = []

    print()
    print(
        "Scanning traffic denominators..."
    )

    for path in files:
        print(
            f"  {path}"
        )

        for chunk in pd.read_csv(
            path,
            usecols=lambda column:
                column in {
                    "time",
                    "icao24",
                },
            chunksize=chunksize,
            low_memory=False,
        ):
            if (
                "time"
                not in chunk.columns
                or
                "icao24"
                not in chunk.columns
            ):
                raise ValueError(
                    f"Expected time and icao24 columns in {path}"
                )

            timestamp = (
                to_epoch_seconds(
                    chunk[
                        "time"
                    ]
                )
                .round()
            )

            mask = (
                timestamp.notna()
                &
                (
                    timestamp
                    >= start_s
                )
                &
                (
                    timestamp
                    <= end_s
                )
            )

            if not mask.any():
                continue

            piece = pd.DataFrame(
                {
                    "timestamp":
                        timestamp.loc[
                            mask
                        ].astype(
                            "int64"
                        ),

                    "icao24":
                        (
                            chunk.loc[
                                mask,
                                "icao24",
                            ]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        ),
                }
            )

            piece = piece.loc[
                piece[
                    "icao24"
                ]
                != ""
            ]

            pieces.append(
                piece
            )

    if not pieces:
        raise RuntimeError(
            "No traffic rows found inside the run interval."
        )

    traffic = pd.concat(
        pieces,
        ignore_index=True,
    )

    traffic.drop_duplicates(
        subset=[
            "timestamp",
            "icao24",
        ],
        inplace=True,
    )

    traffic.reset_index(
        drop=True,
        inplace=True,
    )

    traffic[
        "timestamp_dt"
    ] = pd.to_datetime(
        traffic[
            "timestamp"
        ],
        unit="s",
        utc=True,
    )

    traffic[
        "day"
    ] = (
        traffic[
            "timestamp_dt"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    snapshot_stats = (
        traffic
        .groupby(
            "timestamp",
            as_index=False,
        )
        .agg(
            aircraft_count=(
                "icao24",
                "nunique",
            ),
            day=(
                "day",
                "first",
            ),
        )
    )

    snapshot_stats[
        "pair_count"
    ] = (
        snapshot_stats[
            "aircraft_count"
        ]
        * (
            snapshot_stats[
                "aircraft_count"
            ]
            - 1
        )
        // 2
    )

    return (
        traffic,
        snapshot_stats,
    )


# =============================================================================
# AIRCRAFT CURRENT-STATE EXPOSURE
# =============================================================================

def aircraft_current_distribution(
    traffic: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every timestamp-aircraft observation, assign the most severe
    CURRENT pair state involving that aircraft.

    Aircraft not participating in a non-green current pair are GREEN.
    """

    total = int(
        len(
            traffic
        )
    )

    current = events.loc[
        events[
            "state_current"
        ]
        != STATE_NORMAL
    ].copy()

    if current.empty:
        return distribution_frame(
            {
                STATE_NORMAL: total,
            },
            total,
        )

    a = current[
        [
            "timestamp",
            "icao24_a",
            "state_current",
        ]
    ].rename(
        columns={
            "icao24_a":
                "icao24",
        }
    )

    b = current[
        [
            "timestamp",
            "icao24_b",
            "state_current",
        ]
    ].rename(
        columns={
            "icao24_b":
                "icao24",
        }
    )

    aircraft_states = pd.concat(
        [
            a,
            b,
        ],
        ignore_index=True,
    )

    aircraft_states[
        "icao24"
    ] = (
        aircraft_states[
            "icao24"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    aircraft_states[
        "priority"
    ] = aircraft_states[
        "state_current"
    ].map(
        STATE_PRIORITY
    )

    aircraft_states = (
        aircraft_states
        .groupby(
            [
                "timestamp",
                "icao24",
            ],
            as_index=False,
        )[
            "priority"
        ]
        .max()
    )

    aircraft_states[
        "state"
    ] = aircraft_states[
        "priority"
    ].map(
        PRIORITY_STATE
    )

    counts_non_green = (
        aircraft_states[
            "state"
        ]
        .value_counts()
        .to_dict()
    )

    non_green_total = int(
        len(
            aircraft_states
        )
    )

    counts = {
        STATE_NORMAL:
            max(
                0,
                total
                - non_green_total,
            ),

        STATE_PROXIMITY:
            int(
                counts_non_green.get(
                    STATE_PROXIMITY,
                    0,
                )
            ),

        STATE_HIGH_RISK:
            int(
                counts_non_green.get(
                    STATE_HIGH_RISK,
                    0,
                )
            ),

        STATE_CONFLICT:
            int(
                counts_non_green.get(
                    STATE_CONFLICT,
                    0,
                )
            ),
    }

    return distribution_frame(
        counts,
        total,
    )


# =============================================================================
# DAILY METRICS
# =============================================================================

def build_daily_metrics(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    snapshot_stats: pd.DataFrame | None,
) -> pd.DataFrame:
    if events.empty:
        event_days = pd.DataFrame(
            columns=[
                "day",
            ]
        )
    else:
        event_days = pd.DataFrame(
            {
                "day":
                    sorted(
                        events[
                            "day"
                        ].unique()
                    )
            }
        )

    if snapshot_stats is not None:
        daily_base = (
            snapshot_stats
            .groupby(
                "day",
                as_index=False,
            )
            .agg(
                snapshots=(
                    "timestamp",
                    "nunique",
                ),
                pair_observations=(
                    "pair_count",
                    "sum",
                ),
                aircraft_observations=(
                    "aircraft_count",
                    "sum",
                ),
            )
        )
    else:
        daily_base = (
            event_days.copy()
        )

    # -------------------------------------------------------------------------
    # Pair current/event counts per day.
    # -------------------------------------------------------------------------

    current_counts = (
        events.loc[
            events[
                "state_current"
            ]
            != STATE_NORMAL
        ]
        .groupby(
            [
                "day",
                "state_current",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        if not events.empty
        else pd.DataFrame()
    )

    event_counts = (
        events
        .groupby(
            [
                "day",
                "event_state",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        if not events.empty
        else pd.DataFrame()
    )

    # -------------------------------------------------------------------------
    # Worst current state per timestamp.
    # -------------------------------------------------------------------------

    current_worst = worst_state_by_timestamp(
        events,
        "state_current",
    )

    if not current_worst.empty:
        current_worst[
            "day"
        ] = (
            pd.to_datetime(
                current_worst[
                    "timestamp"
                ],
                unit="s",
                utc=True,
            )
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

    snapshot_current_counts = (
        current_worst.loc[
            current_worst[
                "state"
            ]
            != STATE_NORMAL
        ]
        .groupby(
            [
                "day",
                "state",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        if not current_worst.empty
        else pd.DataFrame()
    )

    # -------------------------------------------------------------------------
    # Episodes by start day and max event state.
    # -------------------------------------------------------------------------

    episode_counts = (
        episodes
        .groupby(
            [
                "start_day",
                "max_event_state",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        if not episodes.empty
        else pd.DataFrame()
    )

    if daily_base.empty:
        all_days = sorted(
            set(
                events[
                    "day"
                ].tolist()
            )
        )

        daily_base = pd.DataFrame(
            {
                "day":
                    all_days
            }
        )

    daily = (
        daily_base
        .copy()
        .sort_values(
            "day"
        )
        .reset_index(
            drop=True
        )
    )

    # Pair counts.
    for state in (
        STATE_PROXIMITY,
        STATE_HIGH_RISK,
        STATE_CONFLICT,
    ):
        current_map = (
            current_counts[
                state
            ]
            if state
            in current_counts.columns
            else pd.Series(
                dtype=float
            )
        )

        event_map = (
            event_counts[
                state
            ]
            if state
            in event_counts.columns
            else pd.Series(
                dtype=float
            )
        )

        snapshot_map = (
            snapshot_current_counts[
                state
            ]
            if state
            in snapshot_current_counts.columns
            else pd.Series(
                dtype=float
            )
        )

        episode_map = (
            episode_counts[
                state
            ]
            if state
            in episode_counts.columns
            else pd.Series(
                dtype=float
            )
        )

        daily[
            f"current_pair_{state.lower()}"
        ] = (
            daily[
                "day"
            ]
            .map(
                current_map
            )
            .fillna(0)
            .astype(int)
        )

        daily[
            f"event_pair_{state.lower()}"
        ] = (
            daily[
                "day"
            ]
            .map(
                event_map
            )
            .fillna(0)
            .astype(int)
        )

        daily[
            f"current_snapshot_{state.lower()}"
        ] = (
            daily[
                "day"
            ]
            .map(
                snapshot_map
            )
            .fillna(0)
            .astype(int)
        )

        daily[
            f"episodes_{state.lower()}"
        ] = (
            daily[
                "day"
            ]
            .map(
                episode_map
            )
            .fillna(0)
            .astype(int)
        )

    if (
        snapshot_stats is not None
        and
        "snapshots"
        in daily.columns
    ):
        non_green_snapshots = (
            daily[
                "current_snapshot_proximity"
            ]
            + daily[
                "current_snapshot_high_risk"
            ]
            + daily[
                "current_snapshot_conflict"
            ]
        )

        daily[
            "current_snapshot_normal"
        ] = (
            daily[
                "snapshots"
            ]
            - non_green_snapshots
        ).clip(
            lower=0
        )

        for state in STATES:
            column = (
                f"current_snapshot_{state.lower()}"
            )

            daily[
                f"{column}_pct"
            ] = (
                100.0
                * daily[
                    column
                ]
                / daily[
                    "snapshots"
                ].replace(
                    0,
                    np.nan,
                )
            )

        current_non_green_pairs = (
            daily[
                "current_pair_proximity"
            ]
            + daily[
                "current_pair_high_risk"
            ]
            + daily[
                "current_pair_conflict"
            ]
        )

        daily[
            "current_pair_normal"
        ] = (
            daily[
                "pair_observations"
            ]
            - current_non_green_pairs
        ).clip(
            lower=0
        )

        event_non_green_pairs = (
            daily[
                "event_pair_proximity"
            ]
            + daily[
                "event_pair_high_risk"
            ]
            + daily[
                "event_pair_conflict"
            ]
        )

        daily[
            "event_pair_normal"
        ] = (
            daily[
                "pair_observations"
            ]
            - event_non_green_pairs
        ).clip(
            lower=0
        )

        for state in (
            STATE_PROXIMITY,
            STATE_HIGH_RISK,
            STATE_CONFLICT,
        ):
            daily[
                f"current_pair_{state.lower()}_per_10000"
            ] = (
                10_000.0
                * daily[
                    f"current_pair_{state.lower()}"
                ]
                / daily[
                    "pair_observations"
                ].replace(
                    0,
                    np.nan,
                )
            )

    return daily


# =============================================================================
# SUMMARY KPI TABLE
# =============================================================================

def append_distribution_metrics(
    rows: list[dict],
    prefix: str,
    dataframe: pd.DataFrame,
    label: str,
):
    for _, row in dataframe.iterrows():
        state = row[
            "state"
        ]

        rows.append(
            {
                "label": label,
                "metric":
                    f"{prefix}_{state.lower()}_count",
                "value":
                    float(
                        row[
                            "count"
                        ]
                    ),
                "unit":
                    "count",
            }
        )

        rows.append(
            {
                "label": label,
                "metric":
                    f"{prefix}_{state.lower()}_pct",
                "value":
                    float(
                        row[
                            "percentage"
                        ]
                    ),
                "unit":
                    "percent",
            }
        )


def build_summary_metrics(
    label: str,
    run_summary: pd.Series,
    pair_current: pd.DataFrame,
    pair_event: pd.DataFrame,
    snapshot_current: pd.DataFrame,
    snapshot_event: pd.DataFrame,
    episodes: pd.DataFrame,
    aircraft_current: pd.DataFrame | None,
    traffic: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []

    def add(
        metric: str,
        value,
        unit: str,
    ):
        rows.append(
            {
                "label": label,
                "metric": metric,
                "value": value,
                "unit": unit,
            }
        )

    add(
        "snapshots_total",
        int(
            run_summary[
                "snapshots"
            ]
        ),
        "count",
    )

    add(
        "pair_observations_total",
        int(
            run_summary[
                "pairs_total"
            ]
        ),
        "count",
    )

    if traffic is not None:
        add(
            "aircraft_observations_total",
            int(
                len(
                    traffic
                )
            ),
            "count",
        )

        add(
            "unique_aircraft_total",
            int(
                traffic[
                    "icao24"
                ].nunique()
            ),
            "count",
        )

    append_distribution_metrics(
        rows,
        "pair_current",
        pair_current,
        label,
    )

    append_distribution_metrics(
        rows,
        "pair_event",
        pair_event,
        label,
    )

    append_distribution_metrics(
        rows,
        "snapshot_current",
        snapshot_current,
        label,
    )

    append_distribution_metrics(
        rows,
        "snapshot_event",
        snapshot_event,
        label,
    )

    if aircraft_current is not None:
        append_distribution_metrics(
            rows,
            "aircraft_current",
            aircraft_current,
            label,
        )

    add(
        "episodes_total",
        int(
            len(
                episodes
            )
        ),
        "count",
    )

    for state in (
        STATE_PROXIMITY,
        STATE_HIGH_RISK,
        STATE_CONFLICT,
    ):
        state_episodes = (
            episodes.loc[
                episodes[
                    "max_event_state"
                ]
                == state
            ]
        )

        add(
            f"episodes_{state.lower()}",
            int(
                len(
                    state_episodes
                )
            ),
            "count",
        )

        if not state_episodes.empty:
            add(
                f"episode_duration_{state.lower()}_median_s",
                float(
                    state_episodes[
                        "duration_s"
                    ].median()
                ),
                "seconds",
            )

            add(
                f"episode_duration_{state.lower()}_mean_s",
                float(
                    state_episodes[
                        "duration_s"
                    ].mean()
                ),
                "seconds",
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PLOTTING
# =============================================================================

def set_adaptive_daily_xaxis(
    ax,
    daily: pd.DataFrame,
) -> None:
    """
    Keep daily time axes readable.

    Rules:
      - If the data cover only one calendar month, label every available day.
      - If the data cover more than one month, label only the 1st and 15th
        of each month.

    The bars themselves are never removed; only the visible tick labels are
    reduced.
    """
    if daily.empty or "day" not in daily.columns:
        return

    dates = pd.to_datetime(
        daily["day"],
        errors="coerce",
    )

    valid = dates.notna()

    if not valid.any():
        return

    valid_dates = dates.loc[valid]

    month_count = int(
        valid_dates.dt.to_period("M").nunique()
    )

    x = np.arange(len(daily))

    if month_count <= 1:
        # One-month analyses: show every available day.
        tick_mask = valid.to_numpy()

        tick_positions = x[tick_mask]

        tick_labels = [
            str(int(date.day))
            for date in dates.loc[valid]
        ]

        rotation = 0

    else:
        # Multi-month analyses: keep only the 1st and 15th.
        tick_mask = (
            valid
            & dates.dt.day.isin([1, 15])
        ).to_numpy()

        tick_positions = x[tick_mask]

        tick_dates = dates.loc[
            valid
            & dates.dt.day.isin([1, 15])
        ]

        tick_labels = [
            date.strftime("%b %-d")
            for date in tick_dates
        ]

        rotation = 0

        # Fallback for platforms where %-d is unsupported.
        if any("%" in label for label in tick_labels):
            tick_labels = [
                f"{date.strftime('%b')} {int(date.day)}"
                for date in tick_dates
            ]

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels,
        rotation=rotation,
        ha="center",
    )

    ax.set_xlabel(
        "Date (UTC)"
    )


def save_figure(
    fig,
    output_base: Path,
):
    fig.savefig(
        output_base.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        output_base.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def plot_state_distribution(
    dataframe: pd.DataFrame,
    colours: dict[str, str],
    title: str,
    output_base: Path,
):
    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.5,
        )
    )

    x = np.arange(
        len(
            dataframe
        )
    )

    values = dataframe[
        "percentage"
    ].to_numpy(
        dtype=float
    )

    bar_colours = [
        colours[
            state
        ]
        for state
        in dataframe[
            "state"
        ]
    ]

    bars = ax.bar(
        x,
        values,
        color=bar_colours,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        dataframe[
            "display_state"
        ]
    )

    ax.set_ylabel(
        "Share of observations (%)"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            float(
                np.nanmax(
                    values
                )
            )
            * 1.14,
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, (_, row) in zip(
        bars,
        dataframe.iterrows(),
    ):
        percentage = float(
            row[
                "percentage"
            ]
        )

        count = int(
            row[
                "count"
            ]
        )

        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height(),
            f"{percentage:.3f}%\n({count:,})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_base,
    )


def plot_non_green_rate(
    dataframe: pd.DataFrame,
    colours: dict[str, str],
    title: str,
    output_base: Path,
):
    non_green = (
        dataframe.loc[
            dataframe[
                "state"
            ]
            != STATE_NORMAL
        ]
        .copy()
    )

    total = int(
        dataframe[
            "count"
        ].sum()
    )

    non_green[
        "per_10000"
    ] = (
        10_000.0
        * non_green[
            "count"
        ]
        / total
        if total > 0
        else np.nan
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.0,
        )
    )

    bars = ax.bar(
        non_green[
            "display_state"
        ],
        non_green[
            "per_10000"
        ],
        color=[
            colours[
                state
            ]
            for state
            in non_green[
                "state"
            ]
        ],
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_ylabel(
        "Observations per 10,000 pair-observations"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ymax = (
        float(
            non_green[
                "per_10000"
            ].max()
        )
        if not non_green.empty
        else 1.0
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            ymax
            * 1.15,
        ),
    )

    for bar, value in zip(
        bars,
        non_green[
            "per_10000"
        ],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_base,
    )


def plot_daily_current_snapshot_share(
    daily: pd.DataFrame,
    colours: dict[str, str],
    title: str,
    output_base: Path,
):
    required = {
        "snapshots",
        "current_snapshot_normal_pct",
        "current_snapshot_proximity_pct",
        "current_snapshot_high_risk_pct",
        "current_snapshot_conflict_pct",
    }

    if not required.issubset(
        daily.columns
    ):
        return

    fig, ax = plt.subplots(
        figsize=(
            13.0,
            6.0,
        )
    )

    x = np.arange(
        len(
            daily
        )
    )

    bottom = np.zeros(
        len(
            daily
        ),
        dtype=float,
    )

    for state in STATES:
        column = (
            f"current_snapshot_{state.lower()}_pct"
        )

        values = (
            daily[
                column
            ]
            .fillna(0.0)
            .to_numpy(
                dtype=float
            )
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=STATE_SHORT[
                state
            ],
            color=colours[
                state
            ],
            edgecolor="black",
            linewidth=0.3,
        )

        bottom += values

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        daily[
            "day"
        ],
        rotation=45,
        ha="right",
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Share of analysed snapshots (%)"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.13,
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_base,
    )


def plot_daily_episodes(
    daily: pd.DataFrame,
    colours: dict[str, str],
    title: str,
    output_base: Path,
):
    if daily.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            13.0,
            6.0,
        )
    )

    x = np.arange(
        len(
            daily
        )
    )

    bottom = np.zeros(
        len(
            daily
        ),
        dtype=float,
    )

    for state in (
        STATE_PROXIMITY,
        STATE_HIGH_RISK,
        STATE_CONFLICT,
    ):
        column = (
            f"episodes_{state.lower()}"
        )

        if column not in daily.columns:
            continue

        values = (
            daily[
                column
            ]
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=STATE_SHORT[
                state
            ],
            color=colours[
                state
            ],
            edgecolor="black",
            linewidth=0.4,
        )

        bottom += values

    set_adaptive_daily_xaxis(
        ax,
        daily,
    )

    ax.set_ylabel(
        "Independent encounter episodes"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Keep the legend outside the plotting area so it never overlaps
    # the title or the bars.
    ax.legend(
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        borderaxespad=0.0,
        frameon=True,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    # Reserve space on the right for the external legend.
    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            0.88,
            1.0,
        )
    )

    save_figure(
        fig,
        output_base,
    )


def plot_episode_duration_histogram(
    episodes: pd.DataFrame,
    colours: dict[str, str],
    title: str,
    output_base: Path,
):
    if episodes.empty:
        return

    values = (
        episodes[
            "duration_s"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        values
    ) == 0:
        return

    maximum = max(
        60.0,
        float(
            np.nanpercentile(
                values,
                99,
            )
        ),
    )

    bins = np.arange(
        0,
        maximum
        + 30,
        30,
    )

    if len(
        bins
    ) < 3:
        bins = np.array(
            [
                0,
                30,
                60,
            ]
        )

    fig, ax = plt.subplots(
        figsize=(
            10.0,
            5.8,
        )
    )

    for state in (
        STATE_PROXIMITY,
        STATE_HIGH_RISK,
        STATE_CONFLICT,
    ):
        subset = (
            episodes.loc[
                episodes[
                    "max_event_state"
                ]
                == state,
                "duration_s",
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(
            subset
        ) == 0:
            continue

        ax.hist(
            subset,
            bins=bins,
            alpha=0.55,
            label=STATE_SHORT[
                state
            ],
            color=colours[
                state
            ],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel(
        "Episode duration (s)"
    )

    ax.set_ylabel(
        "Number of episodes"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_base,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse OpenSky separation safety performance "
            "and create thesis-ready plots."
        )
    )

    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=(
            "Detector run directory name. "
            "Example: 20190101T000000_20190115T235950"
        ),
    )

    parser.add_argument(
        "--area",
        default=DEFAULT_AREA,
    )

    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=(
            "Human-readable controller/scenario label, "
            'e.g. "Manual control" or "Automated controller".'
        ),
    )

    parser.add_argument(
        "--profile",
        default="ENROUTE",
    )

    parser.add_argument(
        "--episode-gap-s",
        type=int,
        default=DEFAULT_EPISODE_GAP_S,
        help=(
            "Start a new encounter episode when the same pair "
            "has no event for more than this number of seconds."
        ),
    )

    parser.add_argument(
        "--snapshot-step-s",
        type=int,
        default=DEFAULT_SNAPSHOT_STEP_S,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=READ_CHUNK_ROWS,
    )

    parser.add_argument(
        "--no-traffic-scan",
        action="store_true",
        help=(
            "Skip scanning opensky_10s. Faster, but disables "
            "aircraft-exposure and daily normalized plots."
        ),
    )

    args = parser.parse_args()

    if args.episode_gap_s <= 0:
        parser.error(
            "--episode-gap-s must be positive"
        )

    if args.snapshot_step_s <= 0:
        parser.error(
            "--snapshot-step-s must be positive"
        )

    if args.chunksize <= 0:
        parser.error(
            "--chunksize must be positive"
        )

    return args


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    run_dir = (
        SEPARATION_ROOT
        / args.area
        / "runs"
        / args.run_name
    ).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist:\n{run_dir}"
        )

    output_dir = (
        run_dir
        / "quality_analysis"
        / safe_slug(
            args.label
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    colours = load_state_colours(
        args.profile
    )

    run_summary = load_run_summary(
        run_dir
    )

    events = load_events(
        run_dir
    )

    snapshots_total = int(
        run_summary[
            "snapshots"
        ]
    )

    pairs_total = int(
        run_summary[
            "pairs_total"
        ]
    )

    print()
    print(
        "=" * 80
    )
    print(
        "ATC SAFETY PERFORMANCE ANALYSIS"
    )
    print(
        "=" * 80
    )
    print(
        f"Label         : {args.label}"
    )
    print(
        f"Area          : {args.area}"
    )
    print(
        f"Run           : {args.run_name}"
    )
    print(
        f"Snapshots     : {snapshots_total:,}"
    )
    print(
        f"Pair obs.     : {pairs_total:,}"
    )
    print(
        f"Event rows    : {len(events):,}"
    )
    print(
        f"Episode gap   : {args.episode_gap_s} s"
    )
    print(
        "=" * 80
    )

    # -------------------------------------------------------------------------
    # Main distributions.
    # -------------------------------------------------------------------------

    pair_current = pair_current_distribution(
        events,
        pairs_total,
    )

    pair_event = pair_event_distribution(
        events,
        pairs_total,
    )

    snapshot_current = snapshot_distribution(
        events,
        snapshots_total,
        "state_current",
    )

    snapshot_event = snapshot_distribution(
        events,
        snapshots_total,
        "event_state",
    )

    episodes = build_episodes(
        events,
        gap_s=args.episode_gap_s,
        snapshot_step_s=args.snapshot_step_s,
    )

    # -------------------------------------------------------------------------
    # Traffic denominators.
    # -------------------------------------------------------------------------

    traffic = None
    snapshot_stats = None
    aircraft_current = None

    start = pd.to_datetime(
        run_summary[
            "start_utc"
        ],
        utc=True,
    )

    end = pd.to_datetime(
        run_summary[
            "end_utc_inclusive"
        ],
        utc=True,
    )

    if not args.no_traffic_scan:
        try:
            (
                traffic,
                snapshot_stats,
            ) = scan_traffic(
                area=args.area,
                start=start,
                end=end,
                chunksize=args.chunksize,
            )

            aircraft_current = aircraft_current_distribution(
                traffic,
                events,
            )

            scanned_snapshots = int(
                snapshot_stats[
                    "timestamp"
                ].nunique()
            )

            scanned_pairs = int(
                snapshot_stats[
                    "pair_count"
                ].sum()
            )

            print()
            print(
                "Traffic-scan consistency check:"
            )
            print(
                f"  snapshots run/scanned : "
                f"{snapshots_total:,} / {scanned_snapshots:,}"
            )
            print(
                f"  pairs run/scanned     : "
                f"{pairs_total:,} / {scanned_pairs:,}"
            )

        except Exception as error:
            print()
            print(
                "WARNING: traffic scan failed."
            )
            print(
                str(
                    error
                )
            )
            print(
                "Continuing with detector-only metrics."
            )

            traffic = None
            snapshot_stats = None
            aircraft_current = None

    # -------------------------------------------------------------------------
    # Daily table.
    # -------------------------------------------------------------------------

    daily = build_daily_metrics(
        events=events,
        episodes=episodes,
        snapshot_stats=snapshot_stats,
    )

    # -------------------------------------------------------------------------
    # Summary KPI table.
    # -------------------------------------------------------------------------

    summary = build_summary_metrics(
        label=args.label,
        run_summary=run_summary,
        pair_current=pair_current,
        pair_event=pair_event,
        snapshot_current=snapshot_current,
        snapshot_event=snapshot_event,
        episodes=episodes,
        aircraft_current=aircraft_current,
        traffic=traffic,
    )

    # -------------------------------------------------------------------------
    # Save CSVs.
    # -------------------------------------------------------------------------

    pair_current.to_csv(
        output_dir
        / "state_distribution_pair_current.csv",
        index=False,
    )

    pair_event.to_csv(
        output_dir
        / "state_distribution_pair_event.csv",
        index=False,
    )

    snapshot_current.to_csv(
        output_dir
        / "state_distribution_snapshot_current.csv",
        index=False,
    )

    snapshot_event.to_csv(
        output_dir
        / "state_distribution_snapshot_event.csv",
        index=False,
    )

    episodes.to_csv(
        output_dir
        / "episodes.csv",
        index=False,
    )

    daily.to_csv(
        output_dir
        / "daily_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "quality_metrics_summary.csv",
        index=False,
    )

    if aircraft_current is not None:
        aircraft_current.to_csv(
            output_dir
            / "aircraft_state_distribution_current.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Plots.
    # -------------------------------------------------------------------------

    plot_state_distribution(
        pair_current,
        colours,
        title="Current pair-state distribution",
        output_base=(
            output_dir
            / "01_pair_current_state_distribution"
        ),
    )

    plot_non_green_rate(
        pair_current,
        colours,
        title="Current non-green pair rate",
        output_base=(
            output_dir
            / "02_pair_current_non_green_rate"
        ),
    )

    plot_state_distribution(
        pair_event,
        colours,
        title="Worst predicted pair-state distribution",
        output_base=(
            output_dir
            / "03_pair_event_state_distribution"
        ),
    )

    plot_state_distribution(
        snapshot_current,
        colours,
        title="Current safety-state exposure over time",
        output_base=(
            output_dir
            / "04_snapshot_current_time_distribution"
        ),
    )

    plot_state_distribution(
        snapshot_event,
        colours,
        title="Exposure to predicted safety events",
        output_base=(
            output_dir
            / "05_snapshot_event_time_distribution"
        ),
    )

    # The former daily 100%-stacked state-share figure is intentionally
    # omitted from the default report: Green dominates almost the entire
    # scale, so the non-green variation is visually unreadable and adds little
    # beyond the normalized/event figures already exported.

    plot_daily_episodes(
        daily,
        colours,
        title="Independent safety-event episodes per day",
        output_base=(
            output_dir
            / "06_daily_event_episodes"
        ),
    )

    plot_episode_duration_histogram(
        episodes,
        colours,
        title="Safety-event episode duration",
        output_base=(
            output_dir
            / "07_episode_duration_histogram"
        ),
    )

    if aircraft_current is not None:
        plot_state_distribution(
            aircraft_current,
            colours,
            title="Current state by aircraft observation",
            output_base=(
                output_dir
                / "08_aircraft_current_state_distribution"
            ),
        )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------

    def pct(
        dataframe: pd.DataFrame,
        state: str,
    ) -> float:
        values = dataframe.loc[
            dataframe[
                "state"
            ]
            == state,
            "percentage",
        ]

        return (
            float(
                values.iloc[0]
            )
            if len(
                values
            )
            else np.nan
        )

    print()
    print(
        "=" * 80
    )
    print(
        "KEY RESULTS"
    )
    print(
        "=" * 80
    )

    print(
        "CURRENT snapshot state "
        "(actual situation now):"
    )

    for state in STATES:
        print(
            f"  {STATE_SHORT[state]:<7}: "
            f"{pct(snapshot_current, state):8.3f}%"
        )

    print()
    print(
        "EVENT snapshot state "
        "(worst NOW + 0-1 + 1-2 + 2-5 min):"
    )

    for state in STATES:
        print(
            f"  {STATE_SHORT[state]:<7}: "
            f"{pct(snapshot_event, state):8.3f}%"
        )

    print()
    print(
        f"Independent event episodes: "
        f"{len(episodes):,}"
    )

    for state in (
        STATE_PROXIMITY,
        STATE_HIGH_RISK,
        STATE_CONFLICT,
    ):
        count = int(
            (
                episodes[
                    "max_event_state"
                ]
                == state
            ).sum()
        )

        print(
            f"  {STATE_SHORT[state]:<7}: "
            f"{count:,}"
        )

    print()
    print(
        "Output:"
    )
    print(
        output_dir
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
