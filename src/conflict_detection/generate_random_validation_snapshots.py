#!/usr/bin/env python3
"""
Generate random validation screenshots for OpenSky separation detection.

For each CURRENT safety state:
    - N random YELLOW / PROXIMITY moments
    - N random ORANGE / HIGH_RISK moments
    - N random RED / CONFLICT moments

The script:

1. Reads:
       yellow_proximity_events.csv
       orange_high_risk_events.csv
       red_loss_of_separation_events.csv

2. Combines the detector outputs.

3. Selects random UNIQUE timestamps according to state_current.

4. Reads the corresponding full traffic snapshots from:
       data/processed/opensky_10s/<AREA>/

5. Re-runs the standard separation analysis using opensky_safety.py.

6. Uses the existing opensky_plot.py renderer.

7. Highlights the specific aircraft pair being audited.

8. Shows outside the map:
       - validation colour/state
       - callsigns
       - horizontal separation
       - vertical separation

9. Reads the PNG resolution directly from bbox_presets.json.

10. Creates:
       selected_random_moments.csv
       validation_report.csv
       N yellow PNGs
       N orange PNGs
       N red PNGs

Expected layout:

PROJECT_ROOT/
├── data/
│   └── processed/
│       ├── opensky_10s/
│       │   └── LECMSAN/
│       └── opensky_separation/
│           └── LECMSAN/
│               └── runs/
│                   └── 20190101T000000_20190115T235950/
│
└── src/
    ├── acquisition/
    │   ├── area_presets.py
    │   ├── bbox_presets.json
    │   ├── opensky_plot.py
    │   ├── opensky_safety.py
    │   ├── separation_profiles.json
    │   ├── waypoints.json
    │   └── corridors.json
    │
    └── conflict_detection/
        └── generate_random_validation_snapshots.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def discover_acquisition_dir() -> Path:
    """
    Support both possible project layouts:

        PROJECT_ROOT/src/acquisition/
        PROJECT_ROOT/acquisition/

    The first one is the preferred/current layout.
    """

    candidates = (
        SRC_DIR / "acquisition",
        PROJECT_ROOT / "acquisition",
    )

    for candidate in candidates:

        if (
            (candidate / "bbox_presets.json").exists()
            and
            (candidate / "opensky_plot.py").exists()
            and
            (candidate / "opensky_safety.py").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "\nCould not locate the acquisition directory.\n"
        "Expected either:\n"
        f"  {SRC_DIR / 'acquisition'}\n"
        "or:\n"
        f"  {PROJECT_ROOT / 'acquisition'}\n"
    )


ACQUISITION_DIR = discover_acquisition_dir()

BBOX_PRESETS_FILE = (
    ACQUISITION_DIR
    / "bbox_presets.json"
)

WAYPOINTS_FILE = (
    ACQUISITION_DIR
    / "waypoints.json"
)

CORRIDORS_FILE = (
    ACQUISITION_DIR
    / "corridors.json"
)

SEPARATION_PROFILES_FILE = (
    ACQUISITION_DIR
    / "separation_profiles.json"
)

PROCESSED_10S_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opensky_10s"
)

SEPARATION_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "opensky_separation"
)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_AREA = "LECMSAN"

DEFAULT_PROFILE = "ENROUTE"

DEFAULT_RUN_NAME = (
    "20190101T000000_20190115T235950"
)

# Number of screenshots PER colour.
DEFAULT_N = 10

# None -> different random sample each run.
DEFAULT_SEED = None

READ_CHUNK_ROWS = 250_000

DEFAULT_RENDER_DPI = 160

DEFAULT_RENDER_WIDTH_PX = 3840

DEFAULT_RENDER_HEIGHT_PX = 2160

# Corridor overlay defaults for validation screenshots.
SHOW_VALIDATION_CORRIDORS = True
SHOW_VALIDATION_CORRIDOR_LABELS = True
VALIDATION_CORRIDORS_ONLY_NEAR_TRAFFIC = False
VALIDATION_CORRIDOR_TRAFFIC_DISTANCE_NM = 20.0


# =============================================================================
# CONVERSIONS
# =============================================================================

MPS_TO_KT = 1.9438444924406

MPS_TO_FPM = 196.8503937008

M_TO_FT = 3.2808398950131


# =============================================================================
# SAFETY STATES
# =============================================================================

STATE_NORMAL = "NORMAL"

STATE_PROXIMITY = "PROXIMITY"

STATE_HIGH_RISK = "HIGH_RISK"

STATE_CONFLICT = "CONFLICT"


STATE_TO_COLOUR_NAME = {

    STATE_PROXIMITY:
        "YELLOW",

    STATE_HIGH_RISK:
        "ORANGE",

    STATE_CONFLICT:
        "RED",
}


STATE_ORDER = {

    STATE_PROXIMITY:
        0,

    STATE_HIGH_RISK:
        1,

    STATE_CONFLICT:
        2,
}


EVENT_FILES = (

    "yellow_proximity_events.csv",

    "orange_high_risk_events.csv",

    "red_loss_of_separation_events.csv",
)


# =============================================================================
# LOCAL IMPORTS
# =============================================================================

if str(ACQUISITION_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ACQUISITION_DIR),
    )


from area_presets import (  # noqa: E402
    filter_dataframe_to_area,
    load_area_presets,
    load_waypoints_for_area,
    load_corridors_for_area,
    resolve_area_preset,
)

from opensky_safety import (  # noqa: E402
    analyse_aircraft_snapshot,
    enabled_states,
    load_safety_profile,
    relevant_pair_results,
    state_colours,
    state_display_names,
)

from opensky_plot import (  # noqa: E402
    build_shared_validation_render_config,
    render_map,
)


# =============================================================================
# RENDER CONFIGURATION
# =============================================================================

def get_render_configuration(
    area: dict,
) -> tuple[int, int, int, tuple[float, float]]:
    """
    Read the requested output resolution from bbox_presets.json.

    Example LECMSAN:

        width_px  = 3840
        height_px = 2160

    FIG_SIZE is derived from:

        pixels / DPI

    This guarantees that the final PNG has the dimensions
    requested by bbox_presets.json.
    """

    render = area.get(
        "render",
        {},
    )

    width_px = int(
        render.get(
            "width_px",
            DEFAULT_RENDER_WIDTH_PX,
        )
    )

    height_px = int(
        render.get(
            "height_px",
            DEFAULT_RENDER_HEIGHT_PX,
        )
    )

    dpi = DEFAULT_RENDER_DPI

    if width_px <= 0:

        raise ValueError(
            f"Invalid render width_px: {width_px}"
        )

    if height_px <= 0:

        raise ValueError(
            f"Invalid render height_px: {height_px}"
        )

    if dpi <= 0:

        raise ValueError(
            f"Invalid render DPI: {dpi}"
        )

    figsize = (

        width_px / dpi,

        height_px / dpi,
    )

    return (
        width_px,
        height_px,
        dpi,
        figsize,
    )


# =============================================================================
# TIMESTAMP CONVERSION
# =============================================================================

def to_epoch_seconds(
    series: pd.Series,
) -> pd.Series:
    """
    Convert timestamps to Unix epoch seconds.

    Supports:
        - seconds
        - milliseconds
        - microseconds
        - nanoseconds
        - numeric strings
        - datetime strings
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_original = (
        series.notna().sum()
    )

    valid_numeric = (
        numeric.notna().sum()
    )

    # If the column is mainly numeric, treat it as epoch time.
    if (
        valid_original > 0
        and
        valid_numeric
        >= 0.90 * valid_original
    ):

        numeric = (
            numeric.astype(
                "float64"
            )
        )

        finite = numeric[
            np.isfinite(
                numeric
            )
        ]

        if finite.empty:

            return numeric

        magnitude = float(
            finite
            .abs()
            .median()
        )

        # Nanoseconds
        if magnitude > 1e17:

            return (
                numeric
                / 1_000_000_000.0
            )

        # Microseconds
        if magnitude > 1e14:

            return (
                numeric
                / 1_000_000.0
            )

        # Milliseconds
        if magnitude > 1e11:

            return (
                numeric
                / 1_000.0
            )

        # Seconds
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


# =============================================================================
# DETECTOR OUTPUT
# =============================================================================

def load_detector_events(
    run_dir: Path,
) -> pd.DataFrame:
    """
    Read and combine yellow/orange/red detector CSVs.

    IMPORTANT:

    event_state
        = worst state during the prediction horizon.

    state_current
        = state at the actual screenshot timestamp.

    For visual validation we select using state_current.
    """

    frames = []

    required_columns = {

        "timestamp",

        "timestamp_utc",

        "state_current",

        "event_state",

        "icao24_a",

        "icao24_b",

        "horizontal_sep_nm",

        "vertical_sep_ft",
    }

    for filename in EVENT_FILES:

        path = (
            run_dir
            / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                "\nMissing detector output:\n"
                f"{path}\n"
            )

        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        print(
            f"{filename:<40}"
            f"{len(dataframe):>12,} rows"
        )

        if dataframe.empty:

            continue

        missing = (
            required_columns
            - set(
                dataframe.columns
            )
        )

        if missing:

            raise ValueError(
                f"\n{filename} is missing columns:\n"
                f"{sorted(missing)}\n"
            )

        dataframe = (
            dataframe.copy()
        )

        dataframe[
            "source_event_file"
        ] = filename

        dataframe[
            "timestamp"
        ] = to_epoch_seconds(
            dataframe[
                "timestamp"
            ]
        )

        dataframe = dataframe.loc[
            dataframe[
                "timestamp"
            ].notna()
        ].copy()

        dataframe[
            "timestamp"
        ] = (
            dataframe[
                "timestamp"
            ]
            .round()
            .astype(
                "int64"
            )
        )

        dataframe[
            "state_current"
        ] = (
            dataframe[
                "state_current"
            ]
            .astype(str)
            .str
            .strip()
            .str
            .upper()
        )

        dataframe[
            "event_state"
        ] = (
            dataframe[
                "event_state"
            ]
            .astype(str)
            .str
            .strip()
            .str
            .upper()
        )

        frames.append(
            dataframe
        )

    if not frames:

        raise RuntimeError(
            "All detector event files are empty."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =============================================================================
# RANDOM SELECTION
# =============================================================================

def select_random_state_samples(
    events: pd.DataFrame,
    state: str,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Select N random UNIQUE timestamps with the requested
    CURRENT safety state.

    If several pairs share the same timestamp, one pair
    is selected randomly.
    """

    candidates = events.loc[

        events[
            "state_current"
        ]
        == state

    ].copy()

    colour = (
        STATE_TO_COLOUR_NAME[
            state
        ]
    )

    if candidates.empty:

        print(
            f"{colour:<8}: "
            f"{0:>10,} "
            "unique current-state timestamps "
            "-> SKIPPED"
        )

        # Return an empty frame with the expected columns instead of failing.
        # This allows validation runs to continue when one or more safety
        # states are absent from the detector output.
        empty = candidates.copy()

        empty[
            "validation_state"
        ] = pd.Series(
            dtype="object"
        )

        empty[
            "validation_colour"
        ] = pd.Series(
            dtype="object"
        )

        return empty

    # Shuffle before removing duplicate timestamps.
    # This makes the selected pair random if several pairs
    # share the same timestamp.
    order = rng.permutation(
        len(candidates)
    )

    candidates = (

        candidates
        .iloc[
            order
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    candidates = (

        candidates
        .drop_duplicates(
            subset=[
                "timestamp"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    available = len(
        candidates
    )

    print(
        f"{colour:<8}: "
        f"{available:>10,} "
        "unique current-state timestamps"
    )

    sample_size = min(
        n,
        available,
    )

    if available < n:

        print(
            f"WARNING: only {available} unique timestamps "
            f"are available for {colour}; "
            f"using all {available} instead of {n}."
        )

    indices = rng.choice(
        available,
        size=sample_size,
        replace=False,
    )

    selected = (

        candidates
        .iloc[
            indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected[
        "validation_state"
    ] = state

    selected[
        "validation_colour"
    ] = colour

    return selected


def select_all_samples(
    events: pd.DataFrame,
    n: int,
    seed: int | None,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        seed
    )

    selections = []

    for state in (

        STATE_PROXIMITY,

        STATE_HIGH_RISK,

        STATE_CONFLICT,
    ):

        state_selection = (
            select_random_state_samples(
                events=events,
                state=state,
                n=n,
                rng=rng,
            )
        )

        if not state_selection.empty:

            selections.append(
                state_selection
            )

    if not selections:

        raise RuntimeError(
            "\nNo validation samples are available for "
            "PROXIMITY, HIGH_RISK or CONFLICT.\n"
        )

    selected = pd.concat(
        selections,
        ignore_index=True,
    )

    selected[
        "_state_order"
    ] = selected[
        "validation_state"
    ].map(
        STATE_ORDER
    )

    selected.sort_values(
        [
            "_state_order",
            "timestamp",
        ],
        inplace=True,
    )

    selected.reset_index(
        drop=True,
        inplace=True,
    )

    return selected


# =============================================================================
# FIND PROCESSED MONTH FILES
# =============================================================================

def find_month_files(
    area: str,
    timestamp: int,
) -> list[Path]:

    dt = pd.to_datetime(
        timestamp,
        unit="s",
        utc=True,
    )

    month = dt.strftime(
        "%Y-%m"
    )

    directory = (
        PROCESSED_10S_ROOT
        / area
    )

    if not directory.exists():

        raise FileNotFoundError(
            "\nProcessed 10-second directory not found:\n"
            f"{directory}\n"
        )

    matches = sorted(

        path

        for path
        in directory.rglob(
            f"*_{month}_10s.csv"
        )

        if "_daily_cache"
        not in path.parts
    )

    if not matches:

        raise FileNotFoundError(
            f"\nNo *_10s.csv file found for "
            f"{month} under:\n"
            f"{directory}\n"
        )

    return matches


# =============================================================================
# READ ONLY REQUIRED SNAPSHOTS
# =============================================================================

def load_selected_snapshots(
    area: str,
    timestamps: list[int],
    chunksize: int,
) -> dict[int, pd.DataFrame]:

    files_to_timestamps: dict[
        Path,
        set[int],
    ] = {}

    for timestamp in timestamps:

        month_files = find_month_files(
            area,
            timestamp,
        )

        for path in month_files:

            files_to_timestamps.setdefault(
                path,
                set(),
            ).add(
                int(
                    timestamp
                )
            )

    pieces: dict[
        int,
        list[pd.DataFrame],
    ] = {

        int(timestamp):
            []

        for timestamp
        in timestamps
    }

    for (
        path,
        wanted_timestamps,
    ) in files_to_timestamps.items():

        print()
        print(
            "-" * 80
        )

        print(
            "Reading processed traffic:"
        )

        print(
            path
        )

        print(
            f"Looking for "
            f"{len(wanted_timestamps)} "
            "selected timestamps..."
        )

        for chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
        ):

            if "time" not in chunk.columns:

                raise ValueError(
                    f"\nNo 'time' column in:\n"
                    f"{path}\n"
                )

            time_seconds = (
                to_epoch_seconds(
                    chunk[
                        "time"
                    ]
                )
                .round()
            )

            mask = (
                time_seconds.isin(
                    wanted_timestamps
                )
            )

            if not mask.any():

                continue

            selected_chunk = (
                chunk.loc[
                    mask
                ]
                .copy()
            )

            selected_chunk[
                "_selected_timestamp"
            ] = (
                time_seconds
                .loc[
                    mask
                ]
                .astype(
                    "int64"
                )
            )

            for (
                timestamp,
                group,
            ) in selected_chunk.groupby(
                "_selected_timestamp"
            ):

                group = group.drop(
                    columns=[
                        "_selected_timestamp"
                    ]
                )

                pieces[
                    int(
                        timestamp
                    )
                ].append(
                    group
                )

    snapshots = {}

    for (
        timestamp,
        timestamp_pieces,
    ) in pieces.items():

        if not timestamp_pieces:

            print(
                "WARNING: no traffic rows found for "
                f"{pd.to_datetime(timestamp, unit='s', utc=True)}"
            )

            continue

        snapshot = pd.concat(
            timestamp_pieces,
            ignore_index=True,
        )

        # Keep one row per aircraft.
        if (
            "lastcontact"
            in snapshot.columns
        ):

            snapshot[
                "_lastcontact_numeric"
            ] = pd.to_numeric(
                snapshot[
                    "lastcontact"
                ],
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
            subset=[
                "icao24"
            ],
            keep="last",
            inplace=True,
        )

        snapshot.drop(
            columns=[
                "_lastcontact_numeric"
            ],
            errors="ignore",
            inplace=True,
        )

        snapshot.reset_index(
            drop=True,
            inplace=True,
        )

        snapshots[
            timestamp
        ] = snapshot

    return snapshots


# =============================================================================
# DATA NORMALIZATION FOR opensky_plot.py
# =============================================================================

def first_numeric_column(
    dataframe: pd.DataFrame,
    names: tuple[str, ...],
) -> pd.Series:

    for name in names:

        if name in dataframe.columns:

            return pd.to_numeric(
                dataframe[
                    name
                ],
                errors="coerce",
            )

    return pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float,
    )


def boolean_series(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (
            series
            .fillna(False)
            .astype(bool)
        )

    return (
        series
        .astype(str)
        .str
        .strip()
        .str
        .lower()
        .isin(
            {
                "true",
                "1",
                "1.0",
                "yes",
            }
        )
    )


def prepare_snapshot_for_plot(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = (
        snapshot.copy()
    )

    # -------------------------------------------------------------------------
    # Position
    # -------------------------------------------------------------------------

    dataframe[
        "latitude"
    ] = first_numeric_column(
        dataframe,
        (
            "latitude",
            "lat",
        ),
    )

    dataframe[
        "longitude"
    ] = first_numeric_column(
        dataframe,
        (
            "longitude",
            "lon",
        ),
    )

    # -------------------------------------------------------------------------
    # Track
    # -------------------------------------------------------------------------

    dataframe[
        "true_track"
    ] = first_numeric_column(
        dataframe,
        (
            "true_track",
            "heading",
        ),
    )

    # -------------------------------------------------------------------------
    # Speed
    # -------------------------------------------------------------------------

    dataframe[
        "velocity"
    ] = first_numeric_column(
        dataframe,
        (
            "velocity",
        ),
    )

    if (
        dataframe[
            "velocity"
        ]
        .isna()
        .all()

        and

        "velocity_kts"
        in dataframe.columns
    ):

        dataframe[
            "velocity"
        ] = (
            pd.to_numeric(
                dataframe[
                    "velocity_kts"
                ],
                errors="coerce",
            )
            / MPS_TO_KT
        )

    dataframe[
        "velocity_kts"
    ] = (
        dataframe[
            "velocity"
        ]
        * MPS_TO_KT
    )

    # -------------------------------------------------------------------------
    # Altitude
    # -------------------------------------------------------------------------

    altitude_ft = first_numeric_column(
        dataframe,
        (
            "baro_altitude_ft",
        ),
    )

    if altitude_ft.isna().all():

        altitude_m = (
            first_numeric_column(
                dataframe,
                (
                    "baro_altitude",
                    "baroaltitude",
                ),
            )
        )

        altitude_ft = (
            altitude_m
            * M_TO_FT
        )

    dataframe[
        "baro_altitude_ft"
    ] = altitude_ft

    # -------------------------------------------------------------------------
    # Vertical rate
    # -------------------------------------------------------------------------

    vertical_rate_fpm = (
        first_numeric_column(
            dataframe,
            (
                "vertical_rate_fpm",
            ),
        )
    )

    if (
        vertical_rate_fpm
        .isna()
        .all()
    ):

        vertical_rate_ms = (
            first_numeric_column(
                dataframe,
                (
                    "vertical_rate",
                    "vertrate",
                ),
            )
        )

        vertical_rate_fpm = (
            vertical_rate_ms
            * MPS_TO_FPM
        )

    dataframe[
        "vertical_rate_fpm"
    ] = vertical_rate_fpm

    # -------------------------------------------------------------------------
    # Callsign
    # -------------------------------------------------------------------------

    if (
        "callsign"
        not in dataframe.columns
    ):

        dataframe[
            "callsign"
        ] = ""

    dataframe[
        "callsign"
    ] = (
        dataframe[
            "callsign"
        ]
        .fillna("")
        .astype(str)
        .str
        .strip()
    )

    # -------------------------------------------------------------------------
    # Ground
    # -------------------------------------------------------------------------

    if (
        "on_ground"
        in dataframe.columns
    ):

        dataframe[
            "on_ground"
        ] = boolean_series(
            dataframe[
                "on_ground"
            ]
        )

    elif (
        "onground"
        in dataframe.columns
    ):

        dataframe[
            "on_ground"
        ] = boolean_series(
            dataframe[
                "onground"
            ]
        )

    else:

        dataframe[
            "on_ground"
        ] = False

    # -------------------------------------------------------------------------
    # Valid position
    # -------------------------------------------------------------------------

    dataframe = dataframe.loc[

        dataframe[
            "latitude"
        ].notna()

        &

        dataframe[
            "longitude"
        ].notna()

    ].copy()

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    return dataframe


# =============================================================================
# PAIR HELPERS
# =============================================================================

def canonical_pair(
    icao_a: str,
    icao_b: str,
) -> tuple[str, str]:

    return tuple(
        sorted(
            (
                str(
                    icao_a
                )
                .strip()
                .lower(),

                str(
                    icao_b
                )
                .strip()
                .lower(),
            )
        )
    )


def find_pair_result(
    pair_results: list[dict],
    icao_a: str,
    icao_b: str,
) -> dict | None:

    wanted = canonical_pair(
        icao_a,
        icao_b,
    )

    for result in pair_results:

        current = canonical_pair(
            result.get(
                "icao24_a",
                "",
            ),
            result.get(
                "icao24_b",
                "",
            ),
        )

        if current == wanted:

            return result

    return None


# =============================================================================
# VALIDATION OVERLAY
# =============================================================================

def draw_validation_overlay(
    fig,
    ax,
    snapshot: pd.DataFrame,
    selected_row: pd.Series,
    pair_result: dict | None,
    colour: str,
) -> bool:
    """
    Highlight the selected aircraft pair.

    The validation information panel is deliberately placed
    OUTSIDE the map axes.

    Therefore it cannot overlap:
        - aircraft
        - aircraft labels
        - cities
        - waypoints
        - trajectory arrows
        - safety circles

    The line connecting the aircraft is placed below the
    trajectory arrows and aircraft symbols.
    """

    wanted_a = (
        str(
            selected_row[
                "icao24_a"
            ]
        )
        .strip()
        .lower()
    )

    wanted_b = (
        str(
            selected_row[
                "icao24_b"
            ]
        )
        .strip()
        .lower()
    )

    aircraft_ids = (
        snapshot[
            "icao24"
        ]
        .astype(str)
        .str
        .strip()
        .str
        .lower()
    )

    rows_a = snapshot.loc[
        aircraft_ids
        == wanted_a
    ]

    rows_b = snapshot.loc[
        aircraft_ids
        == wanted_b
    ]

    if (
        rows_a.empty
        or
        rows_b.empty
    ):

        print(
            "WARNING: selected pair not found "
            "in reconstructed snapshot:"
        )

        print(
            wanted_a,
            wanted_b,
        )

        return False

    aircraft_a = (
        rows_a.iloc[0]
    )

    aircraft_b = (
        rows_b.iloc[0]
    )

    lon_a = float(
        aircraft_a[
            "longitude"
        ]
    )

    lat_a = float(
        aircraft_a[
            "latitude"
        ]
    )

    lon_b = float(
        aircraft_b[
            "longitude"
        ]
    )

    lat_b = float(
        aircraft_b[
            "latitude"
        ]
    )

    # -------------------------------------------------------------------------
    # Pair line.
    #
    # opensky_plot trajectory vectors are zorder=14.
    # aircraft symbols are zorder=20.
    #
    # This line therefore remains underneath both.
    # -------------------------------------------------------------------------

    ax.plot(
        [
            lon_a,
            lon_b,
        ],
        [
            lat_a,
            lat_b,
        ],
        color=colour,
        linewidth=2.0,
        linestyle="--",
        alpha=0.70,
        zorder=13.5,
        solid_capstyle="round",
    )

    # -------------------------------------------------------------------------
    # Values
    # -------------------------------------------------------------------------

    if pair_result is not None:

        horizontal_nm = (
            pair_result.get(
                "horizontal_sep_nm"
            )
        )

        vertical_ft = (
            pair_result.get(
                "vertical_sep_ft"
            )
        )

        recomputed_state = (
            pair_result.get(
                "state_current"
            )
        )

    else:

        horizontal_nm = (
            selected_row.get(
                "horizontal_sep_nm"
            )
        )

        vertical_ft = (
            selected_row.get(
                "vertical_sep_ft"
            )
        )

        recomputed_state = (
            selected_row.get(
                "state_current"
            )
        )

    # -------------------------------------------------------------------------
    # Callsigns
    # -------------------------------------------------------------------------

    callsign_a = str(
        selected_row.get(
            "callsign_a",
            "",
        )
        or ""
    ).strip()

    callsign_b = str(
        selected_row.get(
            "callsign_b",
            "",
        )
        or ""
    ).strip()

    if not callsign_a:

        callsign_a = (
            wanted_a.upper()
        )

    if not callsign_b:

        callsign_b = (
            wanted_b.upper()
        )

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    if (
        horizontal_nm is None
        or
        pd.isna(
            horizontal_nm
        )
    ):

        horizontal_text = (
            "N/A"
        )

    else:

        horizontal_text = (
            f"{float(horizontal_nm):.2f} NM"
        )

    if (
        vertical_ft is None
        or
        pd.isna(
            vertical_ft
        )
    ):

        vertical_text = (
            "N/A"
        )

    else:

        vertical_text = (
            f"{float(vertical_ft):.0f} ft"
        )

    colour_name = str(
        selected_row[
            "validation_colour"
        ]
    )

    validation_text = (
        f"{colour_name} VALIDATION"
        f"   |   "
        f"{callsign_a} ↔ {callsign_b}"
        f"   |   "
        f"H = {horizontal_text}"
        f"   |   "
        f"V = {vertical_text}"
        f"   |   "
        f"State = {recomputed_state}"
    )

    # -------------------------------------------------------------------------
    # Move original title slightly upwards.
    # -------------------------------------------------------------------------

    if (
        fig._suptitle
        is not None
    ):

        fig._suptitle.set_y(
            0.979
        )

    # -------------------------------------------------------------------------
    # Validation panel.
    #
    # render_map() leaves the top of the axes around y=0.92.
    #
    # This panel is centred at y=0.944 and therefore sits entirely
    # in the free strip between the map and the main title.
    # -------------------------------------------------------------------------

    fig.text(
        0.5,
        0.944,

        validation_text,

        ha="center",
        va="center",

        fontsize=10.5,

        fontweight="bold",

        color="white",

        bbox={
            "boxstyle":
                "round,pad=0.38",

            "facecolor":
                "black",

            "edgecolor":
                colour,

            "linewidth":
                2.2,

            "alpha":
                0.95,
        },

        zorder=100,
    )

    return True


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate random OpenSky "
            "yellow/orange/red validation screenshots."
        )
    )

    parser.add_argument(
        "--run-name",

        default=
            DEFAULT_RUN_NAME,

        help=(
            "Separation detector run directory name. "
            "Example: "
            "20190101T000000_20190115T235950"
        ),
    )

    parser.add_argument(
        "--area",

        default=
            DEFAULT_AREA,

        help=(
            "Study area. "
            f"Default: {DEFAULT_AREA}"
        ),
    )

    parser.add_argument(
        "--profile",

        default=
            DEFAULT_PROFILE,

        help=(
            "Separation profile. "
            f"Default: {DEFAULT_PROFILE}"
        ),
    )

    parser.add_argument(
        "--n",

        type=int,

        default=
            DEFAULT_N,

        help=(
            "Number of random CURRENT-state "
            "screenshots PER colour. "
            f"Default: {DEFAULT_N}"
        ),
    )

    parser.add_argument(
        "--seed",

        type=int,

        default=
            DEFAULT_SEED,

        help=(
            "Optional random seed. "
            "Omit for a different random selection each run. "
            "Example: --seed 42"
        ),
    )

    parser.add_argument(
        "--chunksize",

        type=int,

        default=
            READ_CHUNK_ROWS,

        help=(
            "CSV read chunksize. "
            f"Default: {READ_CHUNK_ROWS}"
        ),
    )

    args = parser.parse_args()

    if args.n <= 0:

        parser.error(
            "--n must be positive"
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

    # -------------------------------------------------------------------------
    # Run directory
    # -------------------------------------------------------------------------

    run_dir = (
        SEPARATION_OUTPUT_ROOT
        / args.area
        / "runs"
        / args.run_name
    ).resolve()

    if not run_dir.exists():

        raise FileNotFoundError(
            "\nRun directory does not exist:\n"
            f"{run_dir}\n"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "RANDOM OPEN-SKY VALIDATION SCREENSHOTS"
    )

    print(
        "=" * 80
    )

    print(
        f"PROJECT_ROOT  : {PROJECT_ROOT}"
    )

    print(
        f"Acquisition   : {ACQUISITION_DIR}"
    )

    print(
        f"Run name      : {args.run_name}"
    )

    print(
        f"Run directory : {run_dir}"
    )

    print(
        f"Area          : {args.area}"
    )

    print(
        f"Profile       : {args.profile}"
    )

    print(
        f"Per colour    : up to {args.n}"
    )

    print(
        f"Requested max : {args.n * 3} PNGs"
    )

    print(
        f"Random seed   : {args.seed}"
    )

    print(
        "=" * 80
    )

    # =========================================================================
    # AREA
    # =========================================================================

    presets = load_area_presets(
        BBOX_PRESETS_FILE
    )

    area = resolve_area_preset(
        args.area,
        presets,
    )

    bbox = (
        area[
            "bbox"
        ]
    )

    # -------------------------------------------------------------------------
    # Render resolution from bbox_presets.json
    # -------------------------------------------------------------------------

    (
        render_width_px,
        render_height_px,
        render_dpi,
        render_figsize,

    ) = get_render_configuration(
        area
    )

    print()
    print(
        "Render configuration:"
    )

    print(
        f"  width       : "
        f"{render_width_px} px"
    )

    print(
        f"  height      : "
        f"{render_height_px} px"
    )

    print(
        f"  DPI         : "
        f"{render_dpi}"
    )

    print(
        f"  figure size : "
        f"{render_figsize[0]:.2f} x "
        f"{render_figsize[1]:.2f} in"
    )

    # -------------------------------------------------------------------------
    # Waypoints
    # -------------------------------------------------------------------------

    waypoints = (
        load_waypoints_for_area(
            WAYPOINTS_FILE,
            area_key=args.area,
            area=area,
        )
    )

    print()
    print(
        f"Loaded waypoints: "
        f"{len(waypoints)}"
    )

    # -------------------------------------------------------------------------
    # ATS corridors / airways
    # -------------------------------------------------------------------------

    corridors = (
        load_corridors_for_area(
            CORRIDORS_FILE,
            area_key=args.area,
            area=area,
        )
    )

    print(
        f"Loaded corridors: "
        f"{len(corridors)}"
    )

    if SHOW_VALIDATION_CORRIDORS and not corridors:
        print(
            "WARNING: corridor display is enabled, "
            "but no corridors were loaded."
        )

    print(
        "Corridor rendering: "
        f"show={SHOW_VALIDATION_CORRIDORS} | "
        f"labels={SHOW_VALIDATION_CORRIDOR_LABELS} | "
        f"only_near_traffic={VALIDATION_CORRIDORS_ONLY_NEAR_TRAFFIC} | "
        f"distance_nm={VALIDATION_CORRIDOR_TRAFFIC_DISTANCE_NM}"
    )

    # =========================================================================
    # SAFETY PROFILE
    # =========================================================================

    profile = load_safety_profile(
        SEPARATION_PROFILES_FILE,
        args.profile,
    )

    state_rules = (
        profile[
            "states"
        ]
    )

    red_rule = (
        state_rules[
            STATE_CONFLICT
        ]
    )

    min_horizontal_nm = float(
        red_rule[
            "horizontal_nm"
        ]
    )

    min_vertical_ft = float(
        red_rule[
            "vertical_ft"
        ]
    )

    safety_colours = (
        state_colours(
            state_rules
        )
    )

    safety_names = (
        state_display_names(
            state_rules
        )
    )

    # =========================================================================
    # LOAD DETECTOR EVENTS
    # =========================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "LOADING DETECTOR EVENTS"
    )

    print(
        "=" * 80
    )

    events = load_detector_events(
        run_dir
    )

    print()
    print(
        f"Combined detector rows: "
        f"{len(events):,}"
    )

    # =========================================================================
    # RANDOM SELECTION
    # =========================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "RANDOM CURRENT-STATE SELECTION"
    )

    print(
        "=" * 80
    )

    selected = select_all_samples(
        events=events,
        n=args.n,
        seed=args.seed,
    )

    print()
    print(
        f"Actual selected samples: "
        f"{len(selected)}"
    )

    # =========================================================================
    # OUTPUT DIRECTORY
    # =========================================================================

    if args.seed is None:

        validation_run_name = (
            "random_"
            + datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

    else:

        validation_run_name = (
            f"random_seed_{args.seed}"
        )

    output_dir = (
        run_dir
        / "validation_screenshots"
        / validation_run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_csv = (
        output_dir
        / "selected_random_moments.csv"
    )

    selected.drop(
        columns=[
            "_state_order"
        ],
        errors="ignore",
    ).to_csv(
        selected_csv,
        index=False,
    )

    print()
    print(
        "Random selection saved to:"
    )

    print(
        selected_csv
    )

    # =========================================================================
    # LOAD ORIGINAL TRAFFIC SNAPSHOTS
    # =========================================================================

    timestamps = (
        selected[
            "timestamp"
        ]
        .astype(int)
        .tolist()
    )

    snapshots = (
        load_selected_snapshots(
            area=args.area,
            timestamps=timestamps,
            chunksize=args.chunksize,
        )
    )

    # =========================================================================
    # PLOT CONFIGURATION
    # =========================================================================

    config = build_shared_validation_render_config(
        area,
        width_px=render_width_px,
        height_px=render_height_px,
        dpi=render_dpi,
        title_prefix="OpenSky Traffic Snapshot",
        overrides={
            "SEPARATION_PROFILE": args.profile,
            "STATE_NORMAL": STATE_NORMAL,
            "STATE_PROXIMITY": STATE_PROXIMITY,
            "STATE_HIGH_RISK": STATE_HIGH_RISK,
            "STATE_CONFLICT": STATE_CONFLICT,
            "ENABLED_SAFETY_STATES": enabled_states(state_rules),
            "SAFETY_STATE_COLORS": safety_colours,
            "SAFETY_STATE_DISPLAY_NAMES": safety_names,
            "NORMAL_AIRCRAFT_COLOR": safety_colours[STATE_NORMAL],
            "PROXIMITY_AIRCRAFT_COLOR": safety_colours[STATE_PROXIMITY],
            "HIGH_RISK_AIRCRAFT_COLOR": safety_colours[STATE_HIGH_RISK],
            "CONFLICT_AIRCRAFT_COLOR": safety_colours[STATE_CONFLICT],
            "WARNING_AIRCRAFT_COLOR": safety_colours[STATE_PROXIMITY],
            "MIN_HORIZONTAL_SEPARATION_NM": min_horizontal_nm,
            "MIN_VERTICAL_SEPARATION_FT": min_vertical_ft,
            "LOOKAHEAD_MINUTES": [1, 2, 5],
            "LOOKAHEAD_INTERVAL_STATES": {},
            "SHOW_CORRIDORS": SHOW_VALIDATION_CORRIDORS,
            "CORRIDORS_ONLY_NEAR_TRAFFIC": VALIDATION_CORRIDORS_ONLY_NEAR_TRAFFIC,
            "CORRIDOR_TRAFFIC_DISTANCE_NM": VALIDATION_CORRIDOR_TRAFFIC_DISTANCE_NM,
            "SHOW_CORRIDOR_LABELS": SHOW_VALIDATION_CORRIDOR_LABELS,
        },
    )

    # =========================================================================
    # GENERATE SCREENSHOTS
    # =========================================================================

    counters = {

        STATE_PROXIMITY:
            0,

        STATE_HIGH_RISK:
            0,

        STATE_CONFLICT:
            0,
    }

    report_rows = []

    total_requested = len(
        selected
    )

    generated = 0

    print()
    print(
        "=" * 80
    )

    print(
        "GENERATING SCREENSHOTS"
    )

    print(
        "=" * 80
    )

    for (
        _,
        selected_row,
    ) in selected.iterrows():

        timestamp = int(
            selected_row[
                "timestamp"
            ]
        )

        validation_state = str(
            selected_row[
                "validation_state"
            ]
        )

        counters[
            validation_state
        ] += 1

        sequence_number = (
            counters[
                validation_state
            ]
        )

        dt = pd.to_datetime(
            timestamp,
            unit="s",
            utc=True,
        )

        colour_name = (
            STATE_TO_COLOUR_NAME[
                validation_state
            ]
        )

        print()
        print(
            "-" * 80
        )

        print(
            f"{colour_name} "
            f"{sequence_number:02d}/{args.n}"
        )

        print(
            dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )

        print(
            f"{selected_row.get('callsign_a', '')} "
            f"<-> "
            f"{selected_row.get('callsign_b', '')}"
        )

        # ---------------------------------------------------------------------
        # Original snapshot
        # ---------------------------------------------------------------------

        if timestamp not in snapshots:

            print(
                "SKIPPED: traffic snapshot "
                "was not found."
            )

            continue

        snapshot = (
            prepare_snapshot_for_plot(
                snapshots[
                    timestamp
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Exact polygon
        # ---------------------------------------------------------------------

        snapshot = (
            filter_dataframe_to_area(
                snapshot,
                area,
                latitude_column=
                    "latitude",
                longitude_column=
                    "longitude",
            )
        )

        snapshot.reset_index(
            drop=True,
            inplace=True,
        )

        # ---------------------------------------------------------------------
        # Recalculate safety state
        # ---------------------------------------------------------------------

        (
            pair_results,
            aircraft_states,
            interval_states,
            skipped_pairs,

        ) = analyse_aircraft_snapshot(

            snapshot,

            min_horizontal_nm=
                min_horizontal_nm,

            min_vertical_ft=
                min_vertical_ft,

            state_rules=
                state_rules,

            lookahead_minutes=
                (
                    1,
                    2,
                    5,
                ),

            prediction_step_seconds=
                5,

            check_ground_aircraft=
                False,
        )

        # Trajectory segment colours.
        config[
            "LOOKAHEAD_INTERVAL_STATES"
        ] = interval_states

        # ---------------------------------------------------------------------
        # Current relevant pairs
        # ---------------------------------------------------------------------

        relevant_pairs = (
            relevant_pair_results(
                pair_results
            )
        )

        warnings = [

            result

            for result
            in relevant_pairs

            if result.get(
                "state_current"
            )
            in {
                STATE_PROXIMITY,
                STATE_HIGH_RISK,
            }
        ]

        conflicts = [

            result

            for result
            in relevant_pairs

            if result.get(
                "state_current"
            )
            == STATE_CONFLICT
        ]

        # ---------------------------------------------------------------------
        # Target pair
        # ---------------------------------------------------------------------

        pair_result = (
            find_pair_result(
                pair_results,
                selected_row[
                    "icao24_a"
                ],
                selected_row[
                    "icao24_b"
                ],
            )
        )

        # ---------------------------------------------------------------------
        # Filename
        # ---------------------------------------------------------------------

        filename = (
            f"{colour_name.lower()}_"
            f"{sequence_number:02d}_"
            f"{dt.strftime('%Y%m%dT%H%M%SZ')}"
            ".png"
        )

        map_file = (
            output_dir
            / filename
        )

        # ---------------------------------------------------------------------
        # Standard OpenSky renderer
        # ---------------------------------------------------------------------

        fig, ax = render_map(

            aircraft=
                snapshot,

            area=
                area,

            bbox=
                bbox,

            aircraft_states=
                aircraft_states,

            warnings=
                warnings,

            conflicts=
                conflicts,

            snapshot_time=
                dt.to_pydatetime(),

            map_file=
                map_file,

            config=
                config,

            waypoints=
                waypoints,

            corridors=
                corridors,

            show=
                False,

            save=
                False,
        )

        # ---------------------------------------------------------------------
        # Validation overlay outside map
        # ---------------------------------------------------------------------

        pair_found_on_map = (
            draw_validation_overlay(

                fig=
                    fig,

                ax=
                    ax,

                snapshot=
                    snapshot,

                selected_row=
                    selected_row,

                pair_result=
                    pair_result,

                colour=
                    safety_colours[
                        validation_state
                    ],
            )
        )

        # ---------------------------------------------------------------------
        # EXACT OUTPUT RESOLUTION
        #
        # IMPORTANT:
        #
        # DO NOT use:
        #
        #     bbox_inches="tight"
        #
        # because Matplotlib would crop the canvas and the output
        # dimensions would no longer necessarily match bbox_presets.json.
        # ---------------------------------------------------------------------

        fig.set_size_inches(
            render_width_px
            / render_dpi,

            render_height_px
            / render_dpi,

            forward=True,
        )

        fig.savefig(
            map_file,

            dpi=
                render_dpi,

            facecolor=
                "white",
        )

        plt.close(
            fig
        )

        generated += 1

        # ---------------------------------------------------------------------
        # Validation report
        # ---------------------------------------------------------------------

        recomputed_state = None

        recomputed_h = np.nan

        recomputed_v = np.nan

        recomputed_cpa = np.nan

        recomputed_tcpa = np.nan

        if pair_result is not None:

            recomputed_state = (
                pair_result.get(
                    "state_current"
                )
            )

            recomputed_h = (
                pair_result.get(
                    "horizontal_sep_nm"
                )
            )

            recomputed_v = (
                pair_result.get(
                    "vertical_sep_ft"
                )
            )

            recomputed_cpa = (
                pair_result.get(
                    "cpa_nm"
                )
            )

            recomputed_tcpa = (
                pair_result.get(
                    "tcpa_s"
                )
            )

        state_matches = (
            recomputed_state
            ==
            validation_state
        )

        report_rows.append(
            {

                "validation_colour":
                    colour_name,

                "timestamp":
                    timestamp,

                "timestamp_utc":
                    dt.isoformat(),

                "icao24_a":
                    selected_row[
                        "icao24_a"
                    ],

                "callsign_a":
                    selected_row.get(
                        "callsign_a",
                        "",
                    ),

                "icao24_b":
                    selected_row[
                        "icao24_b"
                    ],

                "callsign_b":
                    selected_row.get(
                        "callsign_b",
                        "",
                    ),

                "source_event_file":
                    selected_row[
                        "source_event_file"
                    ],

                "event_state_worst_5min":
                    selected_row[
                        "event_state"
                    ],

                "detector_state_current":
                    selected_row[
                        "state_current"
                    ],

                "recomputed_state_current":
                    recomputed_state,

                "state_matches":
                    state_matches,

                "detector_horizontal_nm":
                    selected_row[
                        "horizontal_sep_nm"
                    ],

                "recomputed_horizontal_nm":
                    recomputed_h,

                "detector_vertical_ft":
                    selected_row[
                        "vertical_sep_ft"
                    ],

                "recomputed_vertical_ft":
                    recomputed_v,

                "detector_cpa_nm":
                    selected_row.get(
                        "cpa_nm",
                        np.nan,
                    ),

                "recomputed_cpa_nm":
                    recomputed_cpa,

                "detector_tcpa_s":
                    selected_row.get(
                        "tcpa_s",
                        np.nan,
                    ),

                "recomputed_tcpa_s":
                    recomputed_tcpa,

                "worst_status_0_1min":
                    selected_row.get(
                        "worst_status_0_1min",
                        "",
                    ),

                "worst_status_1_2min":
                    selected_row.get(
                        "worst_status_1_2min",
                        "",
                    ),

                "worst_status_2_5min":
                    selected_row.get(
                        "worst_status_2_5min",
                        "",
                    ),

                "pair_found_on_map":
                    pair_found_on_map,

                "image_width_px":
                    render_width_px,

                "image_height_px":
                    render_height_px,

                "screenshot":
                    filename,
            }
        )

        print(
            f"Saved: "
            f"{map_file.name}"
        )

        print(
            f"Resolution               : "
            f"{render_width_px} x "
            f"{render_height_px}"
        )

        print(
            f"Detector current state   : "
            f"{validation_state}"
        )

        print(
            f"Recomputed current state : "
            f"{recomputed_state}"
        )

        print(
            f"State match              : "
            f"{state_matches}"
        )

    # =========================================================================
    # SAVE REPORT
    # =========================================================================

    report = pd.DataFrame(
        report_rows
    )

    report_file = (
        output_dir
        / "validation_report.csv"
    )

    report.to_csv(
        report_file,
        index=False,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "VALIDATION SCREENSHOTS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Generated: "
        f"{generated}/{total_requested}"
    )

    print()

    print(
        "Output resolution:"
    )

    print(
        f"{render_width_px} x "
        f"{render_height_px} px"
    )

    print()

    print(
        "Output folder:"
    )

    print(
        output_dir
    )

    print()

    print(
        "Random selection:"
    )

    print(
        selected_csv
    )

    print()

    print(
        "Validation report:"
    )

    print(
        report_file
    )

    if not report.empty:

        matches = int(
            report[
                "state_matches"
            ]
            .fillna(False)
            .sum()
        )

        mismatches = (
            len(report)
            - matches
        )

        print()

        print(
            "Detector/recalculation "
            f"state matches: "
            f"{matches}/{len(report)}"
        )

        print(
            "Detector/recalculation "
            f"state mismatches: "
            f"{mismatches}/{len(report)}"
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
