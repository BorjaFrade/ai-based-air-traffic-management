#!/usr/bin/env python3

"""
Download historical OpenSky state vectors through Trino.

The script:

1. Queries OpenSky Trino in one-hour chunks.
2. Restricts the query to a geographic bounding box.
3. Stores ALL returned state vectors in a CSV file.
4. Optionally generates one traffic map every N seconds.
5. Optionally creates an MP4 video from those PNG frames.

The database is NOT queried every 10 seconds.
The 10-second snapshots are generated locally from the downloaded data.

Master's Thesis:
AI-Based Air Traffic Management

Author:
Borja Rodríguez Frade
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyopensky.schema import StateVectorsData4
from pyopensky.trino import Trino
from pyproj import Geod

from opensky_plot import render_map

# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

load_dotenv(
    PROJECT_ROOT / "credentials.env"
)


def print_authentication_mode():
    """Report whether Trino credentials are available from .env/environment."""

    username = os.getenv("OPENSKY_USERNAME")
    password = os.getenv("OPENSKY_PASSWORD")

    if username and password:
        print("OpenSky authentication: credentials loaded from environment/.env")
    else:
        print(
            "OpenSky authentication: no username/password in environment; "
            "pyopensky may fall back to browser authentication"
        )


# ============================================================
# USER CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# 1. TIME INTERVAL
# ------------------------------------------------------------
#
# All times are UTC.
#

START_UTC = "2025-06-16 20:00:00"

END_UTC = "2025-06-16 20:05:00"


# ------------------------------------------------------------
# 2. REGION
# ------------------------------------------------------------

BBOX_PRESETS_FILE = Path(__file__).with_name(
    "bbox_presets.json"
)

GEOD = Geod(ellps="WGS84")


def load_bbox_presets():

    with BBOX_PRESETS_FILE.open(
        encoding="utf-8"
    ) as presets_file:
        presets = json.load(presets_file)

    if not isinstance(presets, dict):
        raise ValueError(
            "BBOX presets must be a JSON object"
        )

    return presets


def bbox_from_center(
    latitude,
    longitude,
    radius_km,
):

    radius_m = radius_km * 1000.0

    _, latitude_max, _ = GEOD.fwd(
        longitude,
        latitude,
        0,
        radius_m,
    )

    _, latitude_min, _ = GEOD.fwd(
        longitude,
        latitude,
        180,
        radius_m,
    )

    longitude_max, _, _ = GEOD.fwd(
        longitude,
        latitude,
        90,
        radius_m,
    )

    longitude_min, _, _ = GEOD.fwd(
        longitude,
        latitude,
        270,
        radius_m,
    )

    return {
        "lamin": latitude_min,
        "lamax": latitude_max,
        "lomin": longitude_min,
        "lomax": longitude_max,
    }


def bbox_from_preset(preset):

    if preset["type"] == "vertices":
        return dict(preset["vertices"])

    if preset["type"] == "center":
        center = preset["center"]
        return bbox_from_center(
            float(center["lat"]),
            float(center["lon"]),
            float(preset["radius_km"]),
        )

    raise ValueError(
        'BBOX preset type must be "vertices" or "center"'
    )


BBOX_PRESETS = load_bbox_presets()

REGIONS = {
    preset_key: bbox_from_preset(preset)
    for preset_key, preset in BBOX_PRESETS.items()
}


SELECTED_REGION = "GALICIA"


# ------------------------------------------------------------
# MAP / VIDEO RENDER SIZE
# ------------------------------------------------------------
#
# Pixel dimensions are defined per region in bbox_presets.json.
#
# Examples:
#
#   GALICIA / BARCELONA_50KM -> 3840 x 3840
#   NORTHERN_SPAIN           -> 3840 x 2160
#
# The values must be even so they are directly compatible with
# H.264 / yuv420p video encoding.
#

DEFAULT_RENDER_WIDTH_PX = 3840
DEFAULT_RENDER_HEIGHT_PX = 2160

FIG_DPI = 160


def resolve_render_configuration():

    preset = BBOX_PRESETS[
        SELECTED_REGION
    ]

    render = preset.get(
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


    if (
        width_px <= 0
        or
        height_px <= 0
    ):

        raise ValueError(
            "Render width/height must be positive."
        )


    if (
        width_px % 2 != 0
        or
        height_px % 2 != 0
    ):

        raise ValueError(
            "Render width/height must be even for "
            "H.264/yuv420p video encoding."
        )


    figure_size = (

        width_px / FIG_DPI,

        height_px / FIG_DPI,

    )


    return (
        width_px,
        height_px,
        figure_size,
    )


(
    FRAME_WIDTH_PX,
    FRAME_HEIGHT_PX,
    FIG_SIZE,

) = resolve_render_configuration()


# ------------------------------------------------------------
# 3. DATABASE FILTERING
# ------------------------------------------------------------

KEEP_ONLY_AIRBORNE = False

MAX_LAST_CONTACT_AGE_SECONDS = 15


# ------------------------------------------------------------
# 4. SNAPSHOT / VIDEO
# ------------------------------------------------------------

#
# Generate PNG for every temporal snapshot.
#

GENERATE_FRAMES = True


#
# Generate MP4 from the PNG files.
#
# VIDEO automatically implies GENERATE_FRAMES.
#

GENERATE_VIDEO = True


#
# State displayed every N seconds.
#

FRAME_STEP_SECONDS = 10


#
# Video playback speed.
#
# With:
#
#   FRAME_STEP_SECONDS = 10
#   VIDEO_FPS = 10
#
# Current 5-minute test:
#
#   5 * 60 / 10 = 30 frames
#
# At 10 fps the video duration is:
#
#   30 / 10 = 3 s
#

VIDEO_FPS = 10


# ------------------------------------------------------------
# 5. OUTPUT
# ------------------------------------------------------------

RAW_DATA_DIR = (

    PROJECT_ROOT
    / "data"
    / "raw"
    / "opensky_trino"

)


RESULTS_DIR = (

    PROJECT_ROOT
    / "results"
    / "historical"

)


# Reuse the CSV already downloaded for the same time range/region.
#
# True  -> if the expected CSV exists, do not query Trino again.
#          This is useful while debugging PNG/video generation.
# False -> query Trino and create a new CSV.
#
# If True and the CSV does not exist, the script automatically
# falls back to a Trino query.
USE_EXISTING_CSV = True


# Print timestamp diagnostics after preparing the historical data.
DEBUG_TIME_SAMPLING = True


OVERWRITE_EXISTING = False


# ============================================================
# MAP CONFIGURATION
# ============================================================
#
# Same visual configuration used by opensky_snapshot.py.
#

SHOW_GROUND_AIRCRAFT = False

SHOW_LABELS = True

SHOW_CITIES = True

SHOW_ALTITUDE = True

SHOW_GROUND_SPEED = True

SHOW_VERTICAL_SPEED = True


SHOW_TRAJECTORY_VECTOR = True

TRAJECTORY_MINUTES = 2


# ------------------------------------------------------------
# Separation
# ------------------------------------------------------------

SEPARATION_PROFILE = "ENROUTE"

MIN_HORIZONTAL_SEPARATION_NM = 5.0

MIN_VERTICAL_SEPARATION_FT = 1000.0


WARNING_HORIZONTAL_FACTOR = 2.0

WARNING_VERTICAL_FACTOR = 2.0


WARNING_HORIZONTAL_NM = (

    MIN_HORIZONTAL_SEPARATION_NM
    * WARNING_HORIZONTAL_FACTOR

)


WARNING_VERTICAL_FT = (

    MIN_VERTICAL_SEPARATION_FT
    * WARNING_VERTICAL_FACTOR

)


# ------------------------------------------------------------
# Safety area
# ------------------------------------------------------------

SHOW_SAFETY_AREA = True

CHECK_GROUND_AIRCRAFT_SAFETY = False

SAFETY_AREA_POINTS = 120


STATE_NORMAL = "NORMAL"

STATE_WARNING = "WARNING"

STATE_CONFLICT = "CONFLICT"


NORMAL_AIRCRAFT_COLOR = "#00C853"

WARNING_AIRCRAFT_COLOR = "#FFD600"

CONFLICT_AIRCRAFT_COLOR = "#FF2020"


SAFETY_AREA_NORMAL_COLOR = (
    NORMAL_AIRCRAFT_COLOR
)

SAFETY_AREA_WARNING_COLOR = (
    WARNING_AIRCRAFT_COLOR
)

SAFETY_AREA_CONFLICT_COLOR = (
    CONFLICT_AIRCRAFT_COLOR
)


SAFETY_AREA_LINEWIDTH = 1.1

SAFETY_AREA_ALPHA = 0.55

SHOW_SAFETY_AREA_FILL = True

SAFETY_AREA_FILL_ALPHA = 0.035


AIRCRAFT_SIZE = 180

MAX_CITIES = 10


VERTICAL_SPEED_LEVEL_THRESHOLD_FPM = 100

VERTICAL_SPEED_CLIMB_COLOR = "#00D26A"

VERTICAL_SPEED_DESCENT_COLOR = "#FF4040"

VERTICAL_SPEED_LEVEL_COLOR = "#D8D8D8"

GROUND_COLOR = "#FFD400"




# ============================================================
# TRINO COLUMNS
# ============================================================

TRINO_COLUMNS = (

    "time",

    "icao24",

    "lat",
    "lon",

    "velocity",
    "heading",
    "vertrate",

    "callsign",

    "onground",

    "alert",
    "spi",

    "squawk",

    "baroaltitude",
    "geoaltitude",

    "lastposupdate",
    "lastcontact",

    "hour",
)


# ============================================================
# DATETIME
# ============================================================

def parse_datetime_utc(
    value: str,
) -> pd.Timestamp:

    timestamp = pd.Timestamp(
        value
    )


    if timestamp.tzinfo is None:

        timestamp = (
            timestamp
            .tz_localize("UTC")
        )

    else:

        timestamp = (
            timestamp
            .tz_convert("UTC")
        )


    return timestamp


# ============================================================
# OUTPUT PATHS
# ============================================================

def build_run_name(
    start,
    end,
):

    return (

        f"{SELECTED_REGION.lower()}_"

        f"{start.strftime('%Y%m%dT%H%M%S')}_"

        f"{end.strftime('%Y%m%dT%H%M%S')}"

    )


def build_output_paths(
    start,
    end,
):

    run_name = build_run_name(
        start,
        end,
    )


    raw_directory = (

        RAW_DATA_DIR
        / SELECTED_REGION
        / f"year={start.year}"
        / f"date={start.date()}"

    )


    results_directory = (

        RESULTS_DIR
        / run_name

    )


    frames_directory = (

        results_directory
        / "frames"

    )


    raw_directory.mkdir(

        parents=True,
        exist_ok=True,

    )


    results_directory.mkdir(

        parents=True,
        exist_ok=True,

    )


    csv_file = (

        raw_directory
        / f"{run_name}.csv"

    )


    video_file = (

        results_directory
        / f"{run_name}.mp4"

    )


    return {

        "csv":
            csv_file,

        "results":
            results_directory,

        "frames":
            frames_directory,

        "video":
            video_file,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration(
    start,
    end,
):

    if end <= start:

        raise ValueError(
            "END_UTC must be later than START_UTC."
        )


    if SELECTED_REGION not in REGIONS:

        raise ValueError(

            f"Unknown region: "
            f"{SELECTED_REGION}"

        )


    if FRAME_STEP_SECONDS <= 0:

        raise ValueError(

            "FRAME_STEP_SECONDS "
            "must be positive."

        )


    if VIDEO_FPS <= 0:

        raise ValueError(
            "VIDEO_FPS must be positive."
        )


# ============================================================
# TRINO
# ============================================================

def query_hour(
    trino,
    start,
    end,
):

    """
    Query one short temporal block.

    pyopensky applies the corresponding OpenSky
    partition filtering internally.
    """

    region = REGIONS[
        SELECTED_REGION
    ]


    bounds = (

        region["lomin"],
        region["lamin"],

        region["lomax"],
        region["lamax"],
    )


    filters = [

        (
            StateVectorsData4.time
            - StateVectorsData4.lastcontact
        )
        <= MAX_LAST_CONTACT_AGE_SECONDS

    ]


    if KEEP_ONLY_AIRBORNE:

        filters.append(

            StateVectorsData4.onground
            == False

        )


    result = trino.history(

        start,

        end,

        *filters,

        bounds=bounds,

        selected_columns=
            TRINO_COLUMNS,

        cached=True,

        date_delta=
            pd.Timedelta(hours=1),

    )


    if result is None:

        return pd.DataFrame(
            columns=TRINO_COLUMNS
        )


    return result.copy()


# ============================================================
# DOWNLOAD
# ============================================================

def download_history(
    trino,
    start,
    end,
    csv_file,
    keep_in_memory,
):

    """
    Query Trino sequentially by hour.

    Every returned row is appended directly to the CSV.

    If frames/video are required, the chunks are also retained
    in memory for later processing.
    """

    if csv_file.exists():

        if not OVERWRITE_EXISTING:

            raise FileExistsError(

                f"Output already exists:\n"
                f"{csv_file}\n\n"

                "Set OVERWRITE_EXISTING = True "
                "to replace it."

            )


        csv_file.unlink()


    chunks = []


    current = start

    first_write = True


    while current < end:

        chunk_end = min(

            current
            + pd.Timedelta(hours=1),

            end,

        )


        print()
        print("=" * 72)

        print(
            "Query:"
        )

        print(
            f"  {current}"
        )

        print(
            f"  -> {chunk_end}"
        )


        dataframe = query_hour(

            trino,

            current,
            chunk_end,

        )


        print(
            f"Rows returned: "
            f"{len(dataframe):,}"
        )


        if not dataframe.empty:

            # ------------------------------------------------
            # Store EXACT Trino result
            # ------------------------------------------------

            dataframe.to_csv(

                csv_file,

                mode=(
                    "w"
                    if first_write
                    else "a"
                ),

                header=
                    first_write,

                index=False,

            )


            first_write = False


            if keep_in_memory:

                chunks.append(
                    dataframe
                )


        current = chunk_end


    if first_write:

        # No rows at all.
        pd.DataFrame(
            columns=TRINO_COLUMNS
        ).to_csv(

            csv_file,

            index=False,

        )


    print()
    print(
        "Raw Trino CSV:"
    )

    print(
        csv_file
    )


    if not keep_in_memory:

        return None


    if not chunks:

        return pd.DataFrame(
            columns=TRINO_COLUMNS
        )


    result = pd.concat(

        chunks,

        ignore_index=True,

    )


    # Defensive duplicate removal at hour boundaries.
    result.drop_duplicates(
        inplace=True
    )


    return result


# ============================================================
# TIMESTAMP NORMALISATION
# ============================================================

def to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Convert OpenSky/pyopensky timestamps to Unix seconds robustly.

    Supports:
    - pandas datetime columns
    - ISO-8601 strings loaded from CSV
    - Unix timestamps expressed in seconds, milliseconds,
      microseconds or nanoseconds

    The conversion from datetime values deliberately uses a timedelta
    from the Unix epoch instead of ``astype("int64")``.  Recent pandas
    versions may store datetimes internally with microsecond resolution,
    so assuming nanoseconds can introduce a factor-of-1000 error.
    """

    # --------------------------------------------------------
    # Numeric Unix timestamps
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).astype("float64")

        finite = numeric[np.isfinite(numeric)]

        if finite.empty:
            return numeric

        magnitude = float(
            finite.abs().median()
        )

        # OpenSky Unix seconds around 2025 are ~1.7e9.
        # Normalise other common timestamp units defensively.
        if magnitude > 1e17:      # nanoseconds
            numeric = numeric / 1_000_000_000.0

        elif magnitude > 1e14:    # microseconds
            numeric = numeric / 1_000_000.0

        elif magnitude > 1e11:    # milliseconds
            numeric = numeric / 1_000.0

        return numeric


    # --------------------------------------------------------
    # Object/string/datetime values
    # --------------------------------------------------------

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

    # Unit-independent conversion.  Do not use astype("int64")
    # here because pandas may use ns or us datetime resolution.
    return (
        values
        .sub(epoch)
        .dt.total_seconds()
        .astype("float64")
    )

# ============================================================
# NORMALISE FOR EXISTING PLOTTER
# ============================================================

def prepare_history(
    dataframe,
    start,
):

    """
    Convert Trino naming to the column names already used by
    opensky_snapshot.py / opensky_plot.py.
    """

    if dataframe.empty:

        return dataframe.copy()


    df = dataframe.copy()


    rename = {

        "lat":
            "latitude",

        "lon":
            "longitude",

        "heading":
            "true_track",

        "vertrate":
            "vertical_rate",

        "onground":
            "on_ground",

        "baroaltitude":
            "baro_altitude",

        "geoaltitude":
            "geo_altitude",

        "lastposupdate":
            "time_position",

        "lastcontact":
            "last_contact",
    }


    df.rename(

        columns=rename,

        inplace=True,

    )


    # --------------------------------------------------------
    # Clean callsigns
    # --------------------------------------------------------

    df["callsign"] = (

        df["callsign"]

        .fillna("")

        .astype(str)

        .str.strip()

    )


    # --------------------------------------------------------
    # Numeric values
    # --------------------------------------------------------

    timestamp_columns = [
        "time",
        "time_position",
        "last_contact",
    ]

    for column in timestamp_columns:
        df[column] = to_epoch_seconds(
            df[column]
        )


    numeric_columns = [

        "latitude",
        "longitude",

        "velocity",
        "true_track",
        "vertical_rate",

        "baro_altitude",
        "geo_altitude",
    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce",

        )


    # --------------------------------------------------------
    # Same derived units as opensky_snapshot.py
    # --------------------------------------------------------

    df[
        "baro_altitude_ft"
    ] = (

        df["baro_altitude"]
        * 3.28084

    )


    df[
        "geo_altitude_ft"
    ] = (

        df["geo_altitude"]
        * 3.28084

    )


    df[
        "velocity_kts"
    ] = (

        df["velocity"]
        * 1.94384

    )


    df[
        "vertical_rate_fpm"
    ] = (

        df["vertical_rate"]
        * 196.8504

    )


    # --------------------------------------------------------
    # 10-second time bucket
    # --------------------------------------------------------

    start_epoch = int(
        start.timestamp()
    )


    df = df.dropna(
        subset=[
            "time",
            "icao24",
        ]
    ).copy()


    df["sample_time"] = (

        start_epoch

        +

        (
            (
                df["time"]
                - start_epoch
            )

            // FRAME_STEP_SECONDS

        )

        * FRAME_STEP_SECONDS

    )


    df["sample_time"] = (

        df["sample_time"]
        .astype("Int64")

    )


    # --------------------------------------------------------
    # Keep latest state per aircraft in each temporal bucket.
    # --------------------------------------------------------

    df = df.sort_values(

        [

            "sample_time",

            "icao24",

            "time",

        ]

    )


    df = (

        df

        .groupby(

            [
                "sample_time",
                "icao24",
            ],

            as_index=False,

        )

        .tail(1)

        .copy()

    )


    return df


# ============================================================
# SEPARATION ANALYSIS
# ============================================================

STATE_PRIORITY = {

    STATE_NORMAL:
        0,

    STATE_WARNING:
        1,

    STATE_CONFLICT:
        2,
}


def callsign_of(
    row,
):

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
        row["icao24"]
    ).upper()


def aircraft_on_ground(
    row,
):

    value = row.get(
        "on_ground",
        False,
    )


    if pd.isna(value):

        return False


    return bool(value)


def classify_pair(
    horizontal_nm,
    vertical_ft,
):

    if (

        horizontal_nm
        < MIN_HORIZONTAL_SEPARATION_NM

        and

        vertical_ft
        < MIN_VERTICAL_SEPARATION_FT

    ):

        return STATE_CONFLICT


    if (

        horizontal_nm
        < WARNING_HORIZONTAL_NM

        and

        vertical_ft
        < WARNING_VERTICAL_FT

    ):

        return STATE_WARNING


    return STATE_NORMAL


def analyse_snapshot(
    aircraft,
):

    """
    Same basic 3D separation classification used by the
    current snapshot visualisation.
    """

    from pyproj import Geod

    geod = Geod(
        ellps="WGS84"
    )


    states = {

        index:
            STATE_NORMAL

        for index
        in aircraft.index

    }


    pairs = []


    rows = list(
        aircraft.iterrows()
    )


    for i in range(
        len(rows)
    ):

        index_a, aircraft_a = (
            rows[i]
        )


        if (

            aircraft_on_ground(
                aircraft_a
            )

            and

            not CHECK_GROUND_AIRCRAFT_SAFETY

        ):

            continue


        for j in range(
            i + 1,
            len(rows)
        ):

            index_b, aircraft_b = (
                rows[j]
            )


            if (

                aircraft_on_ground(
                    aircraft_b
                )

                and

                not CHECK_GROUND_AIRCRAFT_SAFETY

            ):

                continue


            _, _, distance_m = (
                geod.inv(

                    float(
                        aircraft_a[
                            "longitude"
                        ]
                    ),

                    float(
                        aircraft_a[
                            "latitude"
                        ]
                    ),

                    float(
                        aircraft_b[
                            "longitude"
                        ]
                    ),

                    float(
                        aircraft_b[
                            "latitude"
                        ]
                    ),

                )
            )


            horizontal_nm = (

                distance_m
                / 1852.0

            )


            altitude_a = (
                aircraft_a.get(
                    "baro_altitude_ft",
                    np.nan,
                )
            )


            altitude_b = (
                aircraft_b.get(
                    "baro_altitude_ft",
                    np.nan,
                )
            )


            if (

                pd.isna(altitude_a)

                or

                pd.isna(altitude_b)

            ):

                if (

                    horizontal_nm
                    < WARNING_HORIZONTAL_NM

                ):

                    state = (
                        STATE_WARNING
                    )

                    vertical_ft = None

                else:

                    continue


            else:

                vertical_ft = abs(

                    float(altitude_a)

                    - float(altitude_b)

                )


                state = classify_pair(

                    horizontal_nm,

                    vertical_ft,

                )


                if state == STATE_NORMAL:

                    continue


            for index in (
                index_a,
                index_b,
            ):

                if (

                    STATE_PRIORITY[state]

                    >

                    STATE_PRIORITY[
                        states[index]
                    ]

                ):

                    states[index] = state


            pairs.append({

                "index_a":
                    index_a,

                "index_b":
                    index_b,

                "callsign_a":
                    callsign_of(
                        aircraft_a
                    ),

                "callsign_b":
                    callsign_of(
                        aircraft_b
                    ),

                "horizontal_nm":
                    horizontal_nm,

                "vertical_ft":
                    vertical_ft,

                "state":
                    state,
            })


    warnings = [

        pair

        for pair in pairs

        if (
            pair["state"]
            == STATE_WARNING
        )

    ]


    conflicts = [

        pair

        for pair in pairs

        if (
            pair["state"]
            == STATE_CONFLICT
        )

    ]


    return (
        states,
        warnings,
        conflicts,
    )


# ============================================================
# CREATE PNG FRAMES
# ============================================================

def generate_frames(
    dataframe,
    start,
    end,
    frames_directory,
):

    frames_directory.mkdir(

        parents=True,
        exist_ok=True,

    )


    region = REGIONS[
        SELECTED_REGION
    ]


    bbox = {

        **region,

        "extended":
            1,

    }


    area = {

        "name":
            BBOX_PRESETS[
                SELECTED_REGION
            ]["name"],

        "icao":
            "",

        "center_lat":
            (
                region["lamin"]
                + region["lamax"]
            ) / 2.0,

        "center_lon":
            (
                region["lomin"]
                + region["lomax"]
            ) / 2.0,
    }


    start_epoch = int(
        start.timestamp()
    )


    end_epoch = int(
        end.timestamp()
    )


    frame_paths = []


    frame_number = 0


    for timestamp in range(

        start_epoch,

        end_epoch,

        FRAME_STEP_SECONDS,

    ):

        snapshot = dataframe.loc[

            dataframe[
                "sample_time"
            ]
            == timestamp

        ].copy()


        # ----------------------------------------------------
        # Valid position
        # ----------------------------------------------------

        valid_position = (

            snapshot[
                "latitude"
            ].notna()

            &

            snapshot[
                "longitude"
            ].notna()

            &

            np.isfinite(
                snapshot[
                    "latitude"
                ]
            )

            &

            np.isfinite(
                snapshot[
                    "longitude"
                ]
            )

        )


        snapshot = (

            snapshot.loc[
                valid_position
            ]

            .copy()

        )


        if not SHOW_GROUND_AIRCRAFT:

            snapshot = (

                snapshot.loc[

                    snapshot[
                        "on_ground"
                    ]
                    != True

                ]

                .copy()

            )


        (
            aircraft_states,
            warnings,
            conflicts,

        ) = analyse_snapshot(
            snapshot
        )


        snapshot_time = (
            datetime.fromtimestamp(

                timestamp,

                tz=
                    start.tzinfo,

            )
        )


        frame_file = (

            frames_directory

            / (

                f"frame_"
                f"{frame_number:06d}_"

                f"{snapshot_time.strftime('%Y%m%dT%H%M%SZ')}"

                f".png"

            )

        )


        print(
            f"Frame "
            f"{frame_number:06d} | "
            f"{snapshot_time} | "
            f"{len(snapshot)} aircraft(s)"
        )


        fig, _ = render_map(

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
                snapshot_time,

            map_file=
                frame_file,

            config=
                globals(),

            show=False,

        )


        # ----------------------------------------------------
        # EXACT FRAME SIZE
        # ----------------------------------------------------
        #
        # opensky_plot.render_map() currently saves using
        # bbox_inches="tight".  That is useful for standalone
        # figures but produces variable / sometimes odd pixel
        # dimensions, which is not desirable for a video.
        #
        # Re-save the already rendered figure without tight
        # cropping so every PNG has exactly the dimensions
        # configured in bbox_presets.json.
        # ----------------------------------------------------

        fig.set_size_inches(

            FIG_SIZE[0],
            FIG_SIZE[1],

            forward=True,
        )


        fig.savefig(

            frame_file,

            dpi=FIG_DPI,

            facecolor="white",

        )


        plt.close(
            fig
        )


        frame_paths.append(
            frame_file
        )


        frame_number += 1


    return frame_paths


# ============================================================
# VIDEO
# ============================================================

def create_video(
    frame_paths,
    video_file,
):

    if not frame_paths:

        raise RuntimeError(
            "No PNG frames available."
        )


    print()
    print("=" * 72)

    print(
        "CREATING VIDEO"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Determine one common, even frame size.
    #
    # H.264 / yuv420p requires an even width and height.
    # bbox_inches="tight" may produce odd PNG dimensions
    # (for example 1463 x 1509), so we pad the images locally
    # instead of resizing/distorting the maps.
    # --------------------------------------------------------

    frame_shapes = []

    for frame_file in frame_paths:

        image = imageio.imread(
            frame_file
        )

        frame_shapes.append(
            image.shape
        )


    target_height = max(
        shape[0]
        for shape in frame_shapes
    )

    target_width = max(
        shape[1]
        for shape in frame_shapes
    )


    if target_height % 2:
        target_height += 1

    if target_width % 2:
        target_width += 1


    print(
        f"Video frame size: "
        f"{target_width}x{target_height}"
    )


    try:

        writer = imageio.get_writer(

            video_file,

            fps=VIDEO_FPS,

            codec="libx264",

            quality=8,

            macro_block_size=None,

            format="FFMPEG",

            pixelformat="yuv420p",
        )

    except (ValueError, ImportError) as exc:

        raise RuntimeError(
            "MP4 generation requires the ImageIO FFMPEG backend.\n"
            "Install it with:\n"
            "  /usr/local/bin/python3 -m pip install imageio-ffmpeg\n"
            "or:\n"
            "  /usr/local/bin/python3 -m pip install \"imageio[ffmpeg]\""
        ) from exc


    with writer:

        for index, frame_file in enumerate(
            frame_paths
        ):

            image = imageio.imread(
                frame_file
            )


            height = image.shape[0]
            width = image.shape[1]


            # ------------------------------------------------
            # Pad to a common even size without distorting
            # the rendered map.
            # ------------------------------------------------

            if (
                height != target_height
                or
                width != target_width
            ):

                if image.ndim == 2:

                    canvas = np.full(

                        (
                            target_height,
                            target_width,
                        ),

                        255,

                        dtype=image.dtype,
                    )

                else:

                    channels = image.shape[2]

                    canvas = np.full(

                        (
                            target_height,
                            target_width,
                            channels,
                        ),

                        255,

                        dtype=image.dtype,
                    )


                top = (
                    target_height
                    - height
                ) // 2

                left = (
                    target_width
                    - width
                ) // 2


                canvas[
                    top:top + height,
                    left:left + width,
                    ...
                ] = image


                image = canvas


            writer.append_data(
                image
            )


            if (
                index % 100
                == 0
            ):

                print(

                    f"Video frames: "
                    f"{index + 1}/"
                    f"{len(frame_paths)}"

                )


    print()
    print(
        "Video saved to:"
    )

    print(
        video_file
    )

# ============================================================
# EXISTING CSV / DEBUG HELPERS
# ============================================================

def load_existing_csv(csv_file: Path) -> pd.DataFrame:
    """Load a previously downloaded raw Trino CSV."""

    print()
    print("Using existing raw Trino CSV:")
    print(csv_file)

    dataframe = pd.read_csv(
        csv_file,
        low_memory=False,
    )

    print(
        f"Rows loaded from CSV: "
        f"{len(dataframe):,}"
    )

    return dataframe


def get_raw_data(
    start,
    end,
    csv_file,
    keep_in_memory,
):
    """Reuse an existing CSV when requested; otherwise query Trino."""

    if USE_EXISTING_CSV and csv_file.exists():
        if keep_in_memory:
            return load_existing_csv(csv_file)

        print()
        print("Existing CSV found; no Trino query required:")
        print(csv_file)
        return None

    if USE_EXISTING_CSV and not csv_file.exists():
        print()
        print("USE_EXISTING_CSV=True, but the expected CSV does not exist.")
        print("Falling back to a Trino query.")

    print_authentication_mode()

    trino = Trino()

    return download_history(
        trino=trino,
        start=start,
        end=end,
        csv_file=csv_file,
        keep_in_memory=keep_in_memory,
    )


def print_time_debug(
    historical_data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """Print enough information to diagnose empty temporal frames."""

    if not DEBUG_TIME_SAMPLING:
        return

    print()
    print("=" * 72)
    print("TIME SAMPLING DEBUG")
    print("=" * 72)

    if historical_data.empty:
        print("Historical dataframe is empty after preparation.")
        print("=" * 72)
        return

    columns = [
        column
        for column in (
            "time",
            "sample_time",
            "icao24",
            "latitude",
            "longitude",
        )
        if column in historical_data.columns
    ]

    print(historical_data[columns].head(10).to_string(index=False))

    minimum = historical_data["sample_time"].min()
    maximum = historical_data["sample_time"].max()

    print()
    print("sample_time min:", minimum)
    print("sample_time max:", maximum)
    print("expected start :", int(start.timestamp()))
    print("expected end   :", int(end.timestamp()))

    if pd.notna(minimum):
        print(
            "sample_time min UTC:",
            pd.to_datetime(
                int(minimum),
                unit="s",
                utc=True,
            ),
        )

    if pd.notna(maximum):
        print(
            "sample_time max UTC:",
            pd.to_datetime(
                int(maximum),
                unit="s",
                utc=True,
            ),
        )

    counts = (
        historical_data
        .groupby("sample_time")
        .size()
        .sort_index()
    )

    print()
    print("First temporal buckets:")
    print(counts.head(10).to_string())
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    start = parse_datetime_utc(
        START_UTC
    )


    end = parse_datetime_utc(
        END_UTC
    )


    validate_configuration(
        start,
        end,
    )


    paths = build_output_paths(

        start,
        end,

    )


    effective_generate_frames = (

        GENERATE_FRAMES
        or GENERATE_VIDEO

    )


    duration_seconds = (

        end - start

    ).total_seconds()


    expected_frames = int(

        np.ceil(

            duration_seconds

            / FRAME_STEP_SECONDS

        )

    )


    print()
    print("=" * 72)

    print(
        "OpenSky Historical Trino"
    )

    print("=" * 72)


    print(
        f"Region: "
        f"{SELECTED_REGION}"
    )


    print(
        f"Start: "
        f"{start}"
    )


    print(
        f"End:   "
        f"{end}"
    )


    print(
        f"Frame interval: "
        f"{FRAME_STEP_SECONDS} s"
    )


    print(
        f"Frame resolution: "
        f"{FRAME_WIDTH_PX} x "
        f"{FRAME_HEIGHT_PX} px"
    )


    if effective_generate_frames:

        print(
            f"Expected PNGs: "
            f"{expected_frames:,}"
        )


    if GENERATE_VIDEO:

        duration_video = (

            expected_frames
            / VIDEO_FPS

        )


        print(
            f"Approx. video duration: "
            f"{duration_video:.1f} s"
        )


    print("=" * 72)


    raw_data = get_raw_data(

        start=
            start,

        end=
            end,

        csv_file=
            paths["csv"],

        keep_in_memory=
            effective_generate_frames,

    )


    if not effective_generate_frames:

        return


    if raw_data is None or raw_data.empty:

        print(
            "No data available "
            "for frame generation."
        )

        return


    historical_data = (
        prepare_history(

            raw_data,

            start,

        )
    )


    print_time_debug(
        historical_data,
        start,
        end,
    )


    frame_paths = (
        generate_frames(

            historical_data,

            start,

            end,

            paths["frames"],

        )
    )


    if GENERATE_VIDEO:

        create_video(

            frame_paths,

            paths["video"],

        )


if __name__ == "__main__":

    main()
