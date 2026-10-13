from pathlib import Path
from datetime import datetime, timezone
import json

import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D, Bbox
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch

import contextily as cx
from pyproj import Geod
import geonamescache

from area_presets import (
    load_area_presets,
    resolve_area_preset,
    filter_dataframe_to_area,
    load_waypoints_for_area,
)
from opensky_safety import (
    STATE_NORMAL,
    STATE_PROXIMITY,
    STATE_HIGH_RISK,
    STATE_CONFLICT,
    analyse_aircraft_snapshot,
    relevant_pair_results,
)


# ============================================================
# ============================================================
# USER CONFIGURATION
# ============================================================
# ============================================================


# ============================================================
# 1. AREA
# ============================================================

# Available:
#
#   "CENTER_RADIUS"
#   "BBOX"

AREA_MODE = "BBOX"
#AREA_MODE = "CENTER_RADIUS"


# ------------------------------------------------------------
# CENTER + RADIUS
# ------------------------------------------------------------

SELECTED_PRESET = "BCN"

RADIUS_KM = 50


LOCATION_PRESETS_FILE = Path(__file__).with_name(
    "location_presets.json"
)


def load_location_presets():

    try:
        with LOCATION_PRESETS_FILE.open(
            encoding="utf-8"
        ) as presets_file:
            presets = json.load(presets_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Location presets file not found: "
            f"{LOCATION_PRESETS_FILE}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in location presets file: "
            f"{LOCATION_PRESETS_FILE}"
        ) from error

    if not isinstance(presets, dict):
        raise ValueError(
            "Location presets must be a JSON object"
        )

    return presets


LOCATION_PRESETS = load_location_presets()


# Used if SELECTED_PRESET = None

CUSTOM_CENTER_NAME = "Custom area"

CUSTOM_CENTER_LAT = 41.2974
CUSTOM_CENTER_LON = 2.0833


# ------------------------------------------------------------
# BBOX
# ------------------------------------------------------------

SELECTED_BBOX_PRESET = "GALICIA"


BBOX_PRESETS_FILE = Path(__file__).with_name(
    "bbox_presets.json"
)


BBOX_PRESETS = load_area_presets(BBOX_PRESETS_FILE)


# ============================================================
# 2. AIRCRAFT DISPLAY
# ============================================================

SHOW_GROUND_AIRCRAFT = False

SHOW_LABELS = True
SHOW_CITIES = True

# Aviation waypoints loaded from waypoints.json.
SHOW_WAYPOINTS = True
SHOW_WAYPOINT_LABELS = True

WAYPOINTS_FILE = Path(__file__).with_name(
    "waypoints.json"
)

# A square distinguishes waypoints from the white circular city markers.
WAYPOINT_MARKER = "s"
WAYPOINT_MARKER_SIZE = 34
WAYPOINT_FACE_COLOR = "white"
WAYPOINT_EDGE_COLOR = "black"
WAYPOINT_EDGE_WIDTH = 0.9
WAYPOINT_LABEL_COLOR = "white"

SHOW_ALTITUDE = True
SHOW_GROUND_SPEED = True
SHOW_VERTICAL_SPEED = True


# ============================================================
# 3. TRAJECTORY
# ============================================================

SHOW_TRAJECTORY_VECTOR = True

# Segmented look-ahead trajectory. Each segment is coloured with the worst
# predicted safety state inside that interval.
LOOKAHEAD_MINUTES = [1, 2, 5]

# Kinematic prediction resolution. Five seconds is the default compromise;
# it can be reduced to 2 s for sensitivity checks without changing the raw
# OpenSky sampling.
PREDICTION_STEP_SECONDS = 5


# ============================================================
# 4. SEPARATION PROFILE
# ============================================================

SEPARATION_PROFILE = "ENROUTE"


SEPARATION_PROFILES_FILE = Path(__file__).with_name(
    "separation_profiles.json"
)


def load_separation_profiles():

    try:
        with SEPARATION_PROFILES_FILE.open(
            encoding="utf-8"
        ) as profiles_file:
            profiles = json.load(profiles_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Separation profiles file not found: "
            f"{SEPARATION_PROFILES_FILE}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in separation profiles file: "
            f"{SEPARATION_PROFILES_FILE}"
        ) from error

    if not isinstance(profiles, dict):
        raise ValueError(
            "Separation profiles must be a JSON object"
        )

    return profiles


SEPARATION_PROFILES = load_separation_profiles()


if SEPARATION_PROFILE not in SEPARATION_PROFILES:

    raise ValueError(
        f"Unknown separation profile: "
        f"{SEPARATION_PROFILE}"
    )


ACTIVE_SEPARATION = (
    SEPARATION_PROFILES[
        SEPARATION_PROFILE
    ]
)


MIN_HORIZONTAL_SEPARATION_NM = float(
    ACTIVE_SEPARATION[
        "horizontal_nm"
    ]
)


MIN_VERTICAL_SEPARATION_FT = float(
    ACTIVE_SEPARATION[
        "vertical_ft"
    ]
)

# Initial TFM experiment: upper en-route RVSM band over the selected study
# area. Barometric altitude is used for the vertical-separation analysis.
FILTER_CRUISE_FL_BAND = True
CRUISE_FL_MIN = 290
CRUISE_FL_MAX = 410


# ============================================================
# 5. WARNING ENVELOPE
# ============================================================
#
# RED:
#   H < minimum horizontal AND V < minimum vertical.
#
# ORANGE:
#   inside the warning envelope and exactly one minimum is infringed.
#
# YELLOW:
#   inside the warning envelope while both minima are still respected.
#
# GREEN:
#   everything else.
#
# A pair such as H = 43 NM, V = 0 ft therefore remains NORMAL / GREEN.
#
# ============================================================

# Research warning envelope. This is NOT an ATC separation minimum.
# 1.50 -> 7.5 NM / 1500 ft for the ENROUTE 5 NM / 1000 ft profile.
# Sensitivity analysis can compare 1.25 and 1.50.
WARNING_FACTOR = 1.50

WARNING_HORIZONTAL_FACTOR = WARNING_FACTOR
WARNING_VERTICAL_FACTOR = WARNING_FACTOR


WARNING_HORIZONTAL_NM = (
    MIN_HORIZONTAL_SEPARATION_NM
    * WARNING_HORIZONTAL_FACTOR
)


WARNING_VERTICAL_FT = (
    MIN_VERTICAL_SEPARATION_FT
    * WARNING_VERTICAL_FACTOR
)


# ============================================================
# 6. SAFETY AREA
# ============================================================

SHOW_SAFETY_AREA = True


# Ground aircraft are normally excluded from airborne
# separation analysis.

CHECK_GROUND_AIRCRAFT_SAFETY = False


# Number of geodesic points used to draw the area boundary.

SAFETY_AREA_POINTS = 120


# ============================================================
# 7. SAFETY STATES
# ============================================================

# Four-state model:
# GREEN  = normal
# YELLOW = proximity inside the warning envelope
# ORANGE = high risk: one separation minimum already infringed
# RED    = both minima infringed simultaneously
STATE_WARNING = STATE_PROXIMITY  # backward-compatible alias


# ============================================================
# 8. SAFETY COLOURS
# ============================================================
#
# GREEN  = normal
# YELLOW = proximity warning
# ORANGE = high risk
# RED    = loss of separation
#
# ============================================================

NORMAL_AIRCRAFT_COLOR = "#00A651"
PROXIMITY_AIRCRAFT_COLOR = "#FFD400"
HIGH_RISK_AIRCRAFT_COLOR = "#FF6F00"
CONFLICT_AIRCRAFT_COLOR = "#D7191C"

# Backward-compatible plotting alias.
WARNING_AIRCRAFT_COLOR = PROXIMITY_AIRCRAFT_COLOR


SAFETY_AREA_NORMAL_COLOR = (
    NORMAL_AIRCRAFT_COLOR
)

SAFETY_AREA_PROXIMITY_COLOR = PROXIMITY_AIRCRAFT_COLOR
SAFETY_AREA_HIGH_RISK_COLOR = HIGH_RISK_AIRCRAFT_COLOR

SAFETY_AREA_WARNING_COLOR = (
    PROXIMITY_AIRCRAFT_COLOR
)

SAFETY_AREA_CONFLICT_COLOR = (
    CONFLICT_AIRCRAFT_COLOR
)


SAFETY_AREA_LINEWIDTH = 1.1

SAFETY_AREA_ALPHA = 0.55


SHOW_SAFETY_AREA_FILL = True

SAFETY_AREA_FILL_ALPHA = 0.035


# ============================================================
# 9. OTHER VISUAL OPTIONS
# ============================================================

AIRCRAFT_SIZE = 180

MAX_CITIES = 10


VERTICAL_SPEED_LEVEL_THRESHOLD_FPM = 100


VERTICAL_SPEED_CLIMB_COLOR = "#00D26A"

VERTICAL_SPEED_DESCENT_COLOR = "#FF4040"

VERTICAL_SPEED_LEVEL_COLOR = "#D8D8D8"


GROUND_COLOR = "#FFD400"


# ============================================================
# 10. OUTPUT
# ============================================================

FIG_DPI = 160


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "opensky"

RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


CSV_FILE = (
    RAW_DATA_DIR
    / "opensky_snapshot.csv"
)


MAP_FILE = (
    RESULTS_FIGURES_DIR
    / "opensky_snapshot_map.png"
)


# ============================================================
# END USER CONFIGURATION
# ============================================================
# ============================================================



# ============================================================
# OPENSKY
# ============================================================

OPENSKY_URL = (
    "https://opensky-network.org/api/states/all"
)


COLUMNS = [

    "icao24",
    "callsign",
    "origin_country",

    "time_position",
    "last_contact",

    "longitude",
    "latitude",

    "baro_altitude",

    "on_ground",

    "velocity",
    "true_track",
    "vertical_rate",

    "sensors",

    "geo_altitude",

    "squawk",
    "spi",

    "position_source",

    "category",
]


# ============================================================
# GEODESIC MODEL
# ============================================================

GEOD = Geod(
    ellps="WGS84"
)


# ============================================================
# AREA
# ============================================================

def bbox_from_center(
    lat,
    lon,
    radius_km,
):

    radius_m = (
        radius_km
        * 1000.0
    )


    _, lat_max, _ = GEOD.fwd(
        lon,
        lat,
        0,
        radius_m,
    )


    _, lat_min, _ = GEOD.fwd(
        lon,
        lat,
        180,
        radius_m,
    )


    lon_max, _, _ = GEOD.fwd(
        lon,
        lat,
        90,
        radius_m,
    )


    lon_min, _, _ = GEOD.fwd(
        lon,
        lat,
        270,
        radius_m,
    )


    return {

        "lamin": lat_min,
        "lamax": lat_max,

        "lomin": lon_min,
        "lomax": lon_max,

        "extended": 1,
    }


def bbox_from_preset(preset_key):
    resolved = resolve_area_preset(preset_key, BBOX_PRESETS)
    return (
        resolved["name"],
        resolved.get("icao", ""),
        resolved["bbox"],
        resolved.get("polygon"),
    )


def resolve_area():

    if AREA_MODE == "CENTER_RADIUS":


        if SELECTED_PRESET is not None:


            if (
                SELECTED_PRESET
                not in LOCATION_PRESETS
            ):

                raise ValueError(
                    f"Unknown preset: "
                    f"{SELECTED_PRESET}"
                )


            location = (
                LOCATION_PRESETS[
                    SELECTED_PRESET
                ]
            )


            name = location["name"]

            icao = location.get(
                "icao",
                "",
            )


            lat = float(
                location["lat"]
            )

            lon = float(
                location["lon"]
            )


        else:


            name = CUSTOM_CENTER_NAME

            icao = ""


            lat = float(
                CUSTOM_CENTER_LAT
            )

            lon = float(
                CUSTOM_CENTER_LON
            )


        bbox = bbox_from_center(
            lat,
            lon,
            RADIUS_KM,
        )


        return {

            "key": SELECTED_PRESET,
            "name": name,
            "icao": icao,

            "center_lat": lat,
            "center_lon": lon,

            "bbox": bbox,
        }


    if AREA_MODE == "BBOX":

        name, icao, bbox, polygon = bbox_from_preset(
            SELECTED_BBOX_PRESET
        )


        lat = (
            bbox["lamin"]
            + bbox["lamax"]
        ) / 2.0


        lon = (
            bbox["lomin"]
            + bbox["lomax"]
        ) / 2.0


        return {

            "key":
                SELECTED_BBOX_PRESET,

            "name":
                name,

            "icao":
                icao,

            "center_lat":
                lat,

            "center_lon":
                lon,

            "bbox":
                bbox,

            "polygon":
                polygon,
        }


    raise ValueError(
        'AREA_MODE must be '
        '"CENTER_RADIUS" or "BBOX".'
    )


AREA = resolve_area()

BBOX = AREA["bbox"]
AREA_POLYGON = AREA.get("polygon")

WAYPOINTS = (
    load_waypoints_for_area(
        WAYPOINTS_FILE,
        area_key=AREA.get("key"),
        area=AREA,
    )
    if SHOW_WAYPOINTS
    else []
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()

print("=" * 80)

print("CONFIGURATION")

print("=" * 80)


print(
    "Selected area:",
    AREA["name"],
)

print(
    "Waypoints loaded:",
    len(WAYPOINTS),
)


if AREA["icao"]:

    print(
        "ICAO:",
        AREA["icao"],
    )


print(
    "Center:",
    f"{AREA['center_lat']:.4f}, "
    f"{AREA['center_lon']:.4f}",
)


if AREA_MODE == "CENTER_RADIUS":

    print(
        "Radius:",
        f"{RADIUS_KM} km",
    )


print(
    f"BBOX latitude : "
    f"{BBOX['lamin']:.4f} -> "
    f"{BBOX['lamax']:.4f}"
)


print(
    f"BBOX longitude: "
    f"{BBOX['lomin']:.4f} -> "
    f"{BBOX['lomax']:.4f}"
)


print()

print(
    "Separation profile:",
    SEPARATION_PROFILE,
)


print(
    f"RED threshold    : "
    f"H < "
    f"{MIN_HORIZONTAL_SEPARATION_NM:.1f} NM "
    f"AND "
    f"V < "
    f"{MIN_VERTICAL_SEPARATION_FT:.0f} ft"
)


print(
    f"YELLOW envelope : "
    f"H < "
    f"{WARNING_HORIZONTAL_NM:.1f} NM "
    f"AND "
    f"V < "
    f"{WARNING_VERTICAL_FT:.0f} ft"
)


print(
    "GREEN           : "
    "outside warning envelope"
)


print("=" * 80)

print()


# ============================================================
# AIRCRAFT MARKER
# ============================================================

def create_aircraft_marker():

    vertices = np.array([

        [0.00, 1.00],

        [0.12, 0.40],

        [0.70, 0.05],

        [0.70, -0.10],

        [0.12, 0.02],

        [0.09, -0.55],

        [0.32, -0.78],

        [0.32, -0.90],

        [0.00, -0.76],

        [-0.32, -0.90],

        [-0.32, -0.78],

        [-0.09, -0.55],

        [-0.12, 0.02],

        [-0.70, -0.10],

        [-0.70, 0.05],

        [-0.12, 0.40],

        [0.00, 1.00],
    ])


    codes = [

        MplPath.MOVETO,

        *(
            [MplPath.LINETO]
            * (
                len(vertices)
                - 2
            )
        ),

        MplPath.CLOSEPOLY,
    ]


    return MplPath(
        vertices,
        codes,
    )


AIRCRAFT_MARKER = (
    create_aircraft_marker()
)


# ============================================================
# BASIC HELPERS
# ============================================================

def aircraft_is_on_ground(
    row,
):

    value = row.get(
        "on_ground",
        False,
    )


    if pd.isna(value):

        return False


    return bool(value)


def get_aircraft_callsign(
    row,
):

    callsign = row.get(
        "callsign",
        "",
    )


    if not callsign:

        callsign = (
            str(
                row["icao24"]
            )
            .upper()
        )


    return callsign


def horizontal_distance_nm(
    row1,
    row2,
):

    _, _, distance_m = GEOD.inv(

        float(
            row1["longitude"]
        ),

        float(
            row1["latitude"]
        ),

        float(
            row2["longitude"]
        ),

        float(
            row2["latitude"]
        ),
    )


    return (
        distance_m
        / 1852.0
    )


def distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    _, _, distance_m = GEOD.inv(

        lon1,
        lat1,

        lon2,
        lat2,
    )


    return (
        distance_m
        / 1000.0
    )


def get_safety_altitude_ft(
    row,
):

    altitude = row.get(
        "baro_altitude_ft",
        np.nan,
    )


    if (
        pd.isna(altitude)
        or
        not np.isfinite(altitude)
    ):

        return None


    return float(
        altitude
    )


def get_map_width_km(
    bbox,
):

    center_lat = (

        bbox["lamin"]
        + bbox["lamax"]

    ) / 2.0


    _, _, width_m = GEOD.inv(

        bbox["lomin"],
        center_lat,

        bbox["lomax"],
        center_lat,
    )


    return (
        width_m
        / 1000.0
    )


# ============================================================
# GEOGRAPHIC ASPECT CORRECTION
# ============================================================

def set_geographic_aspect(
    ax,
    bbox,
):
    """
    Correct the visual deformation produced by plotting
    latitude and longitude directly.

    The safety area is geodesically circular.

    This correction makes equal physical horizontal and
    vertical distances appear approximately equal on screen.
    """

    mean_latitude = (

        bbox["lamin"]
        + bbox["lamax"]

    ) / 2.0


    latitude_radians = (
        np.deg2rad(
            mean_latitude
        )
    )


    aspect = (

        1.0

        /

        np.cos(
            latitude_radians
        )
    )


    ax.set_aspect(
        aspect,
        adjustable="box",
    )


# ============================================================
# AIRCRAFT TERMINAL TABLE
# ============================================================

def print_aircraft_table(
    aircraft,
):

    print()

    print("=" * 150)

    print(
        "AIRCRAFT FOUND"
    )

    print("=" * 150)


    if aircraft.empty:

        print(
            "No aircraft found."
        )

        print("=" * 150)

        return


    display = (
        aircraft.copy()
    )


    display["ID"] = (
        display.apply(
            get_aircraft_callsign,
            axis=1,
        )
    )


    def number(
        value,
        decimals=0,
        signed=False,
    ):

        if (
            pd.isna(value)
            or
            not np.isfinite(value)
        ):

            return "-"


        if signed:

            return (
                f"{value:+.{decimals}f}"
            )


        return (
            f"{value:.{decimals}f}"
        )


    def ground_text(
        value,
    ):

        if pd.isna(value):

            return "?"

        return (
            "YES"
            if bool(value)
            else "NO"
        )


    display["LAT"] = (
        display["latitude"]
        .map(
            lambda x:
                number(
                    x,
                    5,
                )
        )
    )


    display["LON"] = (
        display["longitude"]
        .map(
            lambda x:
                number(
                    x,
                    5,
                )
        )
    )


    display["BARO_FT"] = (
        display[
            "baro_altitude_ft"
        ]
        .map(
            lambda x:
                number(
                    x,
                    0,
                )
        )
    )


    display["GEO_FT"] = (
        display[
            "geo_altitude_ft"
        ]
        .map(
            lambda x:
                number(
                    x,
                    0,
                )
        )
    )


    display["GS_KT"] = (
        display[
            "velocity_kts"
        ]
        .map(
            lambda x:
                number(
                    x,
                    0,
                )
        )
    )


    display["TRACK_DEG"] = (
        display[
            "true_track"
        ]
        .map(
            lambda x:
                number(
                    x,
                    1,
                )
        )
    )


    display["VS_FPM"] = (
        display[
            "vertical_rate_fpm"
        ]
        .map(
            lambda x:
                number(
                    x,
                    0,
                    signed=True,
                )
        )
    )


    display["GROUND"] = (
        display[
            "on_ground"
        ]
        .map(
            ground_text
        )
    )


    columns = [

        "ID",

        "icao24",

        "LAT",
        "LON",

        "BARO_FT",
        "GEO_FT",

        "GS_KT",

        "TRACK_DEG",

        "VS_FPM",

        "GROUND",

        "origin_country",
    ]


    print(
        display[
            columns
        ]
        .to_string(
            index=False
        )
    )


    print("=" * 150)


# ============================================================
# SAFETY ANALYSIS
# ============================================================
#
# Pair classification, look-ahead prediction and CPA/TCPA calculations are
# implemented in opensky_safety.py so the live snapshot and historical Trino
# workflows use the same logic.
#


# ============================================================
# DOWNLOAD OPENSKY
# ============================================================

print(
    "Downloading OpenSky snapshot..."
)


response = requests.get(

    OPENSKY_URL,

    params=BBOX,

    timeout=30,
)


response.raise_for_status()


payload = (
    response.json()
)


states = (
    payload.get(
        "states"
    )

    or []
)


df = pd.DataFrame(

    states,

    columns=COLUMNS,
)


# ============================================================
# CLEAN DATA
# ============================================================

if not df.empty:


    df["callsign"] = (

        df["callsign"]

        .fillna("")

        .str.strip()
    )


    numeric_columns = [

        "longitude",
        "latitude",

        "baro_altitude",
        "geo_altitude",

        "velocity",
        "true_track",
        "vertical_rate",
    ]


    for column in numeric_columns:


        df[column] = (
            pd.to_numeric(

                df[column],

                errors="coerce",
            )
        )


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


# ============================================================
# SNAPSHOT TIME
# ============================================================

payload_time = (
    payload.get(
        "time"
    )
)


if payload_time is None:

    snapshot_time = (
        datetime.now(
            timezone.utc
        )
    )

else:

    snapshot_time = (
        datetime.fromtimestamp(

            payload_time,

            tz=timezone.utc,
        )
    )


print()

print(
    "OpenSky snapshot:",
    snapshot_time,
)


print(
    "Aircraft found:",
    len(df),
)


# ============================================================
# SAVE RAW SNAPSHOT
# ============================================================

df.to_csv(

    CSV_FILE,

    index=False,
)


print(
    "CSV saved to:",
    CSV_FILE,
)


# ============================================================
# FILTER AIRCRAFT
# ============================================================

print()

print(
    "Aircraft filtering:"
)


valid_position_mask = (

    df["latitude"].notna()

    &

    df["longitude"].notna()

    &

    np.isfinite(
        df["latitude"]
    )

    &

    np.isfinite(
        df["longitude"]
    )
)


invalid_position_count = int(

    (
        ~valid_position_mask
    )
    .sum()
)


aircraft = (
    df[
        valid_position_mask
    ]
    .copy()
)

# Polygon presets are queried through their enclosing BBOX and clipped locally
# so Trino/OpenSky can remain rectangular while the study area is exact.
aircraft = filter_dataframe_to_area(
    aircraft,
    AREA,
    latitude_column="latitude",
    longitude_column="longitude",
)

if FILTER_CRUISE_FL_BAND:
    cruise_mask = (
        aircraft["baro_altitude_ft"].notna()
        & (aircraft["baro_altitude_ft"] >= CRUISE_FL_MIN * 100.0)
        & (aircraft["baro_altitude_ft"] <= CRUISE_FL_MAX * 100.0)
    )
    aircraft = aircraft.loc[cruise_mask].copy()


ground_count = int(

    (
        aircraft["on_ground"]
        == True
    )
    .sum()
)


airborne_count = int(

    (
        aircraft["on_ground"]
        == False
    )
    .sum()
)


unknown_ground_count = int(

    aircraft[
        "on_ground"
    ]
    .isna()
    .sum()
)


print(
    f"  Raw OpenSky states: "
    f"{len(df)}"
)


print(
    f"  Invalid/missing position: "
    f"{invalid_position_count}"
)


print(
    f"  Valid position: "
    f"{len(aircraft)}"
)


print(
    f"  On ground: "
    f"{ground_count}"
)


print(
    f"  Airborne: "
    f"{airborne_count}"
)


print(
    f"  Unknown ground state: "
    f"{unknown_ground_count}"
)


if not SHOW_GROUND_AIRCRAFT:


    aircraft = aircraft[

        aircraft["on_ground"]
        == False

    ].copy()


print(
    f"  Aircraft plotted: "
    f"{len(aircraft)}"
)


# ============================================================
# PRINT ALL AIRCRAFT
# ============================================================

print_aircraft_table(
    aircraft
)


# ============================================================
# SAFETY ANALYSIS
# ============================================================

(
    pair_results,
    aircraft_states,
    LOOKAHEAD_INTERVAL_STATES,
    skipped_pairs,

) = analyse_aircraft_snapshot(
    aircraft,
    min_horizontal_nm=MIN_HORIZONTAL_SEPARATION_NM,
    min_vertical_ft=MIN_VERTICAL_SEPARATION_FT,
    warning_factor=WARNING_FACTOR,
    lookahead_minutes=LOOKAHEAD_MINUTES,
    prediction_step_seconds=PREDICTION_STEP_SECONDS,
    check_ground_aircraft=CHECK_GROUND_AIRCRAFT_SAFETY,
)

warnings = [
    pair for pair in pair_results
    if pair["state_current"] == STATE_PROXIMITY
]

high_risk_pairs = [
    pair for pair in pair_results
    if pair["state_current"] == STATE_HIGH_RISK
]

conflicts = [
    pair for pair in pair_results
    if pair["state_current"] == STATE_CONFLICT
]


print()

print("=" * 100)

print(
    "SAFETY ANALYSIS"
)

print("=" * 100)


print(
    f"Profile: "
    f"{SEPARATION_PROFILE}"
)


print(
    f"RED safety area:"
)


print(
    f"  Horizontal < "
    f"{MIN_HORIZONTAL_SEPARATION_NM:.1f} NM"
)


print(
    f"  Vertical   < "
    f"{MIN_VERTICAL_SEPARATION_FT:.0f} ft"
)


print()

print(
    f"WARNING envelope (yellow/orange):"
)


print(
    f"  Horizontal < "
    f"{WARNING_HORIZONTAL_NM:.1f} NM"
)


print(
    f"  Vertical   < "
    f"{WARNING_VERTICAL_FT:.0f} ft"
)


print()

print(
    f"Yellow proximity pairs: "
    f"{len(warnings)}"
)

print(
    f"Orange high-risk pairs: "
    f"{len(high_risk_pairs)}"
)


print(
    f"Red losses of separation: "
    f"{len(conflicts)}"
)


if skipped_pairs > 0:


    print(
        f"Pairs with incomplete altitude data: "
        f"{skipped_pairs}"
    )


# ============================================================
# WARNINGS
# ============================================================

if warnings:


    print()

    print(
        "YELLOW - PROXIMITY"
    )

    print("-" * 100)


    for pair in warnings:


        if (
            pair["vertical_sep_ft"]
            is None
        ):


            vertical_text = (
                "unknown"
            )


        else:


            vertical_text = (
                f"{pair['vertical_sep_ft']:.0f} ft"
            )


        print(

            f"{pair['callsign_a']:>10} "
            f"<-> "
            f"{pair['callsign_b']:<10}"

            f" | "

            f"H = "
            f"{pair['horizontal_sep_nm']:6.2f} NM"

            f" | "

            f"V = "
            f"{vertical_text}"
        )


# ============================================================
# HIGH RISK
# ============================================================

if high_risk_pairs:
    print()
    print("ORANGE - HIGH RISK")
    print("-" * 100)

    for pair in high_risk_pairs:
        vertical_value = pair.get("vertical_sep_ft")
        vertical_text = (
            "unknown"
            if vertical_value is None
            else f"{vertical_value:.0f} ft"
        )
        print(
            f"{pair['callsign_a']:>10} <-> {pair['callsign_b']:<10}"
            f" | H = {pair['horizontal_sep_nm']:6.2f} NM"
            f" | V = {vertical_text}"
        )


# ============================================================
# CONFLICTS
# ============================================================

if conflicts:


    print()

    print(
        "RED - LOSS OF SEPARATION"
    )

    print("-" * 100)


    for pair in conflicts:


        print(

            f"{pair['callsign_a']:>10} "
            f"<-> "
            f"{pair['callsign_b']:<10}"

            f" | "

            f"H = "
            f"{pair['horizontal_sep_nm']:6.2f} NM"

            f" | "

            f"V = "
            f"{pair['vertical_sep_ft']:.0f} ft"
        )


print("=" * 100)


# Export pairwise safety results separately from the raw OpenSky snapshot.
SEPARATION_RESULTS_FILE = (
    PROJECT_ROOT / "results" / "figures" / "opensky_snapshot_separation.csv"
)
SEPARATION_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

relevant_pairs = relevant_pair_results(pair_results)
pd.DataFrame(relevant_pairs).to_csv(
    SEPARATION_RESULTS_FILE,
    index=False,
)

print(
    "Relevant separation cases saved to:",
    SEPARATION_RESULTS_FILE,
)


from opensky_plot import render_map

fig, ax = render_map(
    aircraft=aircraft, area=AREA, bbox=BBOX,
    aircraft_states=aircraft_states, warnings=warnings, conflicts=conflicts,
    snapshot_time=snapshot_time, map_file=MAP_FILE, config=globals(),
    waypoints=WAYPOINTS,
)
