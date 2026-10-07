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

SELECTED_BBOX_PRESET = "NORTHERN_SPAIN"


BBOX_PRESETS_FILE = Path(__file__).with_name(
    "bbox_presets.json"
)


def load_bbox_presets():

    try:
        with BBOX_PRESETS_FILE.open(
            encoding="utf-8"
        ) as presets_file:
            presets = json.load(presets_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"BBOX presets file not found: "
            f"{BBOX_PRESETS_FILE}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in BBOX presets file: "
            f"{BBOX_PRESETS_FILE}"
        ) from error

    if not isinstance(presets, dict):
        raise ValueError(
            "BBOX presets must be a JSON object"
        )

    return presets


BBOX_PRESETS = load_bbox_presets()


# ============================================================
# 2. AIRCRAFT DISPLAY
# ============================================================

SHOW_GROUND_AIRCRAFT = False

SHOW_LABELS = True
SHOW_CITIES = True

SHOW_ALTITUDE = True
SHOW_GROUND_SPEED = True
SHOW_VERTICAL_SPEED = True


# ============================================================
# 3. TRAJECTORY
# ============================================================

SHOW_TRAJECTORY_VECTOR = True

TRAJECTORY_MINUTES = 2


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


# ============================================================
# 5. WARNING ENVELOPE
# ============================================================
#
# RED:
#
#   H < minimum horizontal
#   AND
#   V < minimum vertical
#
# YELLOW:
#
#   H < warning horizontal
#   AND
#   V < warning vertical
#
#   but not already red.
#
# GREEN:
#
#   everything else.
#
# This avoids false warnings such as:
#
#   H = 43 NM
#   V = 0 ft
#
# which is NORMAL / GREEN.
#
# ============================================================

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

STATE_NORMAL = "NORMAL"

STATE_WARNING = "WARNING"

STATE_CONFLICT = "CONFLICT"


# ============================================================
# 8. SAFETY COLOURS
# ============================================================
#
# GREEN  = normal
# YELLOW = proximity warning
# RED    = loss of separation
#
# ============================================================

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

    if preset_key not in BBOX_PRESETS:
        raise ValueError(
            f"Unknown BBOX preset: {preset_key}"
        )

    preset = BBOX_PRESETS[preset_key]
    definition_type = preset.get("type")

    if definition_type == "center":
        center = preset["center"]
        bbox = bbox_from_center(
            float(center["lat"]),
            float(center["lon"]),
            float(preset["radius_km"]),
        )
    elif definition_type == "vertices":
        bbox = dict(preset["vertices"])
        bbox.setdefault("extended", 1)
    else:
        raise ValueError(
            f"BBOX preset '{preset_key}' must use "
            'type "center" or "vertices"'
        )

    return preset["name"], preset.get("icao", ""), bbox


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

            "name": name,
            "icao": icao,

            "center_lat": lat,
            "center_lon": lon,

            "bbox": bbox,
        }


    if AREA_MODE == "BBOX":

        name, icao, bbox = bbox_from_preset(
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
        }


    raise ValueError(
        'AREA_MODE must be '
        '"CENTER_RADIUS" or "BBOX".'
    )


AREA = resolve_area()

BBOX = AREA["bbox"]


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
# SAFETY STATES
# ============================================================

STATE_PRIORITY = {

    STATE_NORMAL: 0,

    STATE_WARNING: 1,

    STATE_CONFLICT: 2,
}


def combine_states(
    current_state,
    new_state,
):

    if (
        STATE_PRIORITY[
            new_state
        ]
        >
        STATE_PRIORITY[
            current_state
        ]
    ):

        return new_state


    return current_state


# ============================================================
# PAIR CLASSIFICATION
# ============================================================

def classify_pair(
    horizontal_nm,
    vertical_ft,
):
    """
    RED:
        Inside minimum separation volume.

        horizontal < minimum
        AND
        vertical < minimum

    YELLOW:
        Inside larger warning envelope,
        but not inside red volume.

        horizontal < warning
        AND
        vertical < warning

    GREEN:
        Outside warning envelope.
    """

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    if (
        horizontal_nm
        < MIN_HORIZONTAL_SEPARATION_NM

        and

        vertical_ft
        < MIN_VERTICAL_SEPARATION_FT
    ):

        return STATE_CONFLICT


    # --------------------------------------------------------
    # YELLOW
    # --------------------------------------------------------

    if (
        horizontal_nm
        < WARNING_HORIZONTAL_NM

        and

        vertical_ft
        < WARNING_VERTICAL_FT
    ):

        return STATE_WARNING


    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------

    return STATE_NORMAL


# ============================================================
# SEPARATION ANALYSIS
# ============================================================

def analyse_separation(
    aircraft,
):

    pair_results = []


    aircraft_states = {

        index:
            STATE_NORMAL

        for index
        in aircraft.index
    }


    skipped_pairs = 0


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
            aircraft_is_on_ground(
                aircraft_a
            )

            and

            not CHECK_GROUND_AIRCRAFT_SAFETY
        ):

            continue


        altitude_a = (
            get_safety_altitude_ft(
                aircraft_a
            )
        )


        for j in range(
            i + 1,
            len(rows)
        ):


            index_b, aircraft_b = (
                rows[j]
            )


            if (
                aircraft_is_on_ground(
                    aircraft_b
                )

                and

                not CHECK_GROUND_AIRCRAFT_SAFETY
            ):

                continue


            altitude_b = (
                get_safety_altitude_ft(
                    aircraft_b
                )
            )


            horizontal_nm = (
                horizontal_distance_nm(
                    aircraft_a,
                    aircraft_b,
                )
            )


            # ------------------------------------------------
            # MISSING ALTITUDE
            # ------------------------------------------------

            if (
                altitude_a is None
                or
                altitude_b is None
            ):

                skipped_pairs += 1


                # Cannot classify as conflict without vertical
                # information.
                #
                # A close horizontal encounter is kept as a
                # warning because altitude information is
                # incomplete.

                if (
                    horizontal_nm
                    < WARNING_HORIZONTAL_NM
                ):

                    pair_state = (
                        STATE_WARNING
                    )


                    aircraft_states[
                        index_a
                    ] = combine_states(

                        aircraft_states[
                            index_a
                        ],

                        pair_state,
                    )


                    aircraft_states[
                        index_b
                    ] = combine_states(

                        aircraft_states[
                            index_b
                        ],

                        pair_state,
                    )


                    pair_results.append({

                        "index_a":
                            index_a,

                        "index_b":
                            index_b,

                        "callsign_a":
                            get_aircraft_callsign(
                                aircraft_a
                            ),

                        "callsign_b":
                            get_aircraft_callsign(
                                aircraft_b
                            ),

                        "horizontal_nm":
                            horizontal_nm,

                        "vertical_ft":
                            None,

                        "state":
                            pair_state,
                    })


                continue


            # ------------------------------------------------
            # COMPLETE 3D SEPARATION
            # ------------------------------------------------

            vertical_ft = abs(

                altitude_a
                - altitude_b
            )


            pair_state = classify_pair(

                horizontal_nm,
                vertical_ft,
            )


            if (
                pair_state
                == STATE_NORMAL
            ):

                continue


            aircraft_states[
                index_a
            ] = combine_states(

                aircraft_states[
                    index_a
                ],

                pair_state,
            )


            aircraft_states[
                index_b
            ] = combine_states(

                aircraft_states[
                    index_b
                ],

                pair_state,
            )


            pair_results.append({

                "index_a":
                    index_a,

                "index_b":
                    index_b,

                "callsign_a":
                    get_aircraft_callsign(
                        aircraft_a
                    ),

                "callsign_b":
                    get_aircraft_callsign(
                        aircraft_b
                    ),

                "horizontal_nm":
                    horizontal_nm,

                "vertical_ft":
                    vertical_ft,

                "state":
                    pair_state,
            })


    return (
        pair_results,
        aircraft_states,
        skipped_pairs,
    )


# ============================================================
# STATE COLOR
# ============================================================

def get_state_color(
    state,
):

    if (
        state
        == STATE_CONFLICT
    ):

        return (
            CONFLICT_AIRCRAFT_COLOR
        )


    if (
        state
        == STATE_WARNING
    ):

        return (
            WARNING_AIRCRAFT_COLOR
        )


    return (
        NORMAL_AIRCRAFT_COLOR
    )


# ============================================================
# SAFETY AREA
# ============================================================

def create_safety_area(
    lon,
    lat,
):
    """
    Generate the boundary of the horizontal safety area.

    Every point is exactly:

        MIN_HORIZONTAL_SEPARATION_NM

    from the aircraft according to WGS84.

    Therefore the physical protected area is circular in the
    horizontal plane and independent of aircraft heading.
    """

    radius_m = (

        MIN_HORIZONTAL_SEPARATION_NM

        * 1852.0
    )


    azimuths = np.linspace(

        0.0,
        360.0,

        SAFETY_AREA_POINTS
        + 1,
    )


    area_lon = []

    area_lat = []


    for azimuth in azimuths:


        point_lon, point_lat, _ = (
            GEOD.fwd(

                lon,
                lat,

                float(
                    azimuth
                ),

                radius_m,
            )
        )


        area_lon.append(
            point_lon
        )


        area_lat.append(
            point_lat
        )


    return (
        np.asarray(
            area_lon
        ),

        np.asarray(
            area_lat
        ),
    )


def draw_safety_area(
    ax,
    row,
    state,
):

    if not SHOW_SAFETY_AREA:

        return


    if (
        aircraft_is_on_ground(
            row
        )

        and

        not CHECK_GROUND_AIRCRAFT_SAFETY
    ):

        return


    lon = float(
        row["longitude"]
    )


    lat = float(
        row["latitude"]
    )


    area_lon, area_lat = (
        create_safety_area(
            lon,
            lat,
        )
    )


    color = (
        get_state_color(
            state
        )
    )


    if SHOW_SAFETY_AREA_FILL:


        ax.fill(

            area_lon,
            area_lat,

            facecolor=color,

            edgecolor="none",

            alpha=(
                SAFETY_AREA_FILL_ALPHA
            ),

            zorder=11,

            clip_on=True,
        )


    ax.plot(

        area_lon,
        area_lat,

        color=color,

        linewidth=(
            SAFETY_AREA_LINEWIDTH
        ),

        linestyle="--",

        alpha=(
            SAFETY_AREA_ALPHA
        ),

        zorder=12,

        clip_on=True,
    )


# ============================================================
# CITIES
# ============================================================

def get_relevant_cities(
    bbox,
):

    gc = (
        geonamescache
        .GeonamesCache()
    )


    all_cities = (
        gc.get_cities()
    )


    map_width = (
        get_map_width_km(
            bbox
        )
    )


    if map_width < 100:

        min_population = 20_000

        min_distance_km = 10


    elif map_width < 200:

        min_population = 40_000

        min_distance_km = 20


    elif map_width < 500:

        min_population = 70_000

        min_distance_km = 30


    elif map_width < 1000:

        min_population = 150_000

        min_distance_km = 60


    else:

        min_population = 400_000

        min_distance_km = 100


    candidates = []


    for city in all_cities.values():


        try:


            lat = float(
                city["latitude"]
            )


            lon = float(
                city["longitude"]
            )


            population = int(

                city.get(
                    "population",
                    0,
                )

                or 0
            )


        except (
            ValueError,
            TypeError,
        ):

            continue


        if not (
            np.isfinite(lat)
            and
            np.isfinite(lon)
        ):

            continue


        if not (

            bbox["lamin"]
            <= lat
            <= bbox["lamax"]

            and

            bbox["lomin"]
            <= lon
            <= bbox["lomax"]

        ):

            continue


        if (
            population
            < min_population
        ):

            continue


        candidates.append({

            "name":
                city["name"],

            "latitude":
                lat,

            "longitude":
                lon,

            "population":
                population,
        })


    candidates.sort(

        key=lambda city:
            city["population"],

        reverse=True,
    )


    selected = []


    for city in candidates:


        too_close = False


        for other in selected:


            city_distance = (
                distance_km(

                    city["latitude"],
                    city["longitude"],

                    other["latitude"],
                    other["longitude"],
                )
            )


            if (
                city_distance
                < min_distance_km
            ):

                too_close = True

                break


        if not too_close:

            selected.append(
                city
            )


        if (
            len(selected)
            >= MAX_CITIES
        ):

            break


    return selected


# ============================================================
# AXES
# ============================================================

def get_axis_step(
    bbox,
):

    span = max(

        bbox["lomax"]
        - bbox["lomin"],

        bbox["lamax"]
        - bbox["lamin"],
    )


    if span <= 1.2:

        return 0.1


    if span <= 2.5:

        return 0.2


    if span <= 5.0:

        return 0.5


    if span <= 10.0:

        return 1.0


    return 2.0


AXIS_STEP = (
    get_axis_step(
        BBOX
    )
)


def coordinate_formatter(
    value,
    position,
):

    return (
        f"{value:.1f}°"
    )


# ============================================================
# SCALE BAR
# ============================================================

def add_scale_bar_outside(
    ax,
    bbox,
):

    lat_ref = (

        bbox["lamin"]

        + 0.1
        * (
            bbox["lamax"]
            - bbox["lamin"]
        )
    )


    _, _, map_width_m = GEOD.inv(

        bbox["lomin"],
        lat_ref,

        bbox["lomax"],
        lat_ref,
    )


    target = (
        map_width_m
        / 5.0
    )


    possible_lengths = np.array([

        1_000,
        2_000,
        5_000,

        10_000,
        20_000,
        25_000,

        50_000,
        100_000,
        200_000,
        500_000,
    ])


    scale_m = possible_lengths[

        np.argmin(

            np.abs(

                possible_lengths
                - target
            )
        )
    ]


    lon_end, _, _ = GEOD.fwd(

        bbox["lomin"],
        lat_ref,

        90,

        scale_m,
    )


    fraction = (

        lon_end
        - bbox["lomin"]

    ) / (

        bbox["lomax"]
        - bbox["lomin"]
    )


    x0 = 0.04

    x1 = (
        x0
        + fraction
    )

    y = 0.05


    ax.plot(

        [x0, x1],
        [y, y],

        transform=ax.transAxes,

        color="black",

        linewidth=3,

        clip_on=False,
    )


    for x in (
        x0,
        x1,
    ):


        ax.plot(

            [x, x],

            [
                y - 0.012,
                y + 0.012,
            ],

            transform=ax.transAxes,

            color="black",

            linewidth=2,

            clip_on=False,
        )


    ax.text(

        (x0 + x1) / 2,

        y + 0.025,

        f"{scale_m / 1000:.0f} km",

        transform=ax.transAxes,

        ha="center",
        va="bottom",

        fontsize=10,

        weight="bold",

        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.7,
            "pad": 3,
        },

        clip_on=False,
    )


# ============================================================
# TRAJECTORY
# ============================================================

def calculate_trajectory_endpoint(
    row,
    minutes,
):

    if aircraft_is_on_ground(
        row
    ):

        return None


    track = row[
        "true_track"
    ]


    speed_ms = row[
        "velocity"
    ]


    if (
        pd.isna(track)
        or
        pd.isna(speed_ms)

        or

        not np.isfinite(track)
        or
        not np.isfinite(speed_ms)

        or

        speed_ms <= 0
    ):

        return None


    lon = float(
        row["longitude"]
    )


    lat = float(
        row["latitude"]
    )


    distance_m = (

        float(
            speed_ms
        )

        * minutes

        * 60.0
    )


    end_lon, end_lat, _ = (
        GEOD.fwd(

            lon,
            lat,

            float(
                track
            ),

            distance_m,
        )
    )


    return (
        end_lon,
        end_lat,
    )


def draw_trajectory_vector(
    ax,
    row,
    state,
):

    endpoint = (
        calculate_trajectory_endpoint(

            row,

            TRAJECTORY_MINUTES,
        )
    )


    if endpoint is None:

        return


    lon = float(
        row["longitude"]
    )


    lat = float(
        row["latitude"]
    )


    end_lon, end_lat = (
        endpoint
    )


    color = (
        get_state_color(
            state
        )
    )


    arrow = FancyArrowPatch(

        (
            lon,
            lat,
        ),

        (
            end_lon,
            end_lat,
        ),

        transform=ax.transData,

        arrowstyle="-|>",

        mutation_scale=8,

        linewidth=1.15,

        color=color,

        alpha=0.72,

        zorder=14,

        clip_on=True,
    )


    ax.add_patch(
        arrow
    )


# ============================================================
# AIRCRAFT LABEL
# ============================================================

def build_aircraft_label_box(
    row,
    fontsize,
):

    lines = []


    callsign = (
        get_aircraft_callsign(
            row
        )
    )


    lines.append(

        TextArea(

            callsign,

            textprops={

                "color":
                    "white",

                "fontsize":
                    fontsize,

                "fontweight":
                    "bold",
            },
        )
    )


    on_ground = (
        aircraft_is_on_ground(
            row
        )
    )


    if on_ground:


        lines.append(

            TextArea(

                "GND",

                textprops={

                    "color":
                        GROUND_COLOR,

                    "fontsize":
                        fontsize - 0.1,

                    "fontweight":
                        "bold",
                },
            )
        )


    elif SHOW_ALTITUDE:


        altitude = row.get(

            "baro_altitude_ft",

            np.nan,
        )


        if (
            pd.notna(altitude)
            and
            np.isfinite(altitude)
        ):


            lines.append(

                TextArea(

                    f"{altitude:,.0f} ft",

                    textprops={

                        "color":
                            "white",

                        "fontsize":
                            fontsize - 0.3,
                    },
                )
            )


    if SHOW_GROUND_SPEED:

        speed = row.get(
            "velocity_kts",
            np.nan,
        )

        if (
            pd.notna(speed)
            and
            np.isfinite(speed)
        ):

            # True track measured clockwise
            # from geographic (true) north.
            track = row.get(
                "true_track",
                np.nan,
            )

            if (
                pd.notna(track)
                and
                np.isfinite(track)
            ):

                track_deg = (
                    int(round(float(track)))
                    % 360
                )

                speed_text = (
                    f"{speed:.0f} kt  "
                    f"{track_deg:03d}°"
                )

            else:

                speed_text = (
                    f"{speed:.0f} kt"
                )

            lines.append(

                TextArea(

                    speed_text,

                    textprops={

                        "color":
                            "white",

                        "fontsize":
                            fontsize - 0.3,
                    },
                )
            )

    if SHOW_VERTICAL_SPEED:


        vertical_speed = row.get(

            "vertical_rate_fpm",

            np.nan,
        )


        if (
            pd.notna(vertical_speed)
            and
            np.isfinite(vertical_speed)
        ):


            if (
                vertical_speed
                >
                VERTICAL_SPEED_LEVEL_THRESHOLD_FPM
            ):


                text = (
                    f"↑ "
                    f"{vertical_speed:.0f} fpm"
                )


                color = (
                    VERTICAL_SPEED_CLIMB_COLOR
                )


            elif (
                vertical_speed
                <
                -VERTICAL_SPEED_LEVEL_THRESHOLD_FPM
            ):


                text = (
                    f"↓ "
                    f"{abs(vertical_speed):.0f} fpm"
                )


                color = (
                    VERTICAL_SPEED_DESCENT_COLOR
                )


            else:


                text = (
                    "→ 0 fpm"
                )


                color = (
                    VERTICAL_SPEED_LEVEL_COLOR
                )


            lines.append(

                TextArea(

                    text,

                    textprops={

                        "color":
                            color,

                        "fontsize":
                            fontsize - 0.2,

                        "fontweight":
                            "bold",
                    },
                )
            )


    return VPacker(

        children=lines,

        align="left",

        pad=0,

        sep=1,
    )


# ============================================================
# LABEL BBOX HELPERS
# ============================================================

def expand_bbox(
    box,
    padding,
):

    return Bbox.from_extents(

        box.x0 - padding,
        box.y0 - padding,

        box.x1 + padding,
        box.y1 + padding,
    )


def bbox_is_finite(
    box,
):

    return np.all(

        np.isfinite([

            box.x0,
            box.y0,

            box.x1,
            box.y1,
        ])
    )


def bbox_inside(
    inner,
    outer,
    margin=2,
):

    return (

        inner.x0
        >= outer.x0 + margin

        and

        inner.y0
        >= outer.y0 + margin

        and

        inner.x1
        <= outer.x1 - margin

        and

        inner.y1
        <= outer.y1 - margin
    )


def intersects_any(
    box,
    boxes,
):

    return any(

        box.overlaps(
            other
        )

        for other
        in boxes
    )


# ============================================================
# LABEL POSITIONS
# ============================================================

DIRECTIONS = [

    (+1.0, +0.75, "left", "bottom"),

    (-1.0, +0.75, "right", "bottom"),

    (+1.0, -0.75, "left", "top"),

    (-1.0, -0.75, "right", "top"),

    (+1.2, 0.0, "left", "center"),

    (-1.2, 0.0, "right", "center"),

    (0.0, +1.0, "center", "bottom"),

    (0.0, -1.0, "center", "top"),
]


def direction_penalty(
    sx,
    sy,
    lon,
    lat,
    bbox,
):

    fx = (

        lon
        - bbox["lomin"]

    ) / (

        bbox["lomax"]
        - bbox["lomin"]
    )


    fy = (

        lat
        - bbox["lamin"]

    ) / (

        bbox["lamax"]
        - bbox["lamin"]
    )


    penalty = 0


    if (
        fx < 0.20
        and
        sx < 0
    ):

        penalty += 100


    if (
        fx > 0.80
        and
        sx > 0
    ):

        penalty += 100


    if (
        fy < 0.20
        and
        sy < 0
    ):

        penalty += 100


    if (
        fy > 0.80
        and
        sy > 0
    ):

        penalty += 100


    if (
        fx < 0.50
        and
        sx < 0
    ):

        penalty += 3


    if (
        fx > 0.50
        and
        sx > 0
    ):

        penalty += 3


    return penalty


def candidate_positions(
    lon,
    lat,
    bbox,
    rings,
):

    directions = sorted(

        DIRECTIONS,

        key=lambda d:
            direction_penalty(

                d[0],
                d[1],

                lon,
                lat,

                bbox,
            ),
    )


    candidates = []


    for radius in rings:


        for (
            sx,
            sy,
            ha,
            va,
        ) in directions:


            candidates.append({

                "dx":
                    sx * radius,

                "dy":
                    sy * radius,

                "ha":
                    ha,

                "va":
                    va,
            })


    return candidates


def get_alignment(
    ha,
    va,
):

    return (

        {
            "left": 0.0,
            "center": 0.5,
            "right": 1.0,
        }[ha],

        {
            "bottom": 0.0,
            "center": 0.5,
            "top": 1.0,
        }[va],
    )


# ============================================================
# AIRCRAFT ICON OBSTACLES
# ============================================================

def build_aircraft_icon_boxes(
    ax,
    aircraft,
    fig,
):

    boxes = []


    marker_diameter_points = (
        np.sqrt(
            AIRCRAFT_SIZE
        )
    )


    marker_diameter_pixels = (

        marker_diameter_points

        * fig.dpi

        / 72.0
    )


    radius = max(

        12,

        marker_diameter_pixels
        * 0.65,
    )


    for _, row in aircraft.iterrows():


        x, y = (
            ax.transData.transform(
                (
                    float(
                        row["longitude"]
                    ),

                    float(
                        row["latitude"]
                    ),
                )
            )
        )


        boxes.append(

            Bbox.from_extents(

                x - radius,
                y - radius,

                x + radius,
                y + radius,
            )
        )


    return boxes


# ============================================================
# CITY LABEL
# ============================================================

def place_city_label(
    ax,
    renderer,
    city,
    bbox_geo,
    rings,
    fontsize,
    occupied,
    obstacles,
):

    axes_box = (
        ax.get_window_extent(
            renderer=renderer
        )
    )


    lon = float(
        city["longitude"]
    )


    lat = float(
        city["latitude"]
    )


    for candidate in candidate_positions(

        lon,
        lat,

        bbox_geo,

        rings,

    ):


        artist = ax.annotate(

            city["name"],

            xy=(
                lon,
                lat,
            ),

            xytext=(

                candidate["dx"],
                candidate["dy"],
            ),

            textcoords="offset points",

            fontsize=fontsize,

            color="white",

            ha=candidate["ha"],
            va=candidate["va"],

            zorder=9,

            path_effects=[

                pe.withStroke(

                    linewidth=3,

                    foreground="black",
                )
            ],
        )


        raw_box = (
            artist.get_window_extent(
                renderer=renderer
            )
        )


        if not bbox_is_finite(
            raw_box
        ):

            artist.remove()

            continue


        box = expand_bbox(
            raw_box,
            3,
        )


        if not bbox_inside(
            box,
            axes_box,
        ):

            artist.remove()

            continue


        if intersects_any(
            box,
            occupied,
        ):

            artist.remove()

            continue


        if intersects_any(
            box,
            obstacles,
        ):

            artist.remove()

            continue


        return (
            artist,
            box,
        )


    return (
        None,
        None,
    )


# ============================================================
# AIRCRAFT LABEL
# ============================================================

def place_aircraft_label(
    ax,
    fig,
    renderer,
    row,
    bbox_geo,
    rings,
    fontsize,
    occupied,
    obstacles,
):

    axes_box = (
        ax.get_window_extent(
            renderer=renderer
        )
    )


    lon = float(
        row["longitude"]
    )


    lat = float(
        row["latitude"]
    )


    for candidate in candidate_positions(

        lon,
        lat,

        bbox_geo,

        rings,

    ):


        label = (
            build_aircraft_label_box(
                row,
                fontsize,
            )
        )


        artist = AnnotationBbox(

            label,

            (
                lon,
                lat,
            ),

            xybox=(

                candidate["dx"],
                candidate["dy"],
            ),

            xycoords="data",

            boxcoords="offset points",

            box_alignment=(
                get_alignment(

                    candidate["ha"],
                    candidate["va"],
                )
            ),

            frameon=True,

            bboxprops={

                "boxstyle":
                    "round,pad=0.25",

                "facecolor":
                    "black",

                "edgecolor":
                    "white",

                "linewidth":
                    0.4,

                "alpha":
                    0.82,
            },

            zorder=30,
        )


        ax.add_artist(
            artist
        )


        fig.canvas.draw()


        raw_box = (
            artist.get_window_extent(
                renderer=renderer
            )
        )


        if not bbox_is_finite(
            raw_box
        ):

            artist.remove()

            continue


        box = expand_bbox(
            raw_box,
            4,
        )


        if not bbox_inside(
            box,
            axes_box,
        ):

            artist.remove()

            continue


        if intersects_any(
            box,
            occupied,
        ):

            artist.remove()

            continue


        if intersects_any(
            box,
            obstacles,
        ):

            artist.remove()

            continue


        return (
            artist,
            box,
        )


    return (
        None,
        None,
    )


# ============================================================
# AIRCRAFT DENSITY ORDER
# ============================================================

def sort_aircraft_by_density(
    aircraft,
    ax,
):

    if len(aircraft) <= 1:

        return aircraft


    coordinates = aircraft[

        [
            "longitude",
            "latitude",
        ]

    ].to_numpy(
        dtype=float
    )


    screen = (
        ax.transData.transform(
            coordinates
        )
    )


    nearest = []


    for i in range(
        len(screen)
    ):


        distances = np.linalg.norm(

            screen
            - screen[i],

            axis=1,
        )


        distances[i] = np.inf


        nearest.append(

            np.min(
                distances
            )
        )


    result = (
        aircraft.copy()
    )


    result[
        "_nearest_px"
    ] = nearest


    return result.sort_values(

        "_nearest_px",

        ascending=True,
    )


# ============================================================
# FINAL LABEL VALIDATION
# ============================================================

def validate_layout(
    ax,
    fig,
    city_artists,
    aircraft_artists,
    aircraft_icon_boxes,
):

    fig.canvas.draw()


    renderer = (
        fig.canvas
        .get_renderer()
    )


    axes_box = (
        ax.get_window_extent(
            renderer=renderer
        )
    )


    artists = (

        city_artists
        + aircraft_artists
    )


    boxes = []


    for artist in artists:


        boxes.append(

            expand_bbox(

                artist.get_window_extent(
                    renderer=renderer
                ),

                3,
            )
        )


    intersections = []


    for i in range(
        len(boxes)
    ):


        for j in range(
            i + 1,
            len(boxes)
        ):


            if boxes[i].overlaps(
                boxes[j]
            ):


                intersections.append(
                    (
                        i,
                        j,
                    )
                )


    outside = []


    for i, box in enumerate(
        boxes
    ):


        if not bbox_inside(

            box,
            axes_box,

            margin=1,

        ):


            outside.append(
                i
            )


    over_aircraft = []


    offset = len(
        city_artists
    )


    for i in range(
        len(aircraft_artists)
    ):


        label_box = boxes[
            offset + i
        ]


        for (
            icon_index,
            icon_box,
        ) in enumerate(
            aircraft_icon_boxes
        ):


            if label_box.overlaps(
                icon_box
            ):


                over_aircraft.append(
                    (
                        i,
                        icon_index,
                    )
                )


    return (
        intersections,
        outside,
        over_aircraft,
    )


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
    skipped_pairs,

) = analyse_separation(
    aircraft
)


warnings = [

    pair

    for pair
    in pair_results

    if (
        pair["state"]
        == STATE_WARNING
    )
]


conflicts = [

    pair

    for pair
    in pair_results

    if (
        pair["state"]
        == STATE_CONFLICT
    )
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
    f"YELLOW warning envelope:"
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
    f"Yellow warnings: "
    f"{len(warnings)}"
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
        "YELLOW WARNINGS"
    )

    print("-" * 100)


    for pair in warnings:


        if (
            pair["vertical_ft"]
            is None
        ):


            vertical_text = (
                "unknown"
            )


        else:


            vertical_text = (
                f"{pair['vertical_ft']:.0f} ft"
            )


        print(

            f"{pair['callsign_a']:>10} "
            f"<-> "
            f"{pair['callsign_b']:<10}"

            f" | "

            f"H = "
            f"{pair['horizontal_nm']:6.2f} NM"

            f" | "

            f"V = "
            f"{vertical_text}"
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
            f"{pair['horizontal_nm']:6.2f} NM"

            f" | "

            f"V = "
            f"{pair['vertical_ft']:.0f} ft"
        )


print("=" * 100)


# ============================================================
# CITIES
# ============================================================

cities = []


if SHOW_CITIES:


    cities = (
        get_relevant_cities(
            BBOX
        )
    )


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(

    figsize=(
        22,
        12,
    ),

    dpi=FIG_DPI,
)


fig.subplots_adjust(

    left=0.045,

    right=0.995,

    bottom=0.115,

    top=0.92,
)


ax.set_xlim(

    BBOX["lomin"],

    BBOX["lomax"],
)


ax.set_ylim(

    BBOX["lamin"],

    BBOX["lamax"],
)


# ============================================================
# BASEMAP
# ============================================================

try:


    cx.add_basemap(

        ax,

        source=(
            cx.providers
            .Esri
            .WorldImagery
        ),

        crs="EPSG:4326",

        zoom="auto",

        attribution_size=5,
    )


except Exception as exc:


    print()

    print(
        "WARNING: basemap unavailable:"
    )

    print(
        exc
    )


    ax.set_facecolor(
        "0.15"
    )


# ============================================================
# CORRECT GEOGRAPHIC PROPORTIONS
# ============================================================

set_geographic_aspect(
    ax,
    BBOX,
)


# ============================================================
# AXES
# ============================================================

ax.xaxis.set_major_formatter(

    FuncFormatter(
        coordinate_formatter
    )
)


ax.yaxis.set_major_formatter(

    FuncFormatter(
        coordinate_formatter
    )
)


ax.xaxis.set_major_locator(

    MultipleLocator(
        AXIS_STEP
    )
)


ax.yaxis.set_major_locator(

    MultipleLocator(
        AXIS_STEP
    )
)


ax.set_xlabel(

    "Longitude",

    fontsize=13,

    weight="bold",

    labelpad=12,
)


ax.set_ylabel(

    "Latitude",

    fontsize=13,

    weight="bold",

    labelpad=12,
)


ax.tick_params(

    axis="both",

    labelsize=11,

    width=1.2,

    length=6,
)


ax.grid(

    True,

    color="white",

    linewidth=0.7,

    linestyle="--",

    alpha=0.35,
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(

    f"OpenSky Traffic Snapshot - "
    f"{AREA['name']} - "
    f"{SEPARATION_PROFILE} - "
    + snapshot_time.strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    ),

    fontsize=17,

    y=0.965,
)


# ============================================================
# CITY POINTS
# ============================================================

for city in cities:


    ax.scatter(

        city["longitude"],
        city["latitude"],

        s=28,

        marker="o",

        facecolor="white",

        edgecolor="black",

        linewidth=0.9,

        zorder=8,
    )


# ============================================================
# SAFETY AREAS
# ============================================================

for (
    aircraft_index,
    row,
) in aircraft.iterrows():


    state = (
        aircraft_states.get(

            aircraft_index,

            STATE_NORMAL,
        )
    )


    draw_safety_area(

        ax,

        row,

        state,
    )


# ============================================================
# TRAJECTORY VECTORS
# ============================================================

if SHOW_TRAJECTORY_VECTOR:


    for (
        aircraft_index,
        row,
    ) in aircraft.iterrows():


        state = (
            aircraft_states.get(

                aircraft_index,

                STATE_NORMAL,
            )
        )


        draw_trajectory_vector(

            ax,

            row,

            state,
        )


# ============================================================
# AIRCRAFT SYMBOLS
# ============================================================

for (
    aircraft_index,
    row,
) in aircraft.iterrows():


    lon = float(
        row["longitude"]
    )


    lat = float(
        row["latitude"]
    )


    track = (
        row["true_track"]
    )


    if (
        pd.isna(track)
        or
        not np.isfinite(track)
    ):


        track = 0.0


    rotated_marker = (

        Affine2D()

        .rotate_deg(
            -float(
                track
            )
        )

        .transform_path(
            AIRCRAFT_MARKER
        )
    )


    state = (
        aircraft_states.get(

            aircraft_index,

            STATE_NORMAL,
        )
    )


    aircraft_color = (
        get_state_color(
            state
        )
    )


    ax.scatter(

        lon,
        lat,

        marker=rotated_marker,

        s=AIRCRAFT_SIZE,

        facecolor=aircraft_color,

        edgecolor="black",

        linewidth=0.9,

        zorder=20,
    )


# ============================================================
# INITIAL RENDER
# ============================================================

fig.canvas.draw()


renderer = (
    fig.canvas
    .get_renderer()
)


# ============================================================
# AIRCRAFT ICON OBSTACLES
# ============================================================

aircraft_icon_boxes = (
    build_aircraft_icon_boxes(

        ax,

        aircraft,

        fig,
    )
)


# ============================================================
# LABEL ORDER
# ============================================================

aircraft_sorted = (
    sort_aircraft_by_density(

        aircraft,

        ax,
    )
)


# ============================================================
# LABEL PLACEMENT ATTEMPTS
# ============================================================

PLACEMENT_ATTEMPTS = [

    {

        "city_font":
            8.5,

        "city_rings":
            [
                8,
                12,
                16,
                20,
            ],

        "aircraft_font":
            7.5,

        "aircraft_rings":
            [
                15,
                22,
                30,
                40,
                52,
                65,
            ],
    },


    {

        "city_font":
            8.2,

        "city_rings":
            [
                8,
                12,
                16,
                20,
                24,
            ],

        "aircraft_font":
            7.1,

        "aircraft_rings":
            [
                15,
                22,
                30,
                40,
                52,
                65,
                80,
            ],
    },


    {

        "city_font":
            8.0,

        "city_rings":
            [
                8,
                12,
                16,
                20,
                24,
                28,
            ],

        "aircraft_font":
            6.7,

        "aircraft_rings":
            [
                15,
                22,
                30,
                40,
                52,
                65,
                80,
                95,
            ],
    },
]


final_city_artists = []

final_aircraft_artists = []


layout_valid = False


# ============================================================
# ZERO-INTERSECTION LABEL PLACEMENT
# ============================================================

if SHOW_LABELS:


    for (
        attempt_number,
        config,
    ) in enumerate(

        PLACEMENT_ATTEMPTS,

        start=1,
    ):


        print()

        print(

            f"Label placement attempt "
            f"{attempt_number}..."
        )


        for artist in (

            final_city_artists
            + final_aircraft_artists

        ):


            try:

                artist.remove()


            except Exception:

                pass


        final_city_artists = []

        final_aircraft_artists = []


        occupied = []

        placement_failed = False


        fig.canvas.draw()


        renderer = (
            fig.canvas
            .get_renderer()
        )


        # ----------------------------------------------------
        # CITY LABELS
        # ----------------------------------------------------

        for city in cities:


            artist, box = (
                place_city_label(

                    ax=ax,

                    renderer=renderer,

                    city=city,

                    bbox_geo=BBOX,

                    rings=(
                        config[
                            "city_rings"
                        ]
                    ),

                    fontsize=(
                        config[
                            "city_font"
                        ]
                    ),

                    occupied=occupied,

                    obstacles=(
                        aircraft_icon_boxes
                    ),
                )
            )


            if artist is None:


                print(
                    "Could not place city:",
                    city["name"],
                )


                placement_failed = True

                break


            final_city_artists.append(
                artist
            )


            occupied.append(
                box
            )


        if placement_failed:

            continue


        # ----------------------------------------------------
        # AIRCRAFT LABELS
        # ----------------------------------------------------

        for _, row in (
            aircraft_sorted
            .iterrows()
        ):


            artist, box = (
                place_aircraft_label(

                    ax=ax,

                    fig=fig,

                    renderer=renderer,

                    row=row,

                    bbox_geo=BBOX,

                    rings=(
                        config[
                            "aircraft_rings"
                        ]
                    ),

                    fontsize=(
                        config[
                            "aircraft_font"
                        ]
                    ),

                    occupied=occupied,

                    obstacles=(
                        aircraft_icon_boxes
                    ),
                )
            )


            if artist is None:


                print(
                    "Could not place aircraft:",
                    get_aircraft_callsign(
                        row
                    ),
                )


                placement_failed = True

                break


            final_aircraft_artists.append(
                artist
            )


            occupied.append(
                box
            )


        if placement_failed:

            continue


        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        (
            intersections,
            outside,
            over_aircraft,

        ) = validate_layout(

            ax,

            fig,

            final_city_artists,

            final_aircraft_artists,

            aircraft_icon_boxes,
        )


        print(
            "Label intersections:",
            len(
                intersections
            ),
        )


        print(
            "Labels outside BBOX:",
            len(
                outside
            ),
        )


        print(
            "Aircraft labels over aircraft:",
            len(
                over_aircraft
            ),
        )


        if (
            len(intersections) == 0

            and

            len(outside) == 0

            and

            len(over_aircraft) == 0
        ):


            layout_valid = True


            print(
                "Layout valid."
            )


            break


else:


    layout_valid = True


# ============================================================
# ABORT INVALID LABEL LAYOUT
# ============================================================

if not layout_valid:


    raise RuntimeError(

        "Could not generate a "
        "zero-intersection label layout. "
        "Image not saved."
    )


# ============================================================
# FINAL SAFETY SUMMARY
# ============================================================

normal_aircraft = sum(

    state == STATE_NORMAL

    for state
    in aircraft_states.values()
)


warning_aircraft = sum(

    state == STATE_WARNING

    for state
    in aircraft_states.values()
)


conflict_aircraft = sum(

    state == STATE_CONFLICT

    for state
    in aircraft_states.values()
)


print()

print("=" * 80)

print(
    "AIRCRAFT SAFETY STATES"
)

print("=" * 80)


print(

    f"GREEN  / normal             : "
    f"{normal_aircraft}"
)


print(

    f"YELLOW / proximity warning  : "
    f"{warning_aircraft}"
)


print(

    f"RED    / loss of separation : "
    f"{conflict_aircraft}"
)


print("=" * 80)


# ============================================================
# SAVE
# ============================================================

plt.savefig(

    MAP_FILE,

    dpi=FIG_DPI,

    facecolor="white",

    bbox_inches="tight",

    pad_inches=0.15,
)


print()

print(
    "Map saved to:",
    MAP_FILE,
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()
