from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D, Bbox
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
import matplotlib.patheffects as pe

import contextily as cx
from pyproj import Geod
import geonamescache


GEOD = Geod(ellps="WGS84")
BBOX = {"lamin": 0.0, "lamax": 1.0, "lomin": 0.0, "lomax": 1.0}
FIG_DPI = 160
FIG_SIZE = (28, 16)
MAP_FILE = Path("opensky_snapshot_map.png")
AREA = {"name": "OpenSky"}
AIRCRAFT_MARKER = None



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



def render_map(
    aircraft, area, bbox, aircraft_states, warnings, conflicts,
    snapshot_time, map_file, config=None,
    show=True,
):
    """Render an analysed aircraft snapshot and return a Matplotlib figure."""
    global BBOX, AREA, MAP_FILE
    if config:
        globals().update(config)
    BBOX = bbox
    AREA = area
    MAP_FILE = map_file
    AXIS_STEP = get_axis_step(BBOX)

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
    
        figsize=FIG_SIZE,
    
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
    
    if show:
        plt.show()

    return fig, ax
    
