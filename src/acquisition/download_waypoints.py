#!/usr/bin/env python3

"""Download and filter aviation waypoints for configured study areas."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from pyproj import Geod

from area_presets import (
    load_area_presets,
    resolve_area_preset,
    point_is_inside_area,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BBOX_PRESETS_FILE = Path(__file__).with_name(
    "bbox_presets.json"
)

WAYPOINTS_FILE = Path(__file__).with_name(
    "waypoints.json"
)

SOURCE_URL = (
    "https://raw.githubusercontent.com/euvl/aviation-waypoints/"
    "master/waypoints.json"
)

SOURCE_REPOSITORY = (
    "https://github.com/euvl/aviation-waypoints"
)

WAYPOINT_TYPES = {"WYP"}

GEOD = Geod(ellps="WGS84")


def load_json(path):

    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def bbox_from_center(
    latitude,
    longitude,
    radius_km,
):

    radius_m = float(radius_km) * 1000.0

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


def normalize_bbox(bbox):

    return {
        key: float(bbox[key])
        for key in (
            "lamin",
            "lamax",
            "lomin",
            "lomax",
        )
    }


def resolve_study_area(area_key, preset):
    resolved = resolve_area_preset(
        area_key,
        {area_key: preset},
    )
    bbox = resolved["bbox"]
    center_lat = (bbox["lamin"] + bbox["lamax"]) / 2.0
    center_lon = (bbox["lomin"] + bbox["lomax"]) / 2.0
    return resolved, center_lat, center_lon


def download_source_waypoints():

    response = requests.get(
        SOURCE_URL,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError(
            "Waypoint source must contain a JSON array"
        )

    return payload


def filter_waypoints(
    source_waypoints,
    area,
):

    filtered = {}

    for waypoint in source_waypoints:

        if waypoint.get("type") not in WAYPOINT_TYPES:
            continue

        coords = waypoint.get("coords", [])

        if len(coords) < 2:
            continue

        longitude = float(coords[0])
        latitude = float(coords[1])

        if not point_is_inside_area(
            latitude,
            longitude,
            area,
        ):
            continue

        item = {
            "name": waypoint["name"],
            "type": waypoint["type"],
            "lon": longitude,
            "lat": latitude,
        }

        key = (
            item["type"],
            item["name"],
            item["lon"],
            item["lat"],
        )

        filtered[key] = item

    return sorted(
        filtered.values(),
        key=lambda item: item["name"],
    )


def build_area_record(
    area_key,
    preset,
    source_waypoints,
):

    area, center_lat, center_lon = resolve_study_area(
        area_key,
        preset,
    )
    bbox = area["bbox"]

    waypoints = filter_waypoints(
        source_waypoints,
        area,
    )

    return {
        "name": preset.get(
            "name",
            area_key,
        ),
        "type": preset.get("type"),
        "bbox": bbox,
        "polygon": area.get("polygon"),
        "center": {
            "lat": center_lat,
            "lon": center_lon,
        },
        "source": {
            "name": "euvl/aviation-waypoints",
            "repository": SOURCE_REPOSITORY,
            "raw_url": SOURCE_URL,
            "license": "MIT",
        },
        "filter": {
            "types": sorted(WAYPOINT_TYPES),
            "coordinates": "longitude, latitude",
            "description": "Points inside the resolved study area; polygon presets are clipped exactly after BBOX resolution",
        },
        "count": len(waypoints),
        "waypoints": waypoints,
    }


def write_waypoints(
    areas,
    source_waypoints,
    output_file,
):

    output = {}

    for area_key, preset in areas.items():
        output[area_key] = build_area_record(
            area_key,
            preset,
            source_waypoints,
        )

    output_file.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Download WYP waypoints for BBOX study-area presets."
        )
    )

    parser.add_argument(
        "--areas",
        nargs="+",
        help="Only update the listed BBOX preset keys",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=WAYPOINTS_FILE,
        help=f"Output JSON file (default: {WAYPOINTS_FILE})",
    )

    return parser.parse_args()


def main():

    arguments = parse_arguments()
    all_areas = load_area_presets(BBOX_PRESETS_FILE)

    if not isinstance(all_areas, dict):
        raise ValueError(
            "bbox_presets.json must contain a JSON object"
        )

    if arguments.areas:
        unknown = set(arguments.areas) - set(all_areas)

        if unknown:
            raise ValueError(
                "Unknown BBOX preset(s): "
                + ", ".join(sorted(unknown))
            )

        areas = {
            key: all_areas[key]
            for key in arguments.areas
        }
    else:
        areas = all_areas

    source_waypoints = download_source_waypoints()

    write_waypoints(
        areas,
        source_waypoints,
        arguments.output,
    )

    print(
        f"Updated {len(areas)} BBOX area(s) in "
        f"{arguments.output}"
    )


if __name__ == "__main__":
    main()
