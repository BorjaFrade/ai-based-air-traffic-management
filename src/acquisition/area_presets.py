from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Geod


GEOD = Geod(ellps="WGS84")


def load_area_presets(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        presets = json.load(f)
    if not isinstance(presets, dict):
        raise ValueError("Area presets must be a JSON object")

    # Keys beginning with "_" are documentation/templates and are ignored.
    # Presets can also be disabled explicitly with "enabled": false.
    return {
        key: value
        for key, value in presets.items()
        if not str(key).startswith("_")
        and not (
            isinstance(value, dict)
            and value.get("enabled") is False
        )
    }


def bbox_from_center(latitude: float, longitude: float, radius_km: float) -> dict:
    radius_m = float(radius_km) * 1000.0
    _, latitude_max, _ = GEOD.fwd(longitude, latitude, 0, radius_m)
    _, latitude_min, _ = GEOD.fwd(longitude, latitude, 180, radius_m)
    longitude_max, _, _ = GEOD.fwd(longitude, latitude, 90, radius_m)
    longitude_min, _, _ = GEOD.fwd(longitude, latitude, 270, radius_m)
    return {
        "lamin": float(latitude_min),
        "lamax": float(latitude_max),
        "lomin": float(longitude_min),
        "lomax": float(longitude_max),
        "extended": 1,
    }


def _normalise_polygon_vertices(preset: dict) -> list[tuple[float, float]]:
    """
    Return polygon vertices as [(lon, lat), ...].

    Accepted schema:
      "type": "polygon",
      "polygon": {
        "vertices": [
          {"lat": 43.0, "lon": -9.0},
          ...
        ]
      }

    For convenience, "polygon" may also directly contain the vertex list.
    """
    polygon = preset.get("polygon")
    if isinstance(polygon, dict):
        vertices = polygon.get("vertices")
    else:
        vertices = polygon

    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError(
            "Polygon preset must define at least three vertices under "
            '"polygon": {"vertices": [...]}'
        )

    resolved: list[tuple[float, float]] = []
    for item in vertices:
        if isinstance(item, dict):
            lon = float(item["lon"])
            lat = float(item["lat"])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Compact form is [lon, lat].
            lon = float(item[0])
            lat = float(item[1])
        else:
            raise ValueError(
                "Polygon vertices must be {'lat': ..., 'lon': ...} "
                "or [lon, lat]"
            )
        resolved.append((lon, lat))

    # Explicitly close the polygon for plotting/debugging.
    if resolved[0] != resolved[-1]:
        resolved.append(resolved[0])

    return resolved


def _bbox_from_polygon(vertices: Iterable[tuple[float, float]]) -> dict:
    points = list(vertices)
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return {
        "lamin": float(min(lats)),
        "lamax": float(max(lats)),
        "lomin": float(min(lons)),
        "lomax": float(max(lons)),
        "extended": 1,
    }


def resolve_area_preset(preset_key: str, presets: dict) -> dict:
    if preset_key not in presets:
        raise ValueError(f"Unknown area preset: {preset_key}")

    preset = presets[preset_key]
    if not isinstance(preset, dict):
        raise ValueError(f"Preset '{preset_key}' must be a JSON object")

    definition_type = preset.get("type")
    polygon_vertices = None

    if definition_type == "center":
        center = preset["center"]
        bbox = bbox_from_center(
            float(center["lat"]),
            float(center["lon"]),
            float(preset["radius_km"]),
        )

    elif definition_type == "vertices":
        bbox = {
            key: float(preset["vertices"][key])
            for key in ("lamin", "lamax", "lomin", "lomax")
        }
        bbox["extended"] = int(preset["vertices"].get("extended", 1))

    elif definition_type == "polygon":
        polygon_vertices = _normalise_polygon_vertices(preset)
        bbox = _bbox_from_polygon(polygon_vertices)

    else:
        raise ValueError(
            f"Preset '{preset_key}' must use type "
            '"center", "vertices" or "polygon"'
        )

    return {
        "key": preset_key,
        "name": preset.get("name", preset_key),
        "icao": preset.get("icao", ""),
        "type": definition_type,
        "bbox": bbox,
        "polygon": polygon_vertices,
        "render": dict(preset.get("render", {})),
    }


def _points_in_polygon(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    vertices: list[tuple[float, float]],
) -> np.ndarray:
    """
    Vectorised ray-casting point-in-polygon test.

    Boundary points are treated as inside within floating-point precision.
    """
    x = np.asarray(longitudes, dtype=float)
    y = np.asarray(latitudes, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    on_boundary = np.zeros(x.shape, dtype=bool)

    poly = vertices[:-1] if vertices and vertices[0] == vertices[-1] else vertices
    if len(poly) < 3:
        return inside

    # Boundary check + standard ray casting.
    eps = 1e-12
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]

        dx = xj - xi
        dy = yj - yi
        denom = dx * dx + dy * dy
        if denom > 0:
            t = ((x - xi) * dx + (y - yi) * dy) / denom
            t_clip = np.clip(t, 0.0, 1.0)
            px = xi + t_clip * dx
            py = yi + t_clip * dy
            on_edge = ((x - px) ** 2 + (y - py) ** 2) <= eps
            on_boundary |= on_edge

        intersects = ((yi > y) != (yj > y))
        x_cross = (xj - xi) * (y - yi) / ((yj - yi) + 1e-300) + xi
        inside ^= intersects & (x < x_cross)
        j = i

    return inside | on_boundary


def filter_dataframe_to_area(
    dataframe: pd.DataFrame,
    area: dict,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> pd.DataFrame:
    """
    Apply the exact local area filter.

    Rectangle/center presets need no extra filtering because their query BBOX
    already represents the study area. Polygon presets are first queried by
    their enclosing BBOX, then clipped locally to the polygon.
    """
    if dataframe.empty or not area.get("polygon"):
        return dataframe.copy()

    valid = (
        dataframe[latitude_column].notna()
        & dataframe[longitude_column].notna()
    )
    out = dataframe.loc[valid].copy()

    mask = _points_in_polygon(
        out[longitude_column].to_numpy(dtype=float),
        out[latitude_column].to_numpy(dtype=float),
        area["polygon"],
    )
    return out.loc[mask].copy()


def point_is_inside_area(
    latitude: float,
    longitude: float,
    area: dict,
) -> bool:
    if not area.get("polygon"):
        bbox = area["bbox"]
        return (
            bbox["lamin"] <= latitude <= bbox["lamax"]
            and bbox["lomin"] <= longitude <= bbox["lomax"]
        )
    return bool(
        _points_in_polygon(
            np.asarray([longitude], dtype=float),
            np.asarray([latitude], dtype=float),
            area["polygon"],
        )[0]
    )



def load_waypoints_for_area(
    path: Path,
    area_key: str | None = None,
    area: dict | None = None,
) -> list[dict]:
    """
    Load aviation waypoints from waypoints.json for one study area.

    Preferred lookup:
      1. Use the exact top-level area_key when it exists.
      2. Otherwise merge/deduplicate all waypoint records in the file and
         geometrically filter them to the supplied area.

    Returned records use:
        {"name": str, "type": str, "lon": float, "lat": float}
    """
    if not path.exists():
        print(f"WARNING: waypoint file not found: {path}")
        return []

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("waypoints.json must contain a JSON object")

    records = []

    if area_key and area_key in payload:
        entry = payload[area_key]
        if isinstance(entry, dict):
            records = entry.get("waypoints", []) or []

    if not records:
        # Fallback for old/stale waypoint files whose top-level keys do not
        # exactly match the current bbox_presets.json keys.
        merged = {}
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            for waypoint in entry.get("waypoints", []) or []:
                try:
                    name = str(waypoint["name"])
                    lon = float(waypoint["lon"])
                    lat = float(waypoint["lat"])
                except (KeyError, TypeError, ValueError):
                    continue

                key = (
                    str(waypoint.get("type", "WYP")),
                    name,
                    lon,
                    lat,
                )
                merged[key] = {
                    "name": name,
                    "type": str(waypoint.get("type", "WYP")),
                    "lon": lon,
                    "lat": lat,
                }

        records = list(merged.values())

    resolved = {}
    for waypoint in records:
        try:
            name = str(waypoint["name"])
            lon = float(waypoint["lon"])
            lat = float(waypoint["lat"])
        except (KeyError, TypeError, ValueError):
            continue

        if area is not None and not point_is_inside_area(
            latitude=lat,
            longitude=lon,
            area=area,
        ):
            continue

        key = (
            str(waypoint.get("type", "WYP")),
            name,
            lon,
            lat,
        )
        resolved[key] = {
            "name": name,
            "type": str(waypoint.get("type", "WYP")),
            "lon": lon,
            "lat": lat,
        }

    return sorted(
        resolved.values(),
        key=lambda item: item["name"],
    )

# ============================================================
# CORRIDORS / ATS ROUTES
# ============================================================

def _corridor_bbox_intersects_area(
    points: list[dict],
    area: dict,
) -> bool:
    """Conservative fallback intersection test for corridor polylines."""
    if len(points) < 2:
        return False

    bbox = area["bbox"]
    lons = [float(point["lon"]) for point in points]
    lats = [float(point["lat"]) for point in points]

    return not (
        max(lons) < bbox["lomin"]
        or min(lons) > bbox["lomax"]
        or max(lats) < bbox["lamin"]
        or min(lats) > bbox["lamax"]
    )


def load_corridors_for_area(
    path: Path,
    area_key: str | None = None,
    area: dict | None = None,
) -> list[dict]:
    """
    Load ATS corridors from corridors.json.

    Preferred lookup:
      1. Exact top-level area_key.
      2. Otherwise merge all corridor records and conservatively filter them
         against the requested area bounding box.

    Returned records use:
        {
            "name": str,
            "route": str,
            "type": str,
            "segment_index": int,
            "points": [
                {"name": str, "label": str, "lon": float, "lat": float},
                ...
            ]
        }
    """
    if not path.exists():
        print(f"WARNING: corridor file not found: {path}")
        return []

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("corridors.json must contain a JSON object")

    records = []

    if area_key and area_key in payload:
        entry = payload[area_key]
        if isinstance(entry, dict):
            records = entry.get("corridors", []) or []

    if not records:
        for key, entry in payload.items():
            if str(key).startswith("_") or not isinstance(entry, dict):
                continue
            records.extend(entry.get("corridors", []) or [])

    resolved = []
    seen = set()

    for corridor in records:
        if not isinstance(corridor, dict):
            continue

        route = str(
            corridor.get("route", corridor.get("name", ""))
        ).strip()

        if not route:
            continue

        points = []

        for raw_point in corridor.get("points", []) or []:
            try:
                lon = float(raw_point["lon"])
                lat = float(raw_point["lat"])
            except (KeyError, TypeError, ValueError):
                continue

            points.append({
                "name": str(raw_point.get("name", "")),
                "label": str(
                    raw_point.get(
                        "label",
                        raw_point.get("name", ""),
                    )
                ),
                "lon": lon,
                "lat": lat,
            })

        if len(points) < 2:
            continue

        if (
            area is not None
            and not _corridor_bbox_intersects_area(points, area)
        ):
            continue

        segment_index = int(corridor.get("segment_index", 1))
        dedup_key = (
            route,
            segment_index,
            tuple((p["lon"], p["lat"]) for p in points),
        )

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        resolved.append({
            "name": str(corridor.get("name", route)),
            "route": route,
            "type": str(corridor.get("type", "ATS")),
            "segment_index": segment_index,
            "points": points,
        })

    return sorted(
        resolved,
        key=lambda item: (
            item["route"],
            item["segment_index"],
        ),
    )
