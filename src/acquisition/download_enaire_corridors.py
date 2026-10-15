#!/usr/bin/env python3
"""
Download/compile ENAIRE AIP ENR 3.2 ATS corridors for one study area.

Default:
    python3 download_enaire_corridors.py --area LECMSAN

Default source:
    Current ENAIRE AIP ENR 3.2.

Historical studies:
    The default URL is NOT historical. For January 2019, use a historical
    ENR 3.2 HTML source if available:

        --source-url <historical_ENR_3_2_URL>

    or:

        --source-html /path/to/historical_ENR_3_2.html
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pyproj import Geod


SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "acquisition" and SCRIPT_DIR.parent.name == "src":
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
elif SCRIPT_DIR.name == "acquisition":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

BBOX_PRESETS_FILE = SCRIPT_DIR / "bbox_presets.json"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "corridors.json"

DEFAULT_SOURCE_URL = (
    "https://aip.enaire.es/AIP/contenido_AIP/ENR/"
    "LE_ENR_3_2_en.html"
)

DEFAULT_AREA = "LECMSAN"

GEOD = Geod(ellps="WGS84")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from area_presets import (  # noqa: E402
    load_area_presets,
    point_is_inside_area,
    resolve_area_preset,
)


ROUTE_RE = re.compile(r"^[A-Z][0-9]{1,3}$")
COORD_RE = re.compile(
    r"(?P<lat>\d{6}(?:\.\d+)?[NS])\s+"
    r"(?P<lon>\d{7}(?:\.\d+)?[EW])"
)

SKIP_LABELS = {"▲", "∆", "△", "▼", "---"}


def parse_lat(value: str) -> float:
    value = value.strip().upper()
    h = value[-1]
    s = value[:-1]
    deg = int(s[:2])
    minute = int(s[2:4])
    sec = float(s[4:])
    out = deg + minute / 60.0 + sec / 3600.0
    return -out if h == "S" else out


def parse_lon(value: str) -> float:
    value = value.strip().upper()
    h = value[-1]
    s = value[:-1]
    deg = int(s[:3])
    minute = int(s[3:5])
    sec = float(s[5:])
    out = deg + minute / 60.0 + sec / 3600.0
    return -out if h == "W" else out


def compact_name(label: str) -> str:
    label = " ".join(str(label).split()).strip()

    codes = re.findall(
        r"\(([A-Z0-9]{2,5})\)",
        label,
    )

    if codes:
        return codes[-1]

    if not label:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        label.split()[0].upper(),
    )


def fetch_html(
    url: str,
    timeout_s: float,
) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(OpenSky academic ATS corridor downloader)"
            )
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    response.encoding = (
        response.apparent_encoding
        or response.encoding
        or "utf-8"
    )
    return response.text


def visible_lines(
    html: str,
) -> list[str]:
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    lines = []

    for raw in soup.get_text(
        "\n",
        strip=True,
    ).splitlines():
        line = " ".join(
            raw.split()
        ).strip()

        if line:
            lines.append(line)

    return lines


def previous_label(
    lines: list[str],
    index: int,
) -> str:
    for j in range(
        index - 1,
        max(-1, index - 8),
        -1,
    ):
        candidate = lines[j].strip()

        if not candidate:
            continue
        if candidate in SKIP_LABELS:
            continue
        if ROUTE_RE.fullmatch(candidate):
            continue
        if candidate.startswith("RNAV"):
            continue
        if candidate.startswith("FL"):
            continue

        return candidate

    return ""


def parse_enr32_routes(
    html: str,
) -> dict[str, list[dict]]:
    """
    Parse route blocks from the visible ENR 3.2 HTML text.

    A block is accepted only if at least two geographic points are found,
    which filters route-like tokens appearing inside explanatory notes.
    """
    lines = visible_lines(html)

    route_starts = [
        (index, line)
        for index, line in enumerate(lines)
        if ROUTE_RE.fullmatch(line)
    ]

    routes = {}

    for route_idx, (start, route_name) in enumerate(route_starts):
        end = (
            route_starts[route_idx + 1][0]
            if route_idx + 1 < len(route_starts)
            else len(lines)
        )

        points = []

        for index in range(start + 1, end):
            match = COORD_RE.search(lines[index])

            if not match:
                continue

            label = previous_label(
                lines,
                index,
            )

            if not label:
                continue

            try:
                point = {
                    "name": compact_name(label),
                    "label": label,
                    "lat": parse_lat(match.group("lat")),
                    "lon": parse_lon(match.group("lon")),
                }
            except Exception:
                continue

            if points:
                prev = points[-1]
                if (
                    abs(prev["lat"] - point["lat"]) < 1e-10
                    and
                    abs(prev["lon"] - point["lon"]) < 1e-10
                ):
                    continue

            points.append(point)

        if len(points) >= 2:
            routes[route_name] = points

    return routes


def sample_segment(
    a: dict,
    b: dict,
    step_nm: float,
):
    lon1, lat1 = float(a["lon"]), float(a["lat"])
    lon2, lat2 = float(b["lon"]), float(b["lat"])

    _, _, distance_m = GEOD.inv(
        lon1,
        lat1,
        lon2,
        lat2,
    )

    intervals = max(
        1,
        int(
            math.ceil(
                float(distance_m)
                / (max(0.5, step_nm) * 1852.0)
            )
        ),
    )

    points = [(lon1, lat1)]

    if intervals > 1:
        points.extend(
            (float(lon), float(lat))
            for lon, lat in GEOD.npts(
                lon1,
                lat1,
                lon2,
                lat2,
                intervals - 1,
            )
        )

    points.append((lon2, lat2))
    return points


def segment_intersects_area(
    a: dict,
    b: dict,
    area: dict,
    step_nm: float,
) -> bool:
    return any(
        point_is_inside_area(
            latitude=lat,
            longitude=lon,
            area=area,
        )
        for lon, lat in sample_segment(
            a,
            b,
            step_nm,
        )
    )


def route_parts_for_area(
    route_name: str,
    points: list[dict],
    area: dict,
    step_nm: float,
) -> list[dict]:
    """
    Return contiguous pieces whose route segments intersect the study area.
    Adjacent published route points are preserved so the line reaches the
    area boundary continuously.
    """
    parts = []
    current = []

    for a, b in zip(
        points[:-1],
        points[1:],
    ):
        if segment_intersects_area(
            a,
            b,
            area,
            step_nm,
        ):
            if not current:
                current = [a]

            if current[-1] is not b:
                current.append(b)

        elif current:
            if len(current) >= 2:
                parts.append(current)
            current = []

    if current and len(current) >= 2:
        parts.append(current)

    return [
        {
            "name": route_name,
            "route": route_name,
            "type": "ATS",
            "segment_index": index,
            "points": part,
        }
        for index, part in enumerate(
            parts,
            start=1,
        )
    ]


def load_existing(
    path: Path,
) -> dict:
    if not path.exists():
        return {}

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def save_output(
    path: Path,
    area_key: str,
    area: dict,
    corridors: list[dict],
    source_url: str | None,
    source_html: Path | None,
):
    payload = load_existing(path)

    payload["_meta"] = {
        "schema": "opensky-corridors-v1",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "important": (
            "Default ENAIRE URL is the current AIP. "
            "Use a historical source for historical traffic studies."
        ),
    }

    source = (
        {
            "name": "Local ENR 3.2 HTML",
            "path": str(source_html),
            "historical": True,
        }
        if source_html is not None
        else
        {
            "name": "ENAIRE AIP ENR 3.2",
            "url": source_url,
            "historical": False,
        }
    )

    payload[area_key] = {
        "name": area.get("name", area_key),
        "source": source,
        "filter": {
            "area_key": area_key,
            "bbox": area["bbox"],
            "polygon": bool(area.get("polygon")),
            "description": (
                "ATS route portions with at least one sampled route "
                "segment intersecting the resolved study area."
            ),
        },
        "count": len(corridors),
        "corridors": corridors,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--area",
        default=DEFAULT_AREA,
        help="Area key from bbox_presets.json.",
    )

    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
    )

    parser.add_argument(
        "--source-html",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--bbox-presets",
        type=Path,
        default=BBOX_PRESETS_FILE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    parser.add_argument(
        "--area-sample-step-nm",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    presets = load_area_presets(
        args.bbox_presets
    )

    area = resolve_area_preset(
        args.area,
        presets,
    )

    if args.source_html is not None:
        source_html = (
            args.source_html
            .expanduser()
            .resolve()
        )

        if not source_html.exists():
            raise FileNotFoundError(
                f"Source HTML does not exist:\n{source_html}"
            )

        html = source_html.read_text(
            encoding="utf-8",
            errors="replace",
        )

        source_url = None

    else:
        source_html = None
        source_url = args.source_url

        print("Downloading:")
        print(source_url)

        html = fetch_html(
            source_url,
            args.timeout,
        )

    routes = parse_enr32_routes(
        html
    )

    print(
        f"Parsed ATS routes: "
        f"{len(routes):,}"
    )

    corridors = []

    for route_name, points in sorted(
        routes.items()
    ):
        corridors.extend(
            route_parts_for_area(
                route_name,
                points,
                area,
                args.area_sample_step_nm,
            )
        )

    output = (
        args.output
        .expanduser()
        .resolve()
    )

    save_output(
        output,
        args.area,
        area,
        corridors,
        source_url,
        source_html,
    )

    print(
        f"Corridor parts in {args.area}: "
        f"{len(corridors):,}"
    )
    print("Saved:")
    print(output)

    if corridors:
        print("Routes:")
        for route in sorted(
            {
                corridor["route"]
                for corridor in corridors
            }
        ):
            print(f"  {route}")


if __name__ == "__main__":
    main()
