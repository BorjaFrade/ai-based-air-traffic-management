#!/usr/bin/env python3
"""
Download and filter official ENAIRE AIP ENR 4.4 significant points
for configured study areas.

The source is the official ENAIRE AIP ENR 4.4 page:
    NAME-CODE DESIGNATORS FOR SIGNIFICANT POINTS

These are AIXM 5 DesignatedPoint records with type = ICAO.

Output remains compatible with the project's existing waypoints.json reader.

Typical usage
-------------
Update only LECMSAN, preserving other existing entries:

    python3 download_waypoints.py --areas LECMSAN

Update several areas:

    python3 download_waypoints.py --areas LECMSAN GALICIA SCQ LCG VGO

Rebuild every configured area from the current ENAIRE AIP:

    python3 download_waypoints.py --all-areas

Use a previously downloaded/historical ENR 4.4 HTML file:

    python3 download_waypoints.py \
        --areas LECMSAN \
        --source-html /path/to/historical_ENR_4_4_en.html

IMPORTANT
---------
The default URL points to the CURRENT ENAIRE AIP. If the traffic study uses
2019 OpenSky data, the current waypoint network must not be described as the
2019 network. Use --source-html or --source-url with an appropriate historical
AIRAC/AIP source when historical consistency is required.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import requests

from area_presets import (
    load_area_presets,
    point_is_inside_area,
    resolve_area_preset,
)


# =============================================================================
# PATHS / SOURCE
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

BBOX_PRESETS_FILE = SCRIPT_DIR / "bbox_presets.json"
WAYPOINTS_FILE = SCRIPT_DIR / "waypoints.json"

SOURCE_URL = (
    "https://aip.enaire.es/AIP/contenido_AIP/ENR/"
    "LE_ENR_4_4_en.html"
)

SOURCE_NAME = "ENAIRE AIP ENR 4.4"
SOURCE_SECTION = "ENR 4.4 - Name-code designators for significant points"

DEFAULT_TIMEOUT_SECONDS = 60


# =============================================================================
# HTML TABLE PARSER (STANDARD LIBRARY ONLY)
# =============================================================================

class _TableParser(HTMLParser):
    """
    Minimal HTML table parser.

    It intentionally avoids BeautifulSoup/lxml so the script adds no new
    dependency beyond requests and the project's existing dependencies.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()

        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []

        elif tag == "tr" and self._table_depth == 1:
            self._current_row = []

        elif tag in {"td", "th"} and self._table_depth == 1:
            self._current_cell_parts = []

        elif tag == "br" and self._current_cell_parts is not None:
            self._current_cell_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._current_cell_parts is not None:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in {"td", "th"} and self._table_depth == 1:
            if (
                self._current_cell_parts is not None
                and self._current_row is not None
            ):
                value = "".join(self._current_cell_parts)
                value = html.unescape(value)
                value = re.sub(r"[ \t\r\f\v]+", " ", value)
                value = re.sub(r"\s*\n\s*", "\n", value)
                self._current_row.append(value.strip())

            self._current_cell_parts = None

        elif tag == "tr" and self._table_depth == 1:
            if (
                self._current_table is not None
                and self._current_row
            ):
                self._current_table.append(self._current_row)
            self._current_row = None

        elif tag == "table":
            if self._table_depth == 1 and self._current_table is not None:
                self.tables.append(self._current_table)
                self._current_table = None

            self._table_depth = max(0, self._table_depth - 1)


# =============================================================================
# COORDINATE / FIELD PARSING
# =============================================================================

LAT_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([NS])(?![A-Z])")
LON_RE = re.compile(r"(?<!\d)(\d{3})(\d{2})(\d{2}(?:\.\d+)?)([EW])(?![A-Z])")
ICAO_NAME_RE = re.compile(r"^[A-Z]{5}$")

WEF_RE = re.compile(
    r"\bWEF\s+(\d{2})[-\s]([A-Z]{3})[-\s](\d{2,4})\b",
    re.IGNORECASE,
)

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""

    value = html.unescape(str(value))
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def _dms_to_decimal(
    degrees: str,
    minutes: str,
    seconds: str,
    hemisphere: str,
) -> float:
    value = (
        float(degrees)
        + float(minutes) / 60.0
        + float(seconds) / 3600.0
    )

    if hemisphere.upper() in {"S", "W"}:
        value *= -1.0

    return value


def parse_coordinate_cell(value: str) -> tuple[float, float] | None:
    """
    Parse ENAIRE coordinate fields such as:

        442048N
        0084852W

    or the same values separated by spaces/HTML line breaks.
    """
    value = _clean_text(value).upper()

    lat_match = LAT_RE.search(value)
    lon_match = LON_RE.search(value)

    if not lat_match or not lon_match:
        return None

    latitude = _dms_to_decimal(*lat_match.groups())
    longitude = _dms_to_decimal(*lon_match.groups())

    return longitude, latitude


def parse_ats_routes(value: str) -> list[str]:
    """
    Preserve only route designators explicitly present in the AIP field.

    Examples:
        "M871" -> ["M871"]
        "N869, N975" -> ["N869", "N975"]
    """
    value = _clean_text(value).upper()

    if not value:
        return []

    candidates = re.findall(
        r"\b[A-Z]{1,2}\d{1,4}[A-Z]?\b",
        value,
    )

    return sorted(set(candidates))


def parse_effective_date(source_text: str) -> str | None:
    match = WEF_RE.search(source_text.upper())

    if not match:
        return None

    day = int(match.group(1))
    month_text = match.group(2).upper()
    year = int(match.group(3))

    if year < 100:
        year += 2000

    month = MONTHS.get(month_text)
    if month is None:
        return None

    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return None


# =============================================================================
# SOURCE RETRIEVAL
# =============================================================================

def download_source_html(
    source_url: str,
    timeout_seconds: int,
) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; OpenSky-TFM-waypoint-downloader/1.0)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    response = requests.get(
        source_url,
        timeout=timeout_seconds,
        headers=headers,
    )

    response.raise_for_status()

    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"

    return response.text


def read_source_html(
    source_html: Path | None,
    source_url: str,
    timeout_seconds: int,
) -> tuple[str, dict]:
    if source_html is not None:
        source_html = source_html.expanduser().resolve()

        if not source_html.exists():
            raise FileNotFoundError(
                f"Source HTML does not exist: {source_html}"
            )

        text = source_html.read_text(
            encoding="utf-8",
            errors="replace",
        )

        source = {
            "name": SOURCE_NAME,
            "section": SOURCE_SECTION,
            "origin": "local_html",
            "local_file": str(source_html),
        }

        return text, source

    text = download_source_html(
        source_url,
        timeout_seconds,
    )

    source = {
        "name": SOURCE_NAME,
        "section": SOURCE_SECTION,
        "origin": "web",
        "url": source_url,
    }

    return text, source


# =============================================================================
# ENAIRE ENR 4.4 PARSING
# =============================================================================

HEADER_NAME_ALIASES = {
    "NAME-CODE DESIGNATOR",
    "DESIGNADOR",
}

HEADER_COORD_ALIASES = {
    "COORDINATES",
    "COORDENADAS",
}

HEADER_FRA_ALIASES = {
    "FRA RELEVANCE",
    "RELEVANCIA FRA",
}

HEADER_ROUTE_ALIASES = {
    "ATS ROUTE",
    "ATS ROUTES",
    "RUTA ATS",
    "RUTAS ATS",
}

HEADER_REMARKS_ALIASES = {
    "REMARKS",
    "OBSERVACIONES",
}


def _normalise_header(value: str) -> str:
    value = _clean_text(value).upper()
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _find_column(
    header: list[str],
    aliases: set[str],
) -> int | None:
    normalised = [
        _normalise_header(value)
        for value in header
    ]

    for index, value in enumerate(normalised):
        if value in aliases:
            return index

    return None


def _find_enr44_table(
    tables: list[list[list[str]]],
) -> tuple[list[list[str]], dict[str, int]]:
    """
    Find the ICAO ENR 4.4 table by header names rather than table position.
    """
    for table in tables:
        if not table:
            continue

        # Usually the first row is the header, but scan the first few rows
        # defensively in case ENAIRE inserts an extra banner row.
        for header_index, header in enumerate(table[:5]):
            name_col = _find_column(header, HEADER_NAME_ALIASES)
            coord_col = _find_column(header, HEADER_COORD_ALIASES)

            if name_col is None or coord_col is None:
                continue

            columns = {
                "name": name_col,
                "coords": coord_col,
            }

            fra_col = _find_column(header, HEADER_FRA_ALIASES)
            route_col = _find_column(header, HEADER_ROUTE_ALIASES)
            remarks_col = _find_column(header, HEADER_REMARKS_ALIASES)

            if fra_col is not None:
                columns["fra"] = fra_col

            if route_col is not None:
                columns["routes"] = route_col

            if remarks_col is not None:
                columns["remarks"] = remarks_col

            return table[header_index + 1 :], columns

    raise RuntimeError(
        "Could not find the ENAIRE ENR 4.4 significant-points table. "
        "The AIP HTML layout may have changed."
    )


def parse_enaire_significant_points(source_html: str) -> list[dict]:
    parser = _TableParser()
    parser.feed(source_html)
    parser.close()

    if not parser.tables:
        raise RuntimeError(
            "No HTML tables were found in the ENAIRE source."
        )

    rows, columns = _find_enr44_table(parser.tables)

    points: dict[tuple[str, float, float], dict] = {}

    for row in rows:
        max_required_index = max(columns.values())

        if len(row) <= max_required_index:
            continue

        name = _clean_text(row[columns["name"]]).upper()

        # ENR 4.4's ICAO table contains five-letter name-code designators.
        # This also prevents accidental ingestion of headings/other tables.
        if not ICAO_NAME_RE.fullmatch(name):
            continue

        coords = parse_coordinate_cell(
            row[columns["coords"]]
        )

        if coords is None:
            continue

        longitude, latitude = coords

        fra_relevance = (
            _clean_text(row[columns["fra"]])
            if "fra" in columns
            else ""
        )

        ats_routes = (
            parse_ats_routes(row[columns["routes"]])
            if "routes" in columns
            else []
        )

        remarks = (
            _clean_text(row[columns["remarks"]])
            if "remarks" in columns
            else ""
        )

        point = {
            "name": name,
            "type": "ICAO",
            "lon": round(float(longitude), 8),
            "lat": round(float(latitude), 8),
            "fra_relevance": fra_relevance,
            "ats_routes": ats_routes,
        }

        if remarks:
            point["remarks"] = remarks

        key = (
            point["name"],
            point["lat"],
            point["lon"],
        )

        points[key] = point

    output = sorted(
        points.values(),
        key=lambda item: (
            item["name"],
            item["lat"],
            item["lon"],
        ),
    )

    if not output:
        raise RuntimeError(
            "The ENR 4.4 table was found, but no ICAO significant points "
            "could be parsed."
        )

    return output


# =============================================================================
# AREA FILTERING
# =============================================================================

def resolve_study_area(
    area_key: str,
    preset: dict,
) -> tuple[dict, float, float]:
    resolved = resolve_area_preset(
        area_key,
        {area_key: preset},
    )

    bbox = resolved["bbox"]

    center_lat = (
        float(bbox["lamin"])
        + float(bbox["lamax"])
    ) / 2.0

    center_lon = (
        float(bbox["lomin"])
        + float(bbox["lomax"])
    ) / 2.0

    return resolved, center_lat, center_lon


def filter_waypoints(
    source_waypoints: Iterable[dict],
    area: dict,
) -> list[dict]:
    filtered: dict[tuple[str, float, float], dict] = {}

    for waypoint in source_waypoints:
        longitude = float(waypoint["lon"])
        latitude = float(waypoint["lat"])

        if not point_is_inside_area(
            latitude,
            longitude,
            area,
        ):
            continue

        item = dict(waypoint)

        key = (
            str(item["name"]),
            float(item["lat"]),
            float(item["lon"]),
        )

        filtered[key] = item

    return sorted(
        filtered.values(),
        key=lambda item: (
            item["name"],
            item["lat"],
            item["lon"],
        ),
    )


def build_area_record(
    area_key: str,
    preset: dict,
    source_waypoints: list[dict],
    source_metadata: dict,
) -> dict:
    area, center_lat, center_lon = resolve_study_area(
        area_key,
        preset,
    )

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
        "bbox": area["bbox"],
        "polygon": area.get("polygon"),
        "center": {
            "lat": center_lat,
            "lon": center_lon,
        },
        "source": dict(source_metadata),
        "filter": {
            "types": ["ICAO"],
            "coordinates": "longitude, latitude",
            "description": (
                "Official ENAIRE AIP ENR 4.4 ICAO significant points "
                "inside the resolved study area; polygon presets are "
                "clipped exactly using area_presets.point_is_inside_area."
            ),
        },
        "count": len(waypoints),
        "waypoints": waypoints,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def load_existing_output(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Existing output is not valid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"Existing output must contain a JSON object: {path}"
        )

    return value


def write_waypoints(
    areas: dict,
    source_waypoints: list[dict],
    source_metadata: dict,
    output_file: Path,
    *,
    preserve_existing: bool,
) -> dict:
    output_file = output_file.expanduser().resolve()

    if preserve_existing:
        output = load_existing_output(
            output_file
        )
    else:
        output = {}

    for area_key, preset in areas.items():
        output[area_key] = build_area_record(
            area_key,
            preset,
            source_waypoints,
            source_metadata,
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    return output


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official ENAIRE AIP ENR 4.4 ICAO significant points "
            "and filter them to project study areas."
        )
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--areas",
        nargs="+",
        help=(
            "Update only the listed bbox/polygon preset keys, preserving "
            "other existing waypoints.json entries. Example: "
            "--areas LECMSAN GALICIA"
        ),
    )

    group.add_argument(
        "--all-areas",
        action="store_true",
        help=(
            "Rebuild all configured study-area entries from the selected "
            "ENAIRE source."
        ),
    )

    parser.add_argument(
        "--bbox-presets",
        type=Path,
        default=BBOX_PRESETS_FILE,
        help=(
            "bbox_presets.json path "
            f"(default: {BBOX_PRESETS_FILE})"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=WAYPOINTS_FILE,
        help=(
            "Output waypoints.json path "
            f"(default: {WAYPOINTS_FILE})"
        ),
    )

    parser.add_argument(
        "--source-url",
        default=SOURCE_URL,
        help=(
            "ENAIRE ENR 4.4 HTML URL. "
            "Can be replaced by a historical amendment URL."
        ),
    )

    parser.add_argument(
        "--source-html",
        type=Path,
        help=(
            "Read ENR 4.4 from a local HTML file instead of downloading it. "
            "Useful for archived/historical AIRAC material."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "HTTP timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS})"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.timeout <= 0:
        raise ValueError(
            "--timeout must be positive"
        )

    presets_path = arguments.bbox_presets.expanduser().resolve()

    all_areas = load_area_presets(
        presets_path
    )

    if not isinstance(all_areas, dict):
        raise ValueError(
            "bbox_presets.json must contain a JSON object"
        )

    if arguments.areas:
        unknown = (
            set(arguments.areas)
            - set(all_areas)
        )

        if unknown:
            raise ValueError(
                "Unknown study-area preset(s): "
                + ", ".join(
                    sorted(unknown)
                )
            )

        areas = {
            key: all_areas[key]
            for key in arguments.areas
        }

        preserve_existing = True

    else:
        # No --areas means rebuild every configured area. --all-areas is
        # accepted mainly to make that intention explicit on the command line.
        areas = all_areas
        preserve_existing = False

    source_text, source_metadata = read_source_html(
        arguments.source_html,
        arguments.source_url,
        arguments.timeout,
    )

    effective_date = parse_effective_date(
        source_text
    )

    source_metadata.update(
        {
            "authority": "ENAIRE / AIS-ESPAÑA",
            "aip_section": "ENR 4.4",
            "aixm_class": "DesignatedPoint",
            "aixm_type": "ICAO",
            "effective_date": effective_date,
            "downloaded_utc": datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat(),
        }
    )

    source_waypoints = parse_enaire_significant_points(
        source_text
    )

    output = write_waypoints(
        areas,
        source_waypoints,
        source_metadata,
        arguments.output,
        preserve_existing=preserve_existing,
    )

    print("=" * 78)
    print("ENAIRE AIP ENR 4.4 waypoint update")
    print("=" * 78)
    print(
        f"Official significant points parsed : "
        f"{len(source_waypoints):,}"
    )
    print(
        f"Source effective date              : "
        f"{effective_date or 'not detected'}"
    )
    print(
        f"Updated area entries               : "
        f"{len(areas)}"
    )

    for area_key in areas:
        count = int(
            output[area_key].get(
                "count",
                0,
            )
        )
        print(
            f"  {area_key:<24} {count:>6,} points"
        )

    print(
        f"Output                            : "
        f"{arguments.output.expanduser().resolve()}"
    )

    if effective_date:
        print()
        print(
            "NOTE: the downloaded network is valid for the ENAIRE AIP "
            f"effective date {effective_date}. It is not automatically a "
            "historical 2019 network."
        )


if __name__ == "__main__":
    main()
