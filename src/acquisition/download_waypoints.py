#!/usr/bin/env python3
"""
Download and filter official AIP ENR 4.4 significant points from ENAIRE
or the Swedish LFV eAIP for configured study areas.

The source is the official ENAIRE AIP ENR 4.4 page:
    NAME-CODE DESIGNATORS FOR SIGNIFICANT POINTS

The output is a common, renderer-compatible waypoint catalogue. ENAIRE and
LFV publish the same ICAO five-letter significant-point concept in their
respective ENR 4.4 sections, although the surrounding HTML layouts differ.

Output remains compatible with the project's existing waypoints.json reader.

Typical usage
-------------
Update only LECMSAN, preserving other existing entries:

    python3 download_waypoints.py --areas LECMSAN

Update several areas:

    python3 download_waypoints.py --source enaire \
        --areas LECMSAN GALICIA SCQ LCG VGO

Update Spanish and Swedish areas in one run, preserving other entries:

    python3 download_waypoints.py --source auto \
        --areas GALICIA SCAT_ESMM_GOTHENBURG_ENROUTE \
        SCAT_ESMM_MALMO_ENROUTE SCAT_ESOS_STOCKHOLM_ENROUTE \
        SCAT_ESOS_CENTRAL_ENROUTE

Rebuild every configured area from the current ENAIRE and LFV sources:

    python3 download_waypoints.py --source auto --all-areas

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

The Swedish source is the official LFV eAIP published through AROWeb. The
default Swedish entry point discovers the current ENR 4.4 document from the
current eAIP issue. For historical SCAT work, use an archived LFV ENR 4.4
HTML document with --source-html or provide its direct URL with --source-url.
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
from urllib.parse import unquote, urljoin

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

SWEDEN_EAIP_INDEX_URL = (
    "https://www.aro.lfv.se/content/eaip/default_offline.html"
)

SOURCE_NAME = "ENAIRE AIP ENR 4.4"
SOURCE_SECTION = "ENR 4.4 - Name-code designators for significant points"
SWEDEN_SOURCE_NAME = "LFV eAIP Sweden ENR 4.4"
SWEDEN_SOURCE_SECTION = (
    "ENR 4.4 - Name-code designators for significant points"
)

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


class _AnchorParser(HTMLParser):
    """Collect anchor URLs and visible text from an eAIP index page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self._href = href
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return

        self.links.append(
            (
                self._href,
                _clean_text("".join(self._text_parts)),
            )
        )
        self._href = None
        self._text_parts = []


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

EFFECTIVE_RE = re.compile(
    r"\bEFFECTIVE\s+(\d{1,2})[-\s]([A-Z]{3,9})[-\s](\d{4})\b",
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
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
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
    upper_text = source_text.upper()
    matches = [
        *WEF_RE.finditer(upper_text),
        *EFFECTIVE_RE.finditer(upper_text),
    ]

    if not matches:
        return None

    dates: list[datetime] = []
    for match in matches:
        day = int(match.group(1))
        month_text = match.group(2).upper()
        year = int(match.group(3))

        if len(match.group(3)) == 2:
            # AIP pages still use two-digit years in WEF labels. Interpret
            # 00-49 as 2000-2049 and 50-99 as 1950-1999.
            year += 2000 if year <= 49 else 1900

        month = MONTHS.get(month_text)
        if month is None:
            continue

        try:
            dates.append(
                datetime(
                    year,
                    month,
                    day,
                )
            )
        except ValueError:
            continue

    if not dates:
        return None

    return max(dates).strftime("%Y-%m-%d")


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


def _normalise_url_text(value: str) -> str:
    return unquote(
        html.unescape(str(value))
    ).replace("_", " ").replace("-", " ").upper()


def _normalise_download_url(value: str) -> str:
    """Normalise LFV's Windows-style relative links for HTTP requests."""
    return str(value).replace("\\", "/").replace(" ", "%20")


def _looks_like_sweden_enr44_link(
    href: str,
    label: str,
) -> bool:
    value = _normalise_url_text(
        f"{href} {label}"
    )

    return bool(
        re.search(
            r"\bENR\s+4\s*\.\s*4\b",
            value,
        )
        or re.search(
            r"\b4\s*\.\s*4\b.*SIGNIFICANT",
            value,
        )
    )


def discover_sweden_issue_urls(
    index_html: str,
    index_url: str,
) -> list[str]:
    """Return current LFV eAIP issue indexes from the offline landing page."""
    parser = _AnchorParser()
    parser.feed(index_html)
    parser.close()

    candidates: list[tuple[int, str]] = []

    for href, label in parser.links:
        absolute = _normalise_download_url(
            urljoin(index_url, href)
        )
        lowered = absolute.lower()
        if "index-v2.html" not in lowered and not lowered.endswith(
            "index.html"
        ):
            continue

        score = 100
        if "current" in _normalise_url_text(label):
            score += 10
        candidates.append((score, absolute))

    # Preserve compatibility with index versions that expose the current
    # issue link in an inline script instead of an anchor.
    for match in re.finditer(
        r"https?://[^\"'\s<>]+/content/eaip/[^\"'\s<>]+/index(?:-v2)?\.html",
        index_html,
        flags=re.IGNORECASE,
    ):
        candidates.append(
            (
                90,
                _normalise_download_url(
                    html.unescape(match.group(0))
                ),
            )
        )

    deduplicated: dict[str, int] = {}
    for score, url in candidates:
        deduplicated[url] = max(score, deduplicated.get(url, -1))

    return [
        url
        for url, _score in sorted(
            deduplicated.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def discover_sweden_enr44_urls(
    index_html: str,
    index_url: str,
) -> list[str]:
    """Return likely LFV eAIP ENR 4.4 HTML documents in ranked order."""
    parser = _AnchorParser()
    parser.feed(index_html)
    parser.close()

    candidates: list[tuple[int, str]] = []

    for href, label in parser.links:
        if not _looks_like_sweden_enr44_link(href, label):
            continue

        absolute = _normalise_download_url(
            urljoin(index_url, href)
        )
        lowered = absolute.lower()

        # Prefer the HTML document over the corresponding PDF link.
        score = 100
        if lowered.endswith(".pdf"):
            score -= 80
        if "significant" in _normalise_url_text(label):
            score += 10
        candidates.append((score, absolute))

    # Some AROWeb/eAIP index versions expose the document links through
    # JavaScript rather than ordinary anchors. Keep a small fallback for the
    # standard LFV eAIP filename convention.
    issue_root = index_url.rsplit("/", 1)[0] + "/"
    for relative in (
        "eAIP/ES-ENR-4.4-en-GB.html",
        "eAIP/ES-ENR%204.4-en-GB.html",
        "eAIP/ES-ENR%204.4-en-GB.htm",
        "eAIP/ES-ENR_4_4_en-GB.html",
        "eAIP/ES_ENR_4_4_en.html",
        "ES-ENR-4.4-en-GB.html",
        "ES-ENR%204.4-en-GB.html",
    ):
        candidates.append((10, urljoin(issue_root, relative)))

    deduplicated: dict[str, int] = {}
    for score, url in candidates:
        deduplicated[url] = max(score, deduplicated.get(url, -1))

    return [
        url
        for url, _score in sorted(
            deduplicated.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def discover_sweden_embedded_table_urls(
    document_html: str,
    document_url: str,
) -> list[str]:
    """Find LFV eAIP iframe documents that contain the actual data table."""
    candidates: list[str] = []
    for match in re.finditer(
        r"<iframe\b[^>]+\bsrc\s*=\s*['\"]([^'\"]+)['\"]",
        document_html,
        flags=re.IGNORECASE,
    ):
        source = match.group(1).strip()
        if not source or source.startswith("#"):
            continue
        candidates.append(
            _normalise_download_url(
                urljoin(document_url, source)
            )
        )

    return list(dict.fromkeys(candidates))


def read_local_source_html(
    source_html: Path,
    source_name: str,
    source_section: str,
) -> tuple[str, dict]:
    source_html = source_html.expanduser().resolve()

    if not source_html.exists():
        raise FileNotFoundError(
            f"Source HTML does not exist: {source_html}"
        )

    text = source_html.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return text, {
        "name": source_name,
        "section": source_section,
        "origin": "local_html",
        "local_file": str(source_html),
    }


def read_sweden_source_html(
    source_html: Path | None,
    source_url: str,
    timeout_seconds: int,
) -> tuple[str, dict]:
    if source_html is not None:
        return read_local_source_html(
            source_html,
            SWEDEN_SOURCE_NAME,
            SWEDEN_SOURCE_SECTION,
        )

    # A direct ENR 4.4 URL can be supplied for an archived or pinned AIRAC
    # issue. Otherwise the current LFV eAIP index is resolved first.
    if _looks_like_sweden_enr44_link(source_url, source_url):
        text = download_source_html(
            source_url,
            timeout_seconds,
        )
        return text, {
            "name": SWEDEN_SOURCE_NAME,
            "section": SWEDEN_SOURCE_SECTION,
            "origin": "web",
            "url": source_url,
        }

    index_html = download_source_html(
        source_url,
        timeout_seconds,
    )

    pages: list[tuple[str, str]] = [(index_html, source_url)]
    issue_failures: list[str] = []

    for issue_url in discover_sweden_issue_urls(
        index_html,
        source_url,
    ):
        try:
            issue_html = download_source_html(
                issue_url,
                timeout_seconds,
            )
        except requests.RequestException as error:
            issue_failures.append(f"{issue_url}: {error}")
            continue

        pages.append((issue_html, issue_url))

    candidates: list[str] = []
    # The landing page also produces generic fallback filenames. Prefer the
    # issue-specific directory first so those candidates cannot shadow the
    # real current document.
    ordered_pages = pages[1:] + pages[:1]
    for page_html, page_url in ordered_pages:
        candidates.extend(
            discover_sweden_enr44_urls(
                page_html,
                page_url,
            )
        )

    candidates = list(dict.fromkeys(candidates))

    failures: list[str] = list(issue_failures)
    for candidate in candidates:
        if candidate.lower().endswith(".pdf"):
            continue

        try:
            text = download_source_html(
                candidate,
                timeout_seconds,
            )
        except requests.RequestException as error:
            failures.append(f"{candidate}: {error}")
            continue

        # LFV's ENR 4.4 page can wrap its data table in TChart.html. Follow
        # that iframe so the parser receives the actual rows rather than the
        # navigation shell.
        selected_text = text
        selected_url = candidate
        for embedded_url in discover_sweden_embedded_table_urls(
            text,
            candidate,
        ):
            try:
                embedded_text = download_source_html(
                    embedded_url,
                    timeout_seconds,
                )
            except requests.RequestException as error:
                failures.append(f"{embedded_url}: {error}")
                continue

            if (
                "<table" in embedded_text.lower()
                and re.search(
                    r"<tr\b[^>]+id\s*=\s*['\"][^'\"]*(?:ENR4\.4_H1|ENR44)",
                    embedded_text,
                    flags=re.IGNORECASE,
                )
            ):
                selected_text = embedded_text
                selected_url = embedded_url
                break

        # Do not accept an index page or an unrelated chapter as ENR 4.4.
        if (
            not _looks_like_sweden_enr44_link(candidate, text[:4000])
            or "<table" not in selected_text.lower()
        ):
            failures.append(
                f"{candidate}: document does not contain an ENR 4.4 table"
            )
            continue

        return selected_text, {
            "name": SWEDEN_SOURCE_NAME,
            "section": SWEDEN_SOURCE_SECTION,
            "origin": "web",
            "index_url": source_url,
            "url": selected_url,
            "document_url": candidate,
        }

    details = "\n".join(failures[-5:])
    raise RuntimeError(
        "Could not resolve the Swedish LFV eAIP ENR 4.4 document from "
        f"{source_url}. Provide a direct ENR 4.4 URL with --source-url "
        "or save the section HTML and use --source-html."
        + (f"\nTried:\n{details}" if details else "")
    )


def read_source_html(
    source: str,
    source_html: Path | None,
    source_url: str | None,
    timeout_seconds: int,
) -> tuple[str, dict]:
    if source == "sweden":
        return read_sweden_source_html(
            source_html,
            source_url or SWEDEN_EAIP_INDEX_URL,
            timeout_seconds,
        )

    if source_html is not None:
        return read_local_source_html(
            source_html,
            SOURCE_NAME,
            SOURCE_SECTION,
        )

    effective_url = source_url or SOURCE_URL
    text = download_source_html(
        effective_url,
        timeout_seconds,
    )

    return text, {
        "name": SOURCE_NAME,
        "section": SOURCE_SECTION,
        "origin": "web",
        "url": effective_url,
    }


# =============================================================================
# ENAIRE ENR 4.4 PARSING
# =============================================================================

HEADER_NAME_ALIASES = {
    "NAME-CODE DESIGNATOR",
    "NAME CODE DESIGNATOR",
    "DESIGNADOR",
}

HEADER_COORD_ALIASES = {
    "COORDINATES",
    "COORDINATE",
    "COORDENADAS",
    "KOORDINATER",
}

HEADER_FRA_ALIASES = {
    "FRA RELEVANCE",
    "RELEVANCIA FRA",
}

HEADER_ROUTE_ALIASES = {
    "ATS ROUTE",
    "ATS ROUTES",
    "ATS ROUTE OR FRA RELEVANCE",
    "ATS ROUTE OR FRA RELEVANCE",
    "RUTA ATS",
    "RUTAS ATS",
}

HEADER_REMARKS_ALIASES = {
    "REMARKS",
    "REMARK",
    "REMARK/USAGE",
    "REMARK USAGE",
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
        compact_value = re.sub(
            r"[^A-Z0-9]",
            "",
            value,
        )

        for alias in aliases:
            normalised_alias = _normalise_header(alias)
            compact_alias = re.sub(
                r"[^A-Z0-9]",
                "",
                normalised_alias,
            )

            if (
                value == normalised_alias
                or normalised_alias in value
                or compact_value == compact_alias
                or compact_alias in compact_value
            ):
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


def _parse_significant_point_rows(
    rows: list[list[str]],
    columns: dict[str, int],
) -> list[dict]:
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


def parse_enaire_significant_points(source_html: str) -> list[dict]:
    parser = _TableParser()
    parser.feed(source_html)
    parser.close()

    if not parser.tables:
        raise RuntimeError(
            "No HTML tables were found in the ENAIRE source."
        )

    rows, columns = _find_enr44_table(parser.tables)
    return _parse_significant_point_rows(rows, columns)


class _InlineTextParser(HTMLParser):
    """Extract readable cell text from LFV's inline eAIP XML fragment."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _inline_cell_text(value: str) -> str:
    parser = _InlineTextParser()
    parser.feed(value)
    parser.close()
    return _clean_text("".join(parser.parts))


def parse_sweden_significant_points(source_html: str) -> list[dict]:
    """Parse LFV eAIP's inline ENR 4.4 table when HTML is not well nested."""
    # Older LFV/AIP HTML and archived snapshots can contain a normal table;
    # reuse the common parser whenever the document is well formed.
    try:
        return parse_enaire_significant_points(source_html)
    except RuntimeError:
        pass

    row_matches = re.findall(
        r"<tr\b[^>]+id\s*=\s*['\"][^'\"]*(?:ENR4\.4_H1|ENR44)[^'\"]*['\"][^>]*>(.*?)</tr>",
        source_html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    tables: list[list[list[str]]] = []
    for row_html in row_matches:
        cells = re.findall(
            r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        tables.append([
            _inline_cell_text(cell)
            for cell in cells
        ])

    if not tables:
        raise RuntimeError(
            "No LFV ENR 4.4 rows were found in the Swedish eAIP source."
        )

    header = tables[0]
    name_col = _find_column(header, HEADER_NAME_ALIASES)
    coord_col = _find_column(header, HEADER_COORD_ALIASES)
    if name_col is None or coord_col is None:
        raise RuntimeError(
            "Could not find the LFV ENR 4.4 significant-points headers."
        )

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

    return _parse_significant_point_rows(
        tables[1:],
        columns,
    )


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
                "Official AIP ENR 4.4 ICAO significant points inside the "
                "resolved study area; polygon presets are clipped exactly "
                "using area_presets.point_is_inside_area."
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


def source_for_area(
    area_key: str,
    preset: dict,
) -> str:
    """Infer the authoritative AIP source for a configured area."""
    airspace = preset.get("airspace", {})
    if (
        isinstance(airspace, dict)
        and str(airspace.get("dataset", "")).upper() == "SCAT"
    ):
        return "sweden"

    if area_key.upper().startswith("SCAT_"):
        return "sweden"

    return "enaire"


def group_areas_by_source(
    areas: dict,
    selected_source: str,
) -> list[tuple[str, dict]]:
    """Return one or two source-specific area groups in stable order."""
    if selected_source in {"enaire", "sweden"}:
        mismatched = [
            key
            for key, preset in areas.items()
            if source_for_area(key, preset) != selected_source
        ]
        if mismatched:
            expected = "sweden" if selected_source == "enaire" else "enaire"
            raise ValueError(
                f"The selected source '{selected_source}' does not match "
                f"area preset(s): {', '.join(sorted(mismatched))}. "
                f"Use --source {expected} for those areas or --source auto."
            )
        return [(selected_source, areas)]

    grouped: dict[str, dict] = {
        "enaire": {},
        "sweden": {},
    }
    for key, preset in areas.items():
        grouped[source_for_area(key, preset)][key] = preset

    return [
        (source, grouped[source])
        for source in ("enaire", "sweden")
        if grouped[source]
    ]


def write_output_payload(
    output: dict,
    output_file: Path,
) -> None:
    output_file = output_file.expanduser().resolve()
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


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official ENAIRE or Swedish LFV AIP ENR 4.4 "
            "significant points and filter them to project study areas."
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
            "ENAIRE/LFV source."
        ),
    )

    parser.add_argument(
        "--source",
        choices=("enaire", "sweden", "auto"),
        default="auto",
        help=(
            "Reference source: enaire, sweden, or auto. The auto mode "
            "uses ENAIRE for Spanish presets and LFV for SCAT Swedish "
            "presets (default: auto)."
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
        default=None,
        help=(
            "Override the selected source URL. For ENAIRE, this is the "
            "direct ENR 4.4 HTML page; for Sweden, it can be the LFV eAIP "
            "index or a direct ENR 4.4 document."
        ),
    )

    parser.add_argument(
        "--source-html",
        type=Path,
        help=(
            "Read the selected source's ENR 4.4 from a local HTML file "
            "instead of downloading it. Useful for archived/historical "
            "AIRAC material; cannot be combined with --source auto."
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

    if (
        arguments.source == "auto"
        and (
            arguments.source_html is not None
            or arguments.source_url is not None
        )
    ):
        raise ValueError(
            "--source-html and --source-url select one source and cannot "
            "be combined with --source auto. Run the source-specific "
            "commands separately or use the default URLs."
        )

    source_groups = group_areas_by_source(
        areas,
        arguments.source,
    )

    output = (
        load_existing_output(arguments.output)
        if preserve_existing
        else {}
    )
    summaries: list[dict] = []

    for source, source_areas in source_groups:
        source_text, source_metadata = read_source_html(
            source,
            arguments.source_html,
            arguments.source_url,
            arguments.timeout,
        )

        effective_date = parse_effective_date(
            source_text
        )

        if source == "sweden":
            authority = "LFV / AIS Sweden"
            aip_class = "SignificantPoint"
        else:
            authority = "ENAIRE / AIS-ESPAÑA"
            aip_class = "DesignatedPoint"

        source_metadata.update(
            {
                "authority": authority,
                "aip_section": "ENR 4.4",
                "aixm_class": aip_class,
                "aixm_type": "ICAO",
                "effective_date": effective_date,
                "downloaded_utc": datetime.now(
                    timezone.utc
                ).replace(
                    microsecond=0
                ).isoformat(),
            }
        )

        if source == "sweden":
            source_waypoints = parse_sweden_significant_points(
                source_text
            )
        else:
            source_waypoints = parse_enaire_significant_points(
                source_text
            )

        for area_key, preset in source_areas.items():
            output[area_key] = build_area_record(
                area_key,
                preset,
                source_waypoints,
                source_metadata,
            )

        summaries.append(
            {
                "source": source,
                "metadata": source_metadata,
                "parsed": len(source_waypoints),
                "areas": source_areas,
            }
        )

    write_output_payload(
        output,
        arguments.output,
    )

    print("=" * 78)
    print("AIP ENR 4.4 waypoint update")
    print("=" * 78)

    for summary in summaries:
        source_metadata = summary["metadata"]
        print(
            f"Source                             : "
            f"{source_metadata['name']}"
        )
        print(
            f"Official significant points parsed : "
            f"{summary['parsed']:,}"
        )
        print(
            f"Source effective date              : "
            f"{source_metadata.get('effective_date') or 'not detected'}"
        )
        print(
            f"Updated area entries               : "
            f"{len(summary['areas'])}"
        )

        for area_key in summary["areas"]:
            count = int(
                output[area_key].get(
                    "count",
                    0,
                )
            )
            print(
                f"  {area_key:<36} {count:>6,} points"
            )
        print()

    print(
        f"Output                            : "
        f"{arguments.output.expanduser().resolve()}"
    )

    print(
        "NOTE: current AIP data is not automatically the historical network "
        "for a 2016 SCAT or 2019 OpenSky study. Use archived AIRAC HTML "
        "with --source-html or --source-url when historical consistency "
        "is required."
    )


if __name__ == "__main__":
    main()
