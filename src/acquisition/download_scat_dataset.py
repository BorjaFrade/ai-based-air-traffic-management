#!/usr/bin/env python3
"""
Download the complete Swedish Civil Air Traffic Control (SCAT) dataset.

Dataset
-------
Nilsson & Unger, SCAT dataset, Mendeley Data, v1
DOI: 10.17632/8yn985bwz5.1
Dataset id: 8yn985bwz5

The dataset contains 13 one-week ZIP archives. Each weekly archive contains:
  - one JSON file per flight,
  - airspace.json,
  - grib_meteo.json.

The downloader keeps the original weekly ZIPs intact. This is intentional:
there are ~170k flight JSON files and later preprocessing can stream directly
from the ZIP archives without creating hundreds of thousands of tiny files.

ATC clearances are NOT stored in a separate database. They are embedded in
each flight JSON under the flight-plan object, normally:

    Fpl -> fpl_clearance

so the surveillance track and controller clearances are already linked by the
same SCAT flight file / flight id.

Default project layout
----------------------
If this script lives in:
    <project>/src/acquisition/download_scat_dataset.py

the default output is:
    <project>/data/raw/scat/

with:
    weekly_archives/
        scat20161015_20161021.zip
        ...
        scat20170916_20170922.zip
    manifest.json
    manifest.csv
    dataset_source.json

Download strategy
-----------------
Mendeley has changed its public download infrastructure over time. In "auto"
mode this script tries, in order:

1. Current public dataset files API / page discovery.
2. Public Mendeley "Download All" endpoint.
3. Current Mendeley Data API bulk ZIP endpoint.
4. Legacy Mendeley cached ZIP URL.

If the current API requires authentication, set MENDELEY_TOKEN or pass
--token. Public direct file downloads themselves normally do not require a
token once file ids have been discovered.

Examples
--------
Download all 13 weeks:

    python3 download_scat_dataset.py

Download and verify the archive central directories:

    python3 download_scat_dataset.py --verify

Download and additionally extract every weekly archive (not recommended
unless you explicitly want ~170k individual JSON files):

    python3 download_scat_dataset.py --extract

Only download selected weeks:

    python3 download_scat_dataset.py \
        --week 20170107_20170113 \
        --week 20170215_20170221
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests


# =============================================================================
# DATASET METADATA
# =============================================================================

DATASET_ID = "8yn985bwz5"
DATASET_VERSION = 1
DATASET_DOI = "10.17632/8yn985bwz5.1"

DATASET_PAGE = (
    f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}"
)

MENDELEY_PUBLIC_FILES_API = (
    f"https://api.data.mendeley.com/"
    f"datasets/publics/{DATASET_ID}/files"
)

MENDELEY_FILES_API = (
    f"https://api.data.mendeley.com/"
    f"datasets/{DATASET_ID}/files"
)

PUBLIC_BULK_URL = (
    f"https://data.mendeley.com/public-files/"
    f"datasets/{DATASET_ID}/zip/file_downloaded"
)

API_BULK_URL = (
    f"https://api.data.mendeley.com/"
    f"datasets/{DATASET_ID}/zip/file_downloaded"
)

LEGACY_BULK_URL = (
    "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/"
    f"{DATASET_ID}-{DATASET_VERSION}.zip"
)

USER_AGENT = (
    "Mozilla/5.0 "
    "(SCAT academic dataset downloader; "
    "https://doi.org/10.17632/8yn985bwz5.1)"
)

DEFAULT_TIMEOUT_S = 90.0
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_RETRIES = 4


@dataclass(frozen=True)
class WeekSpec:
    filename: str
    start_date: str
    end_date: str
    expected_flights: int

    @property
    def key(self) -> str:
        return (
            self.filename
            .removeprefix("scat")
            .removesuffix(".zip")
        )


WEEKS = (
    WeekSpec(
        "scat20161015_20161021.zip",
        "2016-10-15",
        "2016-10-21",
        13_138,
    ),
    WeekSpec(
        "scat20161112_20161118.zip",
        "2016-11-12",
        "2016-11-18",
        12_248,
    ),
    WeekSpec(
        "scat20161210_20161216.zip",
        "2016-12-10",
        "2016-12-16",
        12_099,
    ),
    WeekSpec(
        "scat20170107_20170113.zip",
        "2017-01-07",
        "2017-01-13",
        11_195,
    ),
    WeekSpec(
        "scat20170215_20170221.zip",
        "2017-02-15",
        "2017-02-21",
        11_610,
    ),
    WeekSpec(
        "scat20170304_20170310.zip",
        "2017-03-04",
        "2017-03-10",
        12_265,
    ),
    WeekSpec(
        "scat20170401_20170407.zip",
        "2017-04-01",
        "2017-04-07",
        13_255,
    ),
    WeekSpec(
        "scat20170429_20170505.zip",
        "2017-04-29",
        "2017-05-05",
        12_855,
    ),
    WeekSpec(
        "scat20170527_20170602.zip",
        "2017-05-27",
        "2017-06-02",
        13_832,
    ),
    WeekSpec(
        "scat20170624_20170630.zip",
        "2017-06-24",
        "2017-06-30",
        13_772,
    ),
    WeekSpec(
        "scat20170722_20170728.zip",
        "2017-07-22",
        "2017-07-28",
        12_950,
    ),
    WeekSpec(
        "scat20170819_20170825.zip",
        "2017-08-19",
        "2017-08-25",
        13_963,
    ),
    WeekSpec(
        "scat20170916_20170922.zip",
        "2017-09-16",
        "2017-09-22",
        14_365,
    ),
)

EXPECTED_FILENAMES = {
    week.filename
    for week in WEEKS
}


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def infer_project_root() -> Path:
    if (
        SCRIPT_DIR.name == "acquisition"
        and SCRIPT_DIR.parent.name == "src"
    ):
        return SCRIPT_DIR.parent.parent

    if SCRIPT_DIR.name == "src":
        return SCRIPT_DIR.parent

    return SCRIPT_DIR.parent


PROJECT_ROOT = infer_project_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "scat"


# =============================================================================
# HTTP HELPERS
# =============================================================================

def make_session(
    token: str | None,
) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        }
    )

    if token:
        session.headers.update(
            {
                "Authorization":
                    f"Bearer {token}"
            }
        )

    return session


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout_s: float,
    retries: int,
    **kwargs,
) -> requests.Response:
    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            response = session.request(
                method,
                url,
                timeout=timeout_s,
                **kwargs,
            )

            if (
                response.status_code
                in {
                    408,
                    425,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                and attempt < retries
            ):
                delay = min(
                    2 ** (attempt - 1),
                    12,
                )
                print(
                    f"  HTTP {response.status_code}; "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                continue

            return response

        except requests.RequestException as error:
            last_error = error

            if attempt >= retries:
                raise

            delay = min(
                2 ** (attempt - 1),
                12,
            )

            print(
                f"  Network error: {error}"
            )
            print(
                f"  Retrying in {delay}s..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Request failed: {last_error}"
    )


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def _flatten_api_items(
    payload,
) -> list[dict]:
    if isinstance(
        payload,
        list,
    ):
        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if not isinstance(
        payload,
        dict,
    ):
        return []

    for key in (
        "results",
        "items",
        "files",
        "data",
    ):
        value = payload.get(key)

        if isinstance(
            value,
            list,
        ):
            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    return []


def _api_item_filename(
    item: dict,
) -> str:
    for key in (
        "filename",
        "file_name",
        "name",
    ):
        value = item.get(key)

        if value:
            return str(value).strip()

    return ""


def _api_item_id(
    item: dict,
) -> str:
    for key in (
        "id",
        "file_id",
        "uuid",
    ):
        value = item.get(key)

        if value:
            return str(value).strip()

    content_details = item.get(
        "content_details"
    )

    if isinstance(
        content_details,
        dict,
    ):
        value = content_details.get(
            "id"
        )

        if value:
            return str(value).strip()

    return ""


def _api_item_download_url(
    item: dict,
) -> str:
    content_details = item.get(
        "content_details"
    )

    if isinstance(
        content_details,
        dict,
    ):
        value = content_details.get(
            "download_url"
        )

        if value:
            return str(value)

    for key in (
        "download_url",
        "url",
    ):
        value = item.get(key)

        if value:
            return str(value)

    return ""


def discover_from_current_api(
    session: requests.Session,
    *,
    timeout_s: float,
    retries: int,
) -> dict[str, str]:
    """
    Return {filename: download_url}.

    This first tries the current public-files list endpoint, then the generic
    dataset-files endpoint. If a file object has no direct download URL but
    has a UUID, use Mendeley's stable public-files download route.
    """
    endpoints = (
        MENDELEY_PUBLIC_FILES_API,
        MENDELEY_FILES_API,
    )

    discovered = {}

    for endpoint in endpoints:
        try:
            response = request_with_retries(
                session,
                "GET",
                endpoint,
                timeout_s=timeout_s,
                retries=retries,
                params={
                    "version":
                        DATASET_VERSION,
                    "$start":
                        0,
                    "$limit":
                        100,
                },
            )
        except requests.RequestException:
            continue

        if response.status_code in {
            401,
            403,
        }:
            continue

        if response.status_code != 200:
            continue

        try:
            payload = response.json()
        except ValueError:
            continue

        for item in _flatten_api_items(
            payload
        ):
            filename = _api_item_filename(
                item
            )

            if filename not in EXPECTED_FILENAMES:
                continue

            download_url = (
                _api_item_download_url(
                    item
                )
            )

            if not download_url:
                file_id = _api_item_id(
                    item
                )

                if file_id:
                    download_url = (
                        "https://data.mendeley.com/"
                        "public-files/datasets/"
                        f"{DATASET_ID}/files/"
                        f"{file_id}/file_downloaded"
                    )

            if download_url:
                discovered[
                    filename
                ] = download_url

        if len(
            discovered
        ) == len(
            EXPECTED_FILENAMES
        ):
            break

    return discovered


def discover_from_dataset_page(
    session: requests.Session,
    *,
    timeout_s: float,
    retries: int,
) -> dict[str, str]:
    """
    Best-effort discovery from Mendeley page hydration/HTML.

    The page implementation can change, so this is a fallback rather than the
    sole download mechanism.
    """
    try:
        response = request_with_retries(
            session,
            "GET",
            DATASET_PAGE,
            timeout_s=timeout_s,
            retries=retries,
        )
    except requests.RequestException:
        return {}

    if response.status_code != 200:
        return {}

    html = response.text
    discovered = {}

    direct_pattern = re.compile(
        r"https://data\.mendeley\.com/"
        r"public-files/datasets/"
        + re.escape(DATASET_ID)
        + r"/files/"
        r"[0-9a-fA-F-]{36}/file_downloaded"
    )

    urls = list(
        dict.fromkeys(
            direct_pattern.findall(
                html
            )
        )
    )

    # Sometimes filename and UUID appear in a hydrated JSON object rather
    # than as a complete direct URL. Search locally around each expected
    # filename for a UUID.
    for filename in EXPECTED_FILENAMES:
        position = html.find(
            filename
        )

        if position < 0:
            continue

        window = html[
            max(
                0,
                position - 2500,
            ):
            min(
                len(html),
                position + 2500,
            )
        ]

        uuid_match = re.search(
            r"[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}",
            window,
        )

        if uuid_match:
            file_id = (
                uuid_match.group(0)
            )

            discovered[
                filename
            ] = (
                "https://data.mendeley.com/"
                "public-files/datasets/"
                f"{DATASET_ID}/files/"
                f"{file_id}/file_downloaded"
            )

    # If there are exactly 13 direct URLs but page hydration does not expose
    # filenames close enough to match reliably, do NOT pair by order. That
    # could silently assign the wrong week. The bulk fallback is safer.
    _ = urls

    return discovered


def discover_week_urls(
    session: requests.Session,
    *,
    timeout_s: float,
    retries: int,
) -> dict[str, str]:
    discovered = discover_from_current_api(
        session,
        timeout_s=timeout_s,
        retries=retries,
    )

    if len(
        discovered
    ) < len(
        EXPECTED_FILENAMES
    ):
        page_discovered = (
            discover_from_dataset_page(
                session,
                timeout_s=timeout_s,
                retries=retries,
            )
        )

        for key, value in (
            page_discovered.items()
        ):
            discovered.setdefault(
                key,
                value,
            )

    return discovered


# =============================================================================
# DOWNLOAD
# =============================================================================

def human_bytes(
    value: int | float | None,
) -> str:
    if value is None:
        return "?"

    size = float(value)

    for unit in (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ):
        if size < 1024.0:
            return (
                f"{size:.1f} {unit}"
            )
        size /= 1024.0

    return f"{size:.1f} PiB"


def download_stream(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout_s: float,
    retries: int,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        and not overwrite
    ):
        # Existing ZIP central directory is enough to treat it as downloaded.
        try:
            with zipfile.ZipFile(
                destination,
                "r",
            ):
                pass

            print(
                f"  Already present: "
                f"{destination.name}"
            )
            return

        except zipfile.BadZipFile:
            print(
                f"  Existing file is not a valid ZIP; "
                f"re-downloading: {destination.name}"
            )

    partial = destination.with_suffix(
        destination.suffix
        + ".part"
    )

    existing_bytes = (
        partial.stat().st_size
        if partial.exists()
        and not overwrite
        else 0
    )

    headers = {}

    if existing_bytes > 0:
        headers[
            "Range"
        ] = (
            f"bytes={existing_bytes}-"
        )

    response = request_with_retries(
        session,
        "GET",
        url,
        timeout_s=timeout_s,
        retries=retries,
        headers=headers,
        stream=True,
        allow_redirects=True,
    )

    if response.status_code not in {
        200,
        206,
    }:
        raise RuntimeError(
            f"Download failed with HTTP "
            f"{response.status_code}: {url}"
        )

    append_mode = (
        existing_bytes > 0
        and response.status_code == 206
    )

    if not append_mode:
        existing_bytes = 0

    mode = (
        "ab"
        if append_mode
        else "wb"
    )

    content_length = response.headers.get(
        "Content-Length"
    )

    total_bytes = (
        existing_bytes
        + int(content_length)
        if content_length
        else None
    )

    downloaded = existing_bytes
    next_report = time.monotonic()

    with partial.open(mode) as handle:
        for chunk in response.iter_content(
            chunk_size=DEFAULT_CHUNK_SIZE
        ):
            if not chunk:
                continue

            handle.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()

            if now >= next_report:
                if total_bytes:
                    percentage = (
                        100.0
                        * downloaded
                        / total_bytes
                    )

                    print(
                        f"\r  {human_bytes(downloaded)} "
                        f"/ {human_bytes(total_bytes)} "
                        f"({percentage:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  {human_bytes(downloaded)}",
                        end="",
                        flush=True,
                    )

                next_report = (
                    now + 1.0
                )

    print()

    partial.replace(
        destination
    )


def try_bulk_download(
    session: requests.Session,
    bulk_path: Path,
    *,
    timeout_s: float,
    retries: int,
    overwrite: bool,
) -> tuple[bool, str | None]:
    candidates = (
        (
            "Mendeley public bulk",
            PUBLIC_BULK_URL,
            {
                "version":
                    DATASET_VERSION
            },
        ),
        (
            "Mendeley API bulk",
            API_BULK_URL,
            {
                "version":
                    DATASET_VERSION
            },
        ),
        (
            "Legacy Mendeley cache",
            LEGACY_BULK_URL,
            None,
        ),
    )

    for label, url, params in candidates:
        print()
        print(
            f"Trying {label}:"
        )
        print(
            f"  {url}"
        )

        temp = bulk_path.with_suffix(
            ".candidate.zip"
        )

        try:
            request_url = url

            if params:
                # Keep downloader generic while preserving standard requests
                # URL parameter handling for signed/redirect endpoints.
                prepared = requests.Request(
                    "GET",
                    request_url,
                    params=params,
                ).prepare()

                request_url = (
                    prepared.url
                    or request_url
                )

            download_stream(
                session,
                request_url,
                temp,
                timeout_s=timeout_s,
                retries=retries,
                overwrite=True,
            )

            with zipfile.ZipFile(
                temp,
                "r",
            ) as archive:
                names = archive.namelist()

                if not names:
                    raise zipfile.BadZipFile(
                        "Empty ZIP"
                    )

            if bulk_path.exists():
                bulk_path.unlink()

            temp.replace(
                bulk_path
            )

            return True, label

        except Exception as error:
            print(
                f"  Not usable: {error}"
            )

            if temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass

    return False, None


# =============================================================================
# BULK ZIP -> 13 WEEKLY ZIP FILES
# =============================================================================

def find_nested_week_archives(
    bulk_zip: Path,
) -> dict[str, str]:
    """
    Return {expected_filename: member_name} for weekly ZIPs inside the bulk ZIP.
    """
    found = {}

    with zipfile.ZipFile(
        bulk_zip,
        "r",
    ) as archive:
        for member in (
            archive.infolist()
        ):
            basename = Path(
                member.filename
            ).name

            if basename in EXPECTED_FILENAMES:
                found[
                    basename
                ] = member.filename

    return found


def extract_week_archives_from_bulk(
    bulk_zip: Path,
    archive_dir: Path,
    selected_weeks: set[str],
    *,
    overwrite: bool,
) -> int:
    found = find_nested_week_archives(
        bulk_zip
    )

    selected_filenames = {
        week.filename
        for week in WEEKS
        if week.key in selected_weeks
    }

    missing = (
        selected_filenames
        - set(
            found
        )
    )

    if missing:
        raise RuntimeError(
            "Bulk ZIP downloaded successfully but did not "
            "contain all requested SCAT weekly ZIP files. "
            f"Missing: {sorted(missing)}"
        )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_count = 0

    with zipfile.ZipFile(
        bulk_zip,
        "r",
    ) as archive:
        for filename in sorted(
            selected_filenames
        ):
            destination = (
                archive_dir
                / filename
            )

            if (
                destination.exists()
                and not overwrite
            ):
                try:
                    with zipfile.ZipFile(
                        destination,
                        "r",
                    ):
                        pass

                    print(
                        f"  Already present: "
                        f"{filename}"
                    )
                    continue

                except zipfile.BadZipFile:
                    pass

            member_name = found[
                filename
            ]

            print(
                f"  Extracting weekly archive: "
                f"{filename}"
            )

            with archive.open(
                member_name,
                "r",
            ) as source:
                with destination.open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(
                        source,
                        target,
                        length=DEFAULT_CHUNK_SIZE,
                    )

            extracted_count += 1

    return extracted_count


# =============================================================================
# VALIDATION / INVENTORY
# =============================================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    DEFAULT_CHUNK_SIZE
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def inspect_week_archive(
    path: Path,
    week: WeekSpec,
    *,
    deep_verify: bool,
) -> dict:
    result = {
        **asdict(
            week
        ),
        "week_key":
            week.key,
        "path":
            str(
                path.resolve()
            ),
        "exists":
            path.exists(),
        "size_bytes":
            (
                path.stat().st_size
                if path.exists()
                else None
            ),
        "valid_zip":
            False,
        "flight_json_count":
            None,
        "expected_flights_match":
            False,
        "has_airspace_json":
            False,
        "has_grib_meteo_json":
            False,
        "sha256":
            None,
        "deep_crc_ok":
            None,
    }

    if not path.exists():
        return result

    try:
        with zipfile.ZipFile(
            path,
            "r",
        ) as archive:
            names = [
                member.filename
                for member
                in archive.infolist()
                if not member.is_dir()
            ]

            basenames = [
                Path(name).name
                for name in names
            ]

            result[
                "valid_zip"
            ] = True

            result[
                "has_airspace_json"
            ] = (
                "airspace.json"
                in basenames
            )

            result[
                "has_grib_meteo_json"
            ] = (
                "grib_meteo.json"
                in basenames
            )

            flight_json_count = 0

            for name in basenames:
                if (
                    not name.endswith(
                        ".json"
                    )
                    or name
                    in {
                        "airspace.json",
                        "grib_meteo.json",
                    }
                ):
                    continue

                stem = Path(
                    name
                ).stem

                # Flight files use their numeric extraction id.
                if stem.isdigit():
                    flight_json_count += 1

            result[
                "flight_json_count"
            ] = flight_json_count

            result[
                "expected_flights_match"
            ] = (
                flight_json_count
                == week.expected_flights
            )

            if deep_verify:
                bad_member = (
                    archive.testzip()
                )

                result[
                    "deep_crc_ok"
                ] = (
                    bad_member is None
                )

    except zipfile.BadZipFile:
        return result

    return result


def extract_week_contents(
    archive_path: Path,
    output_root: Path,
    week: WeekSpec,
    *,
    overwrite: bool,
) -> None:
    destination = (
        output_root
        / "extracted_weeks"
        / week.key
    )

    marker = (
        destination
        / ".complete"
    )

    if (
        marker.exists()
        and not overwrite
    ):
        print(
            f"  Already extracted: "
            f"{week.key}"
        )
        return

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  Extracting flight JSONs: "
        f"{week.key}"
    )

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:
        archive.extractall(
            destination
        )

    marker.write_text(
        "complete\n",
        encoding="utf-8",
    )


def write_manifest(
    output_root: Path,
    rows: list[dict],
    *,
    source_method: str,
) -> None:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_json = (
        output_root
        / "manifest.json"
    )

    payload = {
        "dataset": {
            "name":
                "Swedish Civil Air Traffic Control (SCAT) dataset",
            "dataset_id":
                DATASET_ID,
            "version":
                DATASET_VERSION,
            "doi":
                DATASET_DOI,
            "dataset_page":
                DATASET_PAGE,
            "source_method":
                source_method,
            "expected_week_count":
                len(WEEKS),
            "expected_total_flights":
                sum(
                    week.expected_flights
                    for week in WEEKS
                ),
        },
        "weeks":
            rows,
    }

    manifest_json.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_csv = (
        output_root
        / "manifest.csv"
    )

    fieldnames = [
        "week_key",
        "filename",
        "start_date",
        "end_date",
        "expected_flights",
        "exists",
        "size_bytes",
        "valid_zip",
        "flight_json_count",
        "expected_flights_match",
        "has_airspace_json",
        "has_grib_meteo_json",
        "deep_crc_ok",
        "sha256",
        "path",
    ]

    with manifest_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key:
                        row.get(key)
                    for key
                    in fieldnames
                }
            )

    source_json = (
        output_root
        / "dataset_source.json"
    )

    source_json.write_text(
        json.dumps(
            {
                "dataset_page":
                    DATASET_PAGE,
                "doi":
                    DATASET_DOI,
                "licence":
                    "CC BY 4.0",
                "notes": [
                    (
                        "Each weekly archive contains flight JSONs, "
                        "airspace.json and grib_meteo.json."
                    ),
                    (
                        "ATC clearances are embedded in each flight JSON "
                        "under Fpl.fpl_clearance; no external join table "
                        "is required to associate a clearance with that "
                        "SCAT flight."
                    ),
                    (
                        "The original surveillance track is stored under "
                        "Plots and the system trajectory predictions under "
                        "predicted_trajectory."
                    ),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all 13 weeks of the SCAT Swedish ATC dataset "
            "and keep the original weekly ZIP archives."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output root. Default: "
            f"{DEFAULT_OUTPUT_ROOT}"
        ),
    )

    parser.add_argument(
        "--week",
        action="append",
        default=[],
        metavar="YYYYMMDD_YYYYMMDD",
        help=(
            "Download only one week. Repeat to select multiple weeks. "
            "Default: all 13."
        ),
    )

    parser.add_argument(
        "--token",
        default=(
            os.environ.get(
                "MENDELEY_TOKEN"
            )
        ),
        help=(
            "Optional Mendeley Data API bearer token. Can also be set in "
            "MENDELEY_TOKEN. Public downloads are attempted without one."
        ),
    )

    parser.add_argument(
        "--method",
        choices=(
            "auto",
            "files",
            "bulk",
        ),
        default="auto",
        help=(
            "auto: discover individual files then bulk fallback; "
            "files: require individual-file discovery; "
            "bulk: use Download All/bulk archive."
        ),
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Inspect every weekly ZIP after download and compare the number "
            "of flight JSONs with the published flight counts."
        ),
    )

    parser.add_argument(
        "--deep-verify",
        action="store_true",
        help=(
            "Additionally CRC-test every ZIP member. Much slower because "
            "the full archive must be decompressed."
        ),
    )

    parser.add_argument(
        "--hash",
        action="store_true",
        help=(
            "Calculate SHA-256 for each weekly archive. Reads all bytes."
        ),
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help=(
            "Extract all flight JSON files. Not required for later "
            "preprocessing and creates ~170k individual files."
        ),
    )

    parser.add_argument(
        "--keep-bulk",
        action="store_true",
        help=(
            "Keep the outer Download-All ZIP after weekly archives have "
            "been extracted from it."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing files.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
    )

    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error(
            "--timeout must be positive"
        )

    if args.retries <= 0:
        parser.error(
            "--retries must be positive"
        )

    return args


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    archive_dir = (
        output_root
        / "weekly_archives"
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    known_week_keys = {
        week.key
        for week in WEEKS
    }

    if args.week:
        selected_week_keys = set(
            args.week
        )

        unknown = (
            selected_week_keys
            - known_week_keys
        )

        if unknown:
            raise ValueError(
                "Unknown SCAT week key(s): "
                + ", ".join(
                    sorted(unknown)
                )
            )

    else:
        selected_week_keys = (
            known_week_keys
        )

    selected_specs = [
        week
        for week in WEEKS
        if week.key
        in selected_week_keys
    ]

    print(
        "=" * 78
    )
    print(
        "SCAT DATASET DOWNLOAD"
    )
    print(
        "=" * 78
    )
    print(
        f"Dataset DOI     : {DATASET_DOI}"
    )
    print(
        f"Dataset page    : {DATASET_PAGE}"
    )
    print(
        f"Output          : {output_root}"
    )
    print(
        f"Requested weeks : {len(selected_specs)}"
    )
    print(
        f"Expected flights: "
        f"{sum(w.expected_flights for w in selected_specs):,}"
    )
    print(
        "=" * 78
    )

    session = make_session(
        args.token
    )

    source_method = ""
    downloaded_any = False

    # -------------------------------------------------------------------------
    # Preferred route: discover the 13 public weekly files and download them
    # individually. This avoids first downloading a second outer ZIP.
    # -------------------------------------------------------------------------

    discovered = {}

    if args.method in {
        "auto",
        "files",
    }:
        print()
        print(
            "Discovering Mendeley weekly file URLs..."
        )

        discovered = discover_week_urls(
            session,
            timeout_s=args.timeout,
            retries=args.retries,
        )

        requested_discovered = {
            filename:
                url
            for filename, url
            in discovered.items()
            if (
                filename
                in {
                    week.filename
                    for week
                    in selected_specs
                }
            )
        }

        print(
            f"  Discovered requested week files: "
            f"{len(requested_discovered)}/"
            f"{len(selected_specs)}"
        )

        if (
            len(
                requested_discovered
            )
            == len(
                selected_specs
            )
        ):
            source_method = (
                "Mendeley individual public files"
            )

            for index, week in enumerate(
                selected_specs,
                start=1,
            ):
                destination = (
                    archive_dir
                    / week.filename
                )

                print()
                print(
                    f"[{index}/{len(selected_specs)}] "
                    f"{week.filename}"
                )

                download_stream(
                    session,
                    requested_discovered[
                        week.filename
                    ],
                    destination,
                    timeout_s=args.timeout,
                    retries=args.retries,
                    overwrite=args.overwrite,
                )

                downloaded_any = True

        elif args.method == "files":
            raise RuntimeError(
                "Could not discover all requested weekly SCAT file URLs. "
                "Use --method auto so the script can fall back to "
                "Mendeley's Download All archive."
            )

    # -------------------------------------------------------------------------
    # Fallback route: Download All and split out the 13 weekly archives.
    # -------------------------------------------------------------------------

    need_bulk = any(
        not (
            archive_dir
            / week.filename
        ).exists()
        for week in selected_specs
    )

    if (
        args.method == "bulk"
        or (
            args.method == "auto"
            and need_bulk
        )
    ):
        print()
        print(
            "Falling back to Mendeley Download All..."
        )

        bulk_path = (
            output_root
            / f"scat_mendeley_{DATASET_ID}_v{DATASET_VERSION}_all.zip"
        )

        ok, bulk_source = (
            try_bulk_download(
                session,
                bulk_path,
                timeout_s=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
            )
        )

        if not ok:
            raise RuntimeError(
                "\nCould not download SCAT automatically using any known "
                "Mendeley public endpoint.\n\n"
                "If Mendeley currently requires API authentication, obtain "
                "a Mendeley Data API bearer token and rerun with either:\n\n"
                "  export MENDELEY_TOKEN='...'\n"
                "  python3 download_scat_dataset.py\n\n"
                "or:\n\n"
                "  python3 download_scat_dataset.py --token '...'\n"
            )

        source_method = (
            bulk_source
            or "Mendeley bulk archive"
        )

        print()
        print(
            "Splitting Download-All ZIP into weekly archives..."
        )

        extract_week_archives_from_bulk(
            bulk_path,
            archive_dir,
            selected_week_keys,
            overwrite=args.overwrite,
        )

        downloaded_any = True

        if not args.keep_bulk:
            try:
                bulk_path.unlink()

                print(
                    "Removed temporary outer Download-All ZIP."
                )
            except OSError:
                pass

    # -------------------------------------------------------------------------
    # Inventory / verification
    # -------------------------------------------------------------------------

    print()
    print(
        "Inspecting weekly archives..."
    )

    rows = []

    for week in selected_specs:
        path = (
            archive_dir
            / week.filename
        )

        row = inspect_week_archive(
            path,
            week,
            deep_verify=(
                args.deep_verify
            ),
        )

        if (
            args.hash
            and path.exists()
        ):
            row[
                "sha256"
            ] = sha256_file(
                path
            )

        rows.append(
            row
        )

        count = row.get(
            "flight_json_count"
        )

        count_text = (
            f"{count:,}"
            if isinstance(
                count,
                int,
            )
            else "?"
        )

        match_text = (
            "OK"
            if row.get(
                "expected_flights_match"
            )
            else "CHECK"
        )

        print(
            f"  {week.key}: "
            f"flights={count_text} "
            f"(expected {week.expected_flights:,}) "
            f"[{match_text}]"
        )

    # Optional full extraction.
    if args.extract:
        print()
        print(
            "Extracting weekly contents..."
        )

        for week in selected_specs:
            path = (
                archive_dir
                / week.filename
            )

            if not path.exists():
                continue

            extract_week_contents(
                path,
                output_root,
                week,
                overwrite=args.overwrite,
            )

    write_manifest(
        output_root,
        rows,
        source_method=(
            source_method
            or "existing local archives"
        ),
    )

    valid_count = sum(
        bool(
            row.get(
                "valid_zip"
            )
        )
        for row in rows
    )

    matching_count = sum(
        bool(
            row.get(
                "expected_flights_match"
            )
        )
        for row in rows
    )

    print()
    print(
        "=" * 78
    )
    print(
        "SCAT DOWNLOAD SUMMARY"
    )
    print(
        "=" * 78
    )
    print(
        f"Weekly ZIPs valid       : "
        f"{valid_count}/{len(rows)}"
    )
    print(
        f"Flight counts matched   : "
        f"{matching_count}/{len(rows)}"
    )
    print(
        f"Weekly archives         : "
        f"{archive_dir}"
    )
    print(
        f"Manifest                : "
        f"{output_root / 'manifest.csv'}"
    )
    print(
        f"Source metadata         : "
        f"{output_root / 'dataset_source.json'}"
    )
    print()
    print(
        "Important SCAT structure:"
    )
    print(
        "  flight JSON -> Plots                 = surveillance track"
    )
    print(
        "  flight JSON -> Fpl -> fpl_clearance = controller clearances"
    )
    print(
        "  flight JSON -> predicted_trajectory  = TopSky predictions"
    )
    print(
        "  weekly ZIP  -> airspace.json         = points/sectors"
    )
    print(
        "  weekly ZIP  -> grib_meteo.json       = weather forecast"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
