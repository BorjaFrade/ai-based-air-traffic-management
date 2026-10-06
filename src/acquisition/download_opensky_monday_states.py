#!/usr/bin/env python3

"""
Download and geographically filter the OpenSky Network
"Weekly 24 Hours of State Vector Data 2017-2022" dataset.

The OpenSky public dataset contains complete Monday state vectors
with one file per hour.

The script:

1. Iterates over Mondays in a configurable date interval.
2. Downloads one global hourly file at a time.
3. Reads the CSV in chunks.
4. Filters one or more geographic bounding boxes.
5. Stores only the selected regions as Parquet files.
6. Deletes the temporary global file.
7. Continues with the next hour.

This avoids storing the complete global OpenSky dataset locally.

Master's Thesis:
AI-Based Air Traffic Management

Author:
Borja Rodríguez Frade
ORCID:
https://orcid.org/0000-0002-9097-1323
"""

from __future__ import annotations

import gzip
import io
import shutil
import tarfile
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Date range
# ------------------------------------------------------------
#
# Public Weekly State Vector dataset:
#
#   2017 -> 2022
#
# The script automatically selects Mondays.
#
# For a first test I strongly recommend:
#
# START_DATE = "2022-06-27"
# END_DATE   = "2022-06-27"
#
# before launching the complete historical download.
#

START_DATE = "2017-06-05"
END_DATE = "2022-06-27"


# ------------------------------------------------------------
# Geographic regions
# ------------------------------------------------------------
#
# Each region is defined as a rectangular bounding box:
#
# lamin = minimum latitude  -> South
# lamax = maximum latitude  -> North
# lomin = minimum longitude -> West
# lomax = maximum longitude -> East
#
# You can define as many regions as required.
#

REGIONS = {

    "GALICIA": {
        "lamin": 41.75,
        "lamax": 43.85,
        "lomin": -9.40,
        "lomax": -6.65,
    },

    # Example of a second region:
    #
    # "REGION_2": {
    #     "lamin": 40.0,
    #     "lamax": 42.0,
    #     "lomin": -5.0,
    #     "lomax": -2.0,
    # },

}


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_ROOT = Path("data/opensky_weekly")


# ------------------------------------------------------------
# CSV processing
# ------------------------------------------------------------
#
# Number of global records read at once.
#
# Increase if the machine has plenty of RAM.
# Decrease if memory consumption is high.
#

CHUNK_SIZE = 500_000


# ------------------------------------------------------------
# Optional filters
# ------------------------------------------------------------

# Normally keep both airborne and ground aircraft.
KEEP_ONLY_AIRBORNE = False


# ------------------------------------------------------------
# Parquet compression
# ------------------------------------------------------------

PARQUET_COMPRESSION = "zstd"


# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 180

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


# ============================================================
# OPENSKY DOWNLOAD LOCATIONS
# ============================================================

#
# OpenSky has used slightly different layouts for these
# historical files over time.
#
# The script tries several known public locations automatically.
#

DOWNLOAD_HOSTS = [

    "https://s3.opensky-network.org/data-samples/states",

    # Historical AWS endpoint
    "https://data-samples.s3.amazonaws.com/states",

]


# ============================================================
# FUNCTIONS
# ============================================================


def create_http_session() -> requests.Session:
    """
    Create a requests session with automatic retries.
    """

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent":
                "UNED-Master-Thesis-OpenSky-Research/1.0"
        }
    )

    return session


# ============================================================


def get_mondays(
    start: date,
    end: date,
):
    """
    Yield every Monday between start and end inclusive.
    """

    current = start

    # Monday = 0
    days_until_monday = (
        7 - current.weekday()
    ) % 7

    current += timedelta(
        days=days_until_monday
    )

    while current <= end:

        yield current

        current += timedelta(
            days=7
        )


# ============================================================


def build_candidate_urls(
    day: date,
    hour: int,
) -> list[str]:
    """
    Generate candidate OpenSky URLs for one hourly file.

    Several historical layouts are tried because the public
    repository structure has changed over time.
    """

    date_string = day.strftime(
        "%Y-%m-%d"
    )

    hour_string = f"{hour:02d}"

    filename = (
        f"states_"
        f"{date_string}-"
        f"{hour_string}.csv"
    )

    urls = []

    for host in DOWNLOAD_HOSTS:

        # Current-looking layout:
        #
        # states/YYYY-MM-DD/HH/...

        normal_path = (
            f"{host}/"
            f"{date_string}/"
            f"{hour_string}"
        )

        urls.extend(
            [
                f"{normal_path}/{filename}.tar",
                f"{normal_path}/{filename}.gz",
            ]
        )

        # Older public dataset layout:
        #
        # states/.YYYY-MM-DD/HH/...

        legacy_path = (
            f"{host}/."
            f"{date_string}/"
            f"{hour_string}"
        )

        urls.extend(
            [
                f"{legacy_path}/{filename}.tar",
                f"{legacy_path}/{filename}.gz",
            ]
        )

    return urls


# ============================================================


def download_hour_file(
    session: requests.Session,
    day: date,
    hour: int,
    temp_directory: Path,
):
    """
    Try the known OpenSky download URLs until one succeeds.

    Returns
    -------
    (Path, URL)
        Path of downloaded temporary file and source URL.

    (None, None)
        If no candidate URL exists.
    """

    urls = build_candidate_urls(
        day,
        hour,
    )

    for url in urls:

        print(
            f"      trying: {url}"
        )

        try:

            with session.get(
                url,
                stream=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
            ) as response:

                if response.status_code == 404:
                    continue

                response.raise_for_status()

                extension = (
                    ".tar"
                    if url.endswith(".tar")
                    else ".gz"
                )

                filename = (
                    f"opensky_"
                    f"{day.isoformat()}_"
                    f"{hour:02d}"
                    f"{extension}"
                )

                output_path = (
                    temp_directory
                    / filename
                )

                downloaded_bytes = 0

                with open(
                    output_path,
                    "wb",
                ) as file:

                    for block in response.iter_content(
                        chunk_size=DOWNLOAD_CHUNK_SIZE
                    ):

                        if not block:
                            continue

                        file.write(block)

                        downloaded_bytes += len(
                            block
                        )

                size_mb = (
                    downloaded_bytes
                    / 1024
                    / 1024
                )

                print(
                    f"      downloaded "
                    f"{size_mb:.1f} MB"
                )

                return (
                    output_path,
                    url,
                )

        except requests.RequestException as exc:

            print(
                f"      download error: "
                f"{exc}"
            )

    return None, None


# ============================================================


def dataframe_chunks_from_gzip(
    filepath: Path,
):
    """
    Read a gzip-compressed OpenSky CSV in chunks.
    """

    with gzip.open(
        filepath,
        mode="rt",
        encoding="utf-8",
        errors="replace",
    ) as stream:

        for chunk in pd.read_csv(
            stream,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            yield chunk


# ============================================================


def dataframe_chunks_from_tar(
    filepath: Path,
):
    """
    Read the state-vector CSV contained inside an OpenSky TAR
    archive without extracting the complete file permanently.
    """

    with tarfile.open(
        filepath,
        mode="r:*",
    ) as archive:

        members = [

            member

            for member
            in archive.getmembers()

            if (
                member.isfile()
                and (
                    member.name.endswith(
                        ".csv"
                    )
                    or
                    member.name.endswith(
                        ".csv.gz"
                    )
                )
            )
        ]

        if not members:

            raise RuntimeError(
                "No CSV file found "
                "inside TAR archive."
            )

        # Prefer the state vector file
        members.sort(
            key=lambda member:
            (
                "states_" not in member.name,
                member.name,
            )
        )

        member = members[0]

        print(
            f"      archive member: "
            f"{member.name}"
        )

        raw_stream = archive.extractfile(
            member
        )

        if raw_stream is None:

            raise RuntimeError(
                "Could not open CSV "
                "inside TAR archive."
            )

        # ----------------------------------------------------
        # CSV.GZ inside TAR
        # ----------------------------------------------------

        if member.name.endswith(
            ".gz"
        ):

            with gzip.GzipFile(
                fileobj=raw_stream,
                mode="rb",
            ) as gzip_stream:

                with io.TextIOWrapper(
                    gzip_stream,
                    encoding="utf-8",
                    errors="replace",
                ) as text_stream:

                    for chunk in pd.read_csv(
                        text_stream,
                        chunksize=CHUNK_SIZE,
                        low_memory=False,
                    ):

                        yield chunk

        # ----------------------------------------------------
        # Plain CSV inside TAR
        # ----------------------------------------------------

        else:

            with io.TextIOWrapper(
                raw_stream,
                encoding="utf-8",
                errors="replace",
            ) as text_stream:

                for chunk in pd.read_csv(
                    text_stream,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                ):

                    yield chunk


# ============================================================


def dataframe_chunks(
    filepath: Path,
):
    """
    Detect archive type and return CSV chunks.
    """

    if filepath.suffix == ".tar":

        yield from (
            dataframe_chunks_from_tar(
                filepath
            )
        )

    elif filepath.suffix == ".gz":

        yield from (
            dataframe_chunks_from_gzip(
                filepath
            )
        )

    else:

        raise RuntimeError(
            f"Unsupported file type: "
            f"{filepath}"
        )


# ============================================================


def normalize_coordinates(
    dataframe: pd.DataFrame,
):
    """
    Ensure latitude and longitude are numeric.
    """

    if "lat" not in dataframe.columns:

        raise RuntimeError(
            "Column 'lat' not found."
        )

    if "lon" not in dataframe.columns:

        raise RuntimeError(
            "Column 'lon' not found."
        )

    dataframe["lat"] = pd.to_numeric(
        dataframe["lat"],
        errors="coerce",
    )

    dataframe["lon"] = pd.to_numeric(
        dataframe["lon"],
        errors="coerce",
    )

    return dataframe


# ============================================================


def airborne_mask(
    dataframe: pd.DataFrame,
):
    """
    Return True for aircraft not marked as on ground.
    """

    if "onground" not in dataframe.columns:

        return pd.Series(
            True,
            index=dataframe.index,
        )

    values = (
        dataframe["onground"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return ~values.isin(
        [
            "true",
            "1",
            "t",
        ]
    )


# ============================================================


def filter_regions(
    filepath: Path,
):
    """
    Read one global hourly file and filter every configured
    region in a single pass.

    Returns
    -------
    dict[str, DataFrame]
        One dataframe per geographic region.
    """

    region_parts = {

        region_name: []

        for region_name
        in REGIONS

    }

    global_rows = 0

    for chunk_number, chunk in enumerate(
        dataframe_chunks(
            filepath
        ),
        start=1,
    ):

        chunk.columns = [

            str(column).strip()

            for column
            in chunk.columns

        ]

        chunk = normalize_coordinates(
            chunk
        )

        global_rows += len(
            chunk
        )

        if KEEP_ONLY_AIRBORNE:

            flight_mask = airborne_mask(
                chunk
            )

        else:

            flight_mask = pd.Series(
                True,
                index=chunk.index,
            )

        for (
            region_name,
            bbox,
        ) in REGIONS.items():

            spatial_mask = (

                chunk["lat"].between(
                    bbox["lamin"],
                    bbox["lamax"],
                    inclusive="both",
                )

                &

                chunk["lon"].between(
                    bbox["lomin"],
                    bbox["lomax"],
                    inclusive="both",
                )

            )

            mask = (
                spatial_mask
                & flight_mask
            )

            if not mask.any():
                continue

            selected = (
                chunk.loc[mask]
                .copy()
            )

            # Useful for merging datasets later.
            selected.insert(
                0,
                "region",
                region_name,
            )

            region_parts[
                region_name
            ].append(
                selected
            )

        print(
            f"      chunk "
            f"{chunk_number:03d} | "
            f"global rows processed: "
            f"{global_rows:,}"
        )

    result = {}

    for (
        region_name,
        parts,
    ) in region_parts.items():

        if parts:

            result[
                region_name
            ] = pd.concat(
                parts,
                ignore_index=True,
            )

        else:

            result[
                region_name
            ] = pd.DataFrame()

    return result


# ============================================================


def get_region_output_path(
    region_name: str,
    day: date,
    hour: int,
):
    """
    Build a partitioned output path.
    """

    directory = (

        OUTPUT_ROOT
        / region_name
        / f"year={day.year}"
        / f"date={day.isoformat()}"

    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"states_"
        f"{day.isoformat()}-"
        f"{hour:02d}.parquet"
    )

    return directory / filename


# ============================================================


def save_regions(
    filtered_data,
    day: date,
    hour: int,
):
    """
    Store each selected region as Parquet.
    """

    counts = {}

    for (
        region_name,
        dataframe,
    ) in filtered_data.items():

        row_count = len(
            dataframe
        )

        counts[
            region_name
        ] = row_count

        if dataframe.empty:

            print(
                f"      {region_name}: "
                f"0 rows"
            )

            continue

        output_path = (
            get_region_output_path(
                region_name,
                day,
                hour,
            )
        )

        dataframe.to_parquet(
            output_path,
            index=False,
            compression=PARQUET_COMPRESSION,
        )

        print(
            f"      {region_name}: "
            f"{row_count:,} rows -> "
            f"{output_path}"
        )

    return counts


# ============================================================


def completed_marker_path(
    day: date,
    hour: int,
):
    """
    File used to avoid downloading an already processed hour.
    """

    directory = (
        OUTPUT_ROOT
        / "_completed"
        / str(day.year)
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (

        directory
        / (
            f"{day.isoformat()}-"
            f"{hour:02d}.done"
        )

    )


# ============================================================


def write_completed_marker(
    day: date,
    hour: int,
    source_url: str,
    counts,
):
    """
    Mark one hourly global file as successfully processed.
    """

    marker = completed_marker_path(
        day,
        hour,
    )

    with open(
        marker,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"source={source_url}\n"
        )

        for (
            region,
            count,
        ) in counts.items():

            file.write(
                f"{region}={count}\n"
            )


# ============================================================


def print_configuration():
    """
    Print current download configuration.
    """

    print()
    print("=" * 72)
    print(
        "OpenSky Weekly State Vector Downloader"
    )
    print("=" * 72)

    print(
        f"Period: "
        f"{START_DATE} -> {END_DATE}"
    )

    print()

    print(
        "Configured regions:"
    )

    for name, bbox in REGIONS.items():

        print()
        print(
            f"  {name}"
        )

        print(
            f"    latitude : "
            f"{bbox['lamin']} "
            f"-> "
            f"{bbox['lamax']}"
        )

        print(
            f"    longitude: "
            f"{bbox['lomin']} "
            f"-> "
            f"{bbox['lomax']}"
        )

    print()

    print(
        f"Output: "
        f"{OUTPUT_ROOT.resolve()}"
    )

    print("=" * 72)
    print()


# ============================================================
# MAIN
# ============================================================


def main():

    start = date.fromisoformat(
        START_DATE
    )

    end = date.fromisoformat(
        END_DATE
    )

    if start > end:

        raise ValueError(
            "START_DATE must be before END_DATE."
        )

    if not REGIONS:

        raise ValueError(
            "At least one geographic region "
            "must be configured."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_configuration()

    session = create_http_session()

    mondays = list(
        get_mondays(
            start,
            end,
        )
    )

    print(
        f"Mondays to process: "
        f"{len(mondays)}"
    )

    print(
        f"Maximum hourly files: "
        f"{len(mondays) * 24}"
    )

    print()

    try:

        for (
            monday_number,
            monday,
        ) in enumerate(
            mondays,
            start=1,
        ):

            print()
            print("=" * 72)

            print(
                f"MONDAY "
                f"{monday_number}/"
                f"{len(mondays)}: "
                f"{monday}"
            )

            print("=" * 72)

            for hour in range(24):

                marker = (
                    completed_marker_path(
                        monday,
                        hour,
                    )
                )

                if marker.exists():

                    print(
                        f"[{monday} "
                        f"{hour:02d}:00] "
                        f"already processed"
                    )

                    continue

                print()
                print(
                    f"[{monday} "
                    f"{hour:02d}:00 UTC]"
                )

                # One global file exists only
                # temporarily.
                with tempfile.TemporaryDirectory(
                    prefix="opensky_"
                ) as temp:

                    temp_directory = Path(
                        temp
                    )

                    (
                        downloaded_file,
                        source_url,

                    ) = download_hour_file(

                        session,
                        monday,
                        hour,
                        temp_directory,

                    )

                    if downloaded_file is None:

                        print(
                            "      file not found"
                        )

                        continue

                    try:

                        filtered_data = (
                            filter_regions(
                                downloaded_file
                            )
                        )

                        counts = save_regions(
                            filtered_data,
                            monday,
                            hour,
                        )

                        write_completed_marker(
                            monday,
                            hour,
                            source_url,
                            counts,
                        )

                    except Exception as exc:

                        print(
                            f"      processing error: "
                            f"{exc}"
                        )

                        raise

                    # TemporaryDirectory automatically
                    # deletes the global OpenSky file here.

    except KeyboardInterrupt:

        print()
        print(
            "Download interrupted by user."
        )

    finally:

        session.close()

    print()
    print("=" * 72)
    print(
        "Processing finished."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()