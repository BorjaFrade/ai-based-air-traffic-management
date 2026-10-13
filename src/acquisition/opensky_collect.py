# opensky_collect.py

import time
from pathlib import Path
from datetime import datetime, timezone

from area_presets import (
    load_area_presets,
    resolve_area_preset,
    filter_dataframe_to_area,
)

import requests
import pandas as pd


URL = "https://opensky-network.org/api/states/all"

BBOX_PRESETS_FILE = Path(__file__).with_name("bbox_presets.json")
BBOX_PRESETS = load_area_presets(BBOX_PRESETS_FILE)

SELECTED_REGION = "GALICIA"
AREA = resolve_area_preset(
    SELECTED_REGION,
    BBOX_PRESETS,
)
BBOX = AREA["bbox"]

INTERVAL_SECONDS = 5
DURATION_MINUTES = 60

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
output = PROJECT_ROOT / "data" / "raw" / "opensky" / "opensky_trajectories.csv"
output.parent.mkdir(parents=True, exist_ok=True)

samples = int(
    DURATION_MINUTES * 60 / INTERVAL_SECONDS
)

session = requests.Session()

all_data = []

for i in range(samples):

    try:
        response = session.get(
            URL,
            params=BBOX,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        timestamp = payload.get("time")

        states = payload.get("states", [])

        if states:

            df = pd.DataFrame(
                states,
                columns=COLUMNS,
            )

            df = filter_dataframe_to_area(
                df,
                AREA,
                latitude_column="latitude",
                longitude_column="longitude",
            )

            df["snapshot_time"] = timestamp

            df["callsign"] = (
                df["callsign"]
                .fillna("")
                .str.strip()
            )

            all_data.append(df)

            print(
                f"{i+1:04d}/{samples} | "
                f"{datetime.fromtimestamp(timestamp, timezone.utc)} | "
                f"{len(df)} aircraft | "
                f"credits={response.headers.get('X-Rate-Limit-Remaining')}"
            )

        else:
            print("No aircraft returned.")

    except Exception as exc:
        print("ERROR:", exc)

    time.sleep(INTERVAL_SECONDS)


if all_data:

    result = pd.concat(
        all_data,
        ignore_index=True
    )

    result.to_csv(
        output,
        index=False
    )

    print("\nDONE")
    print("Observations:", len(result))
    print("Aircraft:", result["icao24"].nunique())
    print("Saved:", output)

else:

    print("No data collected.")
