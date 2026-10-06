# opensky_collect.py

import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd


URL = "https://opensky-network.org/api/states/all"

BBOX = {
    "lamin": 41.5,
    "lamax": 44.0,
    "lomin": -3.5,
    "lomax": 1.5,
    "extended": 1,
}

INTERVAL_SECONDS = 10
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

output = Path("opensky_trajectories.csv")

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