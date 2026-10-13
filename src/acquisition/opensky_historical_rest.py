# opensky_historical_rest.py

import os
import time
import requests
import pandas as pd
from pathlib import Path

from area_presets import (
    load_area_presets,
    resolve_area_preset,
    filter_dataframe_to_area,
)


TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)

API_URL = "https://opensky-network.org/api/states/all"


CLIENT_ID = os.environ["OPENSKY_CLIENT_ID"]
CLIENT_SECRET = os.environ["OPENSKY_CLIENT_SECRET"]


def get_token():

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["access_token"]


token = get_token()


# 30 minutes ago
timestamp = int(time.time()) - 30 * 60


BBOX_PRESETS_FILE = Path(__file__).with_name("bbox_presets.json")
BBOX_PRESETS = load_area_presets(BBOX_PRESETS_FILE)
SELECTED_REGION = "NORTHERN_SPAIN"
AREA = resolve_area_preset(SELECTED_REGION, BBOX_PRESETS)

params = {
    "time": timestamp,
    **AREA["bbox"],
}


response = requests.get(
    API_URL,
    params=params,
    headers={
        "Authorization": f"Bearer {token}"
    },
    timeout=30,
)

response.raise_for_status()

data = response.json()

columns = [
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

df = pd.DataFrame(
    data.get("states", []),
    columns=columns,
)

df = filter_dataframe_to_area(
    df,
    AREA,
    latitude_column="latitude",
    longitude_column="longitude",
)

print(df.head(20))

print()
print("Aircraft:", len(df))
print(
    "Remaining credits:",
    response.headers.get("X-Rate-Limit-Remaining")
)