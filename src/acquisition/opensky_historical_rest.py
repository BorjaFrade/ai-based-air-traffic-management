# opensky_historical_rest.py

import os
import time
import requests
import pandas as pd


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


# Hace 30 minutos
timestamp = int(time.time()) - 30 * 60


params = {
    "time": timestamp,

    # Norte de España
    "lamin": 41.5,
    "lamax": 44.0,
    "lomin": -3.5,
    "lomax": 1.5,

    "extended": 1,
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

print(df.head(20))

print()
print("Aircraft:", len(df))
print(
    "Remaining credits:",
    response.headers.get("X-Rate-Limit-Remaining")
)