# Data acquisition

This folder contains the current OpenSky data-acquisition scripts. They use
state-vector data from the OpenSky Network and are intended for academic
experimentation, not operational air-traffic control.

## Scripts

### `opensky_snapshot.py`

Downloads one current state-vector snapshot for the configured geographic area,
processes the aircraft states, performs proximity and separation analysis, and
generates a map. The area can be configured near the beginning of the file using
`AREA_MODE`, `SELECTED_PRESET`, `RADIUS_KM` and `CUSTOM_BBOX`.

Run from the repository root:

```bash
python src/acquisition/opensky_snapshot.py
```

It writes the raw snapshot to `data/raw/opensky/opensky_snapshot.csv` and the
visualisation to `results/figures/opensky_snapshot_map.png`. The script also
reports aircraft states, altitude and speed information, pairwise proximity and
conflict/proximity warnings according to its configured research thresholds.

### `opensky_collect.py`

Polls the OpenSky state-vector endpoint at regular intervals. The default
configuration collects one hour of observations every ten seconds for the
configured bounding box. Run it with:

```bash
python src/acquisition/opensky_collect.py
```

The resulting time series is saved as
`data/raw/opensky/opensky_trajectories.csv`. Edit `BBOX`, `INTERVAL_SECONDS` and
`DURATION_MINUTES` before running a different collection. Because this script
performs a live collection, check OpenSky rate limits and terms of use first.

### `opensky_historical_rest.py`

Demonstrates an authenticated request to the OpenSky REST API for a historical
state-vector timestamp. It requires the environment variables
`OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET`:

```bash
export OPENSKY_CLIENT_ID="your-client-id"
export OPENSKY_CLIENT_SECRET="your-client-secret"
python src/acquisition/opensky_historical_rest.py
```

The current script prints the returned aircraft table and remaining API credits;
it does not yet persist the response to a file. Do not commit credentials or
tokens to the repository.

## Dependencies and reproducibility

The scripts require Python packages including `requests`, `pandas`, `numpy`,
`matplotlib`, `contextily`, `pyproj` and `geonamescache`. The exact dependency
manifest will be formalised in the root `requirements.txt`. Raw data and maps
should be generated into the repository folders described above, not next to the
source scripts.
