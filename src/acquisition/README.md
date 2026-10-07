# Data acquisition

This folder contains the current OpenSky data-acquisition scripts. They use
state-vector data from the OpenSky Network and are intended for academic
experimentation, not operational air-traffic control.

## Scripts

### `opensky_snapshot.py`

Downloads one current state-vector snapshot for the configured geographic area,
processes the aircraft states, performs proximity and separation analysis, and
generates a map. The area can be configured near the beginning of the file using
`AREA_MODE`, `SELECTED_PRESET`, `RADIUS_KM` and `SELECTED_BBOX_PRESET`. The
center-radius
locations are loaded from [`location_presets.json`](location_presets.json), so
new locations can be added there without modifying the Python script. Each entry
must contain `name`, `icao`, `lat` and `lon`.

The separation thresholds are loaded from
[`separation_profiles.json`](separation_profiles.json). Select one with
`SEPARATION_PROFILE`; each profile must contain `horizontal_nm` and `vertical_ft`.

The BBOX definitions are loaded from [`bbox_presets.json`](bbox_presets.json).
Select one with `SELECTED_BBOX_PRESET`. A preset can use `type: "vertices"`
with `lamin`, `lamax`, `lomin` and `lomax`, or `type: "center"` with `center`
(`lat` and `lon`) plus `radius_km`.

Run from the repository root:

```bash
python src/acquisition/opensky_snapshot.py
```

It writes the raw snapshot to `data/raw/opensky/opensky_snapshot.csv` and the
visualisation to `results/figures/opensky_snapshot_map.png`. The script also
reports aircraft states, altitude and speed information, pairwise proximity and
conflict/proximity warnings according to its configured research thresholds.
Each trajectory vector is annotated at its tip with the three-digit true track
relative to geographic north, such as `072°` or `250°`.

### `opensky_plot.py`

Contains the reusable map renderer used by `opensky_snapshot.py`. It draws the
basemap, aircraft symbols, trajectory vectors, safety areas, cities and
collision-free labels. Other scripts can import `render_map()` and provide
their own aircraft dataframe, area, analysis results and visual configuration
without performing an OpenSky download.

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

### `download_opensky_monday_states.py`

Downloads and geographically filters OpenSky's
[Weekly 24 Hours of State Vector Data 2017–2022](https://opensky-network.org/data/scientific#d1)
scientific dataset. OpenSky describes this dataset as complete Monday state
vectors, available in hourly files with 10-second update intervals and CSV, Avro
or JSON formats. This script downloads one global hourly archive at a time,
reads it in chunks, keeps only the configured bounding boxes, writes regional
Parquet files, and deletes the temporary global archive. This avoids retaining
the full global dataset locally.

Before a complete download, edit `START_DATE`, `END_DATE` and `REGIONS` near the
top of the script. The script automatically selects Mondays within the date
range. A safe first test is a single Monday:

```python
START_DATE = "2022-06-27"
END_DATE = "2022-06-27"
```

Run it from the repository root:

```bash
python src/acquisition/download_opensky_monday_states.py
```

By default, results are written under `data/raw/opensky_weekly/`, partitioned by
region, year, date and hour:

```text
data/raw/opensky_weekly/
├── GALICIA/
│   └── year=2022/date=2022-06-27/states_2022-06-27-00.parquet
└── _completed/
    └── 2022/2022-06-27-00.done
```

The `.done` marker records the source URL and row counts and allows the script to
resume without downloading an already processed hour. The global source files
are temporary; the retained data is the geographically filtered regional output.
Configure `CHUNK_SIZE` according to available memory, and set
`KEEP_ONLY_AIRBORNE = True` if ground aircraft should be excluded. Parquet
output requires a suitable pandas engine such as `pyarrow`. Historical files are
kept under `data/raw/opensky_weekly/` because they are downloaded source-derived
data and should not be committed unless redistribution and storage are explicitly
intended.

## Dependencies and reproducibility

The scripts require the packages listed in the root `requirements.txt`, including
`requests`, `pandas`, `numpy`, `matplotlib`, `contextily`, `pyproj`,
`geonamescache` and `pyarrow`. Raw data and maps
should be generated into the repository folders described above, not next to the
source scripts.
