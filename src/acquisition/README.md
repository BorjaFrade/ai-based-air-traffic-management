# Data acquisition and map rendering

This directory contains OpenSky acquisition scripts, ENAIRE reference-data
downloaders, the configurable separation model and the shared map renderer.
They are research tools and must not be used as operational ATC software.

Run every command from the repository root. Historical times are UTC.

## Configuration files

- [`bbox_presets.json`](bbox_presets.json): bounding boxes, polygons and render
  sizes. Current keys include `LECMSAN_ENVELOPE`, `LECMSAN`, `GALICIA`,
  `NORTHERN_SPAIN`, `BARCELONA_50KM` and the SCAT-backed Swedish sectors
  `SCAT_ESMM_GOTHENBURG_GG`, `SCAT_ESMM_MALMO_MS` and
  `SCAT_ESOS_STOCKHOLM_SE`.
- [`location_presets.json`](location_presets.json): airport-centred locations
  used by the live snapshot script.
- [`separation_profiles.json`](separation_profiles.json): `ENROUTE`,
  `TERMINAL`, `FINAL_APPROACH` and `CUSTOM` research profiles.
- [`waypoints.json`](waypoints.json) and [`corridors.json`](corridors.json):
  ENAIRE reference layers used by map rendering and validation screenshots.

The SCAT airspace envelopes can be checked against the downloaded
`airspace.json` with:

```bash
python3 src/acquisition/validate_scat_airspace_presets.py
```

## Historical OpenSky archive

### Resumable monthly downloader

[`download_opensky_sector.py`](download_opensky_sector.py) queries OpenSky
Trino in bounded windows and writes one monthly CSV per sector. It requires
`pyopensky`, `python-dotenv` and OpenSky credentials in
`credentials.env` or the environment. The end timestamp is exclusive.

```bash
python3 src/acquisition/download_opensky_sector.py \
  --sector LECMSAN_ENVELOPE \
  --start "2019-01-01 00:00:00" \
  --end "2019-03-01 00:00:00"
```

Options:

```text
--sector NAME             bbox_presets.json key; default GALICIA
--start TIMESTAMP         UTC inclusive; default 2025-01-01 00:00:00
--end TIMESTAMP           UTC exclusive; default 2025-02-01 00:00:00
--max-windows N           Maximum new Trino windows; 0 means unlimited
```

For a first connection test, limit the run:

```bash
python3 src/acquisition/download_opensky_sector.py \
  --sector LECMSAN_ENVELOPE \
  --start "2019-01-01 00:00:00" \
  --end "2019-01-02 00:00:00" \
  --max-windows 1
```

Raw output is stored under
`data/raw/opensky_sectors/<sector>/year=YYYY/month=MM/`. The downloader caches
completed windows and can be rerun after interruption.

### Configurable Trino visualisation workflow

[`download_opensky_trino_states.py`](download_opensky_trino_states.py) is an
older, richer workflow that queries a configured interval and can create map
frames and an MP4. It has no CLI date arguments: edit `START_UTC`, `END_UTC`,
`SELECTED_REGION`, `GENERATE_VIDEO` and the other user-configuration constants
near the top of the file before running it.

```python
START_UTC = "2019-01-15 12:00:00"
END_UTC = "2019-01-15 12:20:00"
SELECTED_REGION = "LECMSAN_ENVELOPE"
```

```bash
python3 src/acquisition/download_opensky_trino_states.py
```

For the reproducible large historical pipeline, prefer
`download_opensky_sector.py` followed by
`src/preprocessing/preprocess_opensky_10s.py`.

## Live snapshot

[`opensky_snapshot.py`](opensky_snapshot.py) calls the live OpenSky states API,
classifies current aircraft pairs and renders one map. It has no CLI arguments;
edit the user configuration at the top of the file. Typical choices are:

```python
AREA_MODE = "BBOX"
SELECTED_BBOX_PRESET = "GALICIA"

# Or use a centre/radius area:
AREA_MODE = "CENTER_RADIUS"
SELECTED_PRESET = "BCN"
RADIUS_KM = 50
```

Run it with:

```bash
python3 src/acquisition/opensky_snapshot.py
```

The current snapshot is written to `data/raw/opensky/opensky_snapshot.csv`.
The map is `results/figures/opensky_snapshot_map.png`; relevant pair data is
also exported under `results/figures/`. These files are overwritten on the next
run, so preserve named copies for comparisons.

## ENAIRE waypoints and ATS corridors

### Waypoints

[`download_waypoints.py`](download_waypoints.py) downloads ENAIRE AIP ENR 4.4
significant points and filters them to one or more configured areas.

```bash
# Update two areas while preserving other entries
python3 src/acquisition/download_waypoints.py \
  --areas LECMSAN GALICIA

# Rebuild every configured area
python3 src/acquisition/download_waypoints.py --all-areas

# Use an archived AIP HTML file for a historical study
python3 src/acquisition/download_waypoints.py \
  --areas LECMSAN \
  --source-html /path/to/historical_ENR_4_4_en.html
```

The output defaults to [`waypoints.json`](waypoints.json). The default source
is the current ENAIRE AIP, so it must not be described as the 2019 waypoint
network without an archived historical source.

### ATS corridors

[`download_enaire_corridors.py`](download_enaire_corridors.py) parses ENAIRE
AIP ENR 3.2 corridors for one area and writes [`corridors.json`](corridors.json).
It requires `beautifulsoup4`; the default source is current ENAIRE data.

```bash
python3 src/acquisition/download_enaire_corridors.py \
  --area LECMSAN \
  --output src/acquisition/corridors.json

# Historical source and a finer sampling interval
python3 src/acquisition/download_enaire_corridors.py \
  --area LECMSAN \
  --source-html /path/to/historical_ENR_3_2_en.html \
  --area-sample-step-nm 5
```

The script also accepts `--source-url`, `--bbox-presets`, `--output` and
`--timeout`.

## Smaller or legacy collectors

- [`opensky_collect.py`](opensky_collect.py) is a live REST collector with
  hard-coded `SELECTED_REGION`, `INTERVAL_SECONDS` and `DURATION_MINUTES` near
  the top of the file. Edit those values before running it; it writes
  `data/raw/opensky/opensky_trajectories.csv`.
- [`opensky_historical_rest.py`](opensky_historical_rest.py) is a one-off REST
  example using `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET` environment
  variables. Its region and relative timestamp are hard-coded, so it is not a
  replacement for the date-ranged Trino downloader.
- [`opensky_plot.py`](opensky_plot.py) is the reusable renderer imported by the
  snapshot, Trino and validation workflows. It is normally not executed
  directly.
- [`opensky_safety.py`](opensky_safety.py) contains the shared state
  classification, look-ahead and CPA/TCPA logic. Its profile is selected by
  callers from `separation_profiles.json`.

## Safety states

The active `ENROUTE` profile uses 5 NM horizontal and 1,000 ft vertical
research thresholds and a 1-, 2- and 5-minute look-ahead sampled every five
seconds. The four states are:

```text
GREEN / NORMAL       outside the warning envelope
YELLOW / PROXIMITY   warning state
ORANGE / HIGH_RISK   near-miss state
RED / CONFLICT       both configured loss-of-separation minima breached
```

The exact dynamic rules and colours are in `separation_profiles.json`. These
values are modelling assumptions for the thesis, not universal regulatory
separation minima.
