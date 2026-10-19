# AI-Based Air Traffic Management

Master's thesis at the [Universidad Nacional de Educación a Distancia
(UNED)](https://www.uned.es/) about the use of aircraft state-vector data and
Artificial Intelligence methods for Air Traffic Management (ATM).

The repository contains research code for acquiring OpenSky data, preparing
regular 10-second traffic snapshots, analysing aircraft separation and
generating figures and quality metrics. It is an academic project, not an
operational air-traffic-control system.

## Thesis

- **Title:** Air Traffic Management Using Artificial Intelligence Models
- **Author:** [Borja Rodríguez Frade](https://orcid.org/0000-0002-9097-1323)
- **Supervisors:** [Félix Hernández del Olmo](https://orcid.org/0000-0002-0567-9572) and [Elena Gaudioso Vázquez](https://orcid.org/0000-0003-2258-6623)
- **Status:** Work in progress
- **Academic year:** 2026–2027

## Installation

Run commands from the repository root. The project currently uses Python 3 and
the packages listed in [`requirements.txt`](requirements.txt):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Some optional acquisition scripts also require the packages they import:

- `download_opensky_sector.py` and `download_opensky_trino_states.py` require
  `pyopensky` and `python-dotenv`.
- `download_enaire_corridors.py` requires `beautifulsoup4`.
- The Trino visualisation workflow requires `imageio` (and an ffmpeg backend if
  MP4 generation is enabled).

Install these only when the corresponding workflow is needed, for example:

```bash
python3 -m pip install pyopensky python-dotenv beautifulsoup4 imageio
```

OpenSky credentials must never be committed. The Trino downloader reads
`src/acquisition/credentials.env` when it exists, or the same variables from
the environment:

```dotenv
OPENSKY_USERNAME=your_username
OPENSKY_PASSWORD=your_password
```

The file is ignored by Git. OpenSky and ENAIRE are external services; respect
their current access, rate-limit and licensing conditions.

## End-to-end historical workflow

The reproducible historical pipeline is:

```text
OpenSky Trino archive
        ↓
data/raw/opensky_sectors/<source-sector>/year=YYYY/month=MM/*.csv
        ↓ preprocess_opensky_10s.py
data/processed/opensky_10s/<analysis-area>/year=YYYY/month=MM/*_10s.csv
        ↓ detect_opensky_conflicts.py
data/processed/opensky_separation/<area>/runs/<run-name>/*.csv
        ↓
quality analysis and validation figures
```

### SCAT workflow

The SCAT dataset is a separate Swedish ATM source with flight JSONs rather
than OpenSky state rows. Its files contain surveillance plots, system-updated
flight plans, controller clearances and predicted trajectories. The SCAT
adapter preserves those time-varying fields while producing the same
chronological 10-second state contract used by the shared safety model:

```text
data/raw/scat/weekly_archives/*.zip
        ↓ preprocess_scat_10s.py
data/processed/scat_10s/<analysis-area>/date=YYYY-MM-DD/*_10s.csv
        ↓ detect_scat_conflicts.py
data/processed/scat_separation/<area>/runs/<run-name>/*.csv
        ↓ generate_scat_validation_snapshots.py
validation_screenshots/random_seed_42/*.png
```

SCAT validation snapshots reuse the OpenSky map style through
`opensky_plot.build_shared_validation_render_config`: by default they use the
same satellite basemap, aircraft labels, state colours, vectors and
separation areas. SCAT-specific flight-plan and controller-clearance fields
remain in the validation footer. Use `--no-basemap` only for offline renders.

Example for the real SCAT local sector envelopes:

```bash
python3 src/preprocessing/preprocess_scat_10s.py \
  --area SCAT_ESMM_GOTHENBURG_GG \
  --overwrite

python3 src/conflict_detection/detect_scat_conflicts.py \
  --area SCAT_ESMM_GOTHENBURG_GG \
  --start "2016-10-15 00:00:00" \
  --end "2016-10-21 23:59:50"
```

Use `src/acquisition/validate_scat_airspace_presets.py` to verify that the
Swedish bbox presets still match the SCAT `airspace.json` shipped inside a
downloaded archive.

For en-route traffic use the sector envelopes
`SCAT_ESMM_GOTHENBURG_ENROUTE`, `SCAT_ESMM_MALMO_ENROUTE`,
`SCAT_ESOS_STOCKHOLM_ENROUTE` or `SCAT_ESOS_CENTRAL_ENROUTE`. The SCAT
detector analyses FL290–FL410 by default; use `--all-flight-levels` only for
terminal-area experiments.

All historical timestamps in the commands below are UTC.

### 1. Download a raw sector archive

The downloader queries one study-area bounding box at a time and writes
monthly CSV files. The end timestamp is **exclusive**.

For a January–February 2019 run:

```bash
python3 src/acquisition/download_opensky_sector.py \
  --sector LECMSAN_ENVELOPE \
  --start "2019-01-01 00:00:00" \
  --end "2019-03-01 00:00:00"
```

Use `--max-windows 1` for a small connectivity test before a large download.
The downloader is resumable and stores its raw archive under
`data/raw/opensky_sectors/LECMSAN_ENVELOPE/`.

The available area keys are defined in
[`src/acquisition/bbox_presets.json`](src/acquisition/bbox_presets.json),
including `LECMSAN_ENVELOPE`, `LECMSAN`, `GALICIA`, `NORTHERN_SPAIN` and
`BARCELONA_50KM`.

### 2. Build regular 10-second snapshots

The preprocessor filters the envelope to the exact analysis polygon, applies
the current cruise-band configuration (FL290–FL410), reconstructs a regular
10-second UTC grid and writes reusable monthly files. Month bounds are
inclusive and must use `YYYY-MM`.

```bash
python3 src/preprocessing/preprocess_opensky_10s.py \
  --source-sector LECMSAN_ENVELOPE \
  --analysis-area LECMSAN \
  --start-month 2019-01 \
  --end-month 2019-02
```

The output is written to `data/processed/opensky_10s/LECMSAN/`. Add
`--overwrite` to regenerate existing monthly outputs, or
`--keep-daily-cache` to retain the resumable daily cache.

### 3. Detect current and predicted separation events

The detector reads only the processed 10-second files. Its end timestamp is
**inclusive**; use a 10-second-aligned value such as `23:59:50`.

```bash
python3 src/conflict_detection/detect_opensky_conflicts.py \
  --area LECMSAN \
  --start "2019-01-01 00:00:00" \
  --end "2019-02-28 23:59:50"
```

The command creates a run directory named
`20190101T000000_20190228T235950` under
`data/processed/opensky_separation/LECMSAN/runs/`. It writes yellow,
orange and red event CSVs, `run_summary.csv` and `daily_summary.csv`.

### 4. Analyse safety quality

Use the detector run name in the next stage. The default analysis label is
`Manual control`; set a different label when comparing scenarios.

```bash
python3 src/conflict_detection/analyze_atc_safety_quality.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --label "Manual control" \
  --profile ENROUTE
```

Results are stored in the run's
`quality_analysis/manual_control/` directory as CSV summaries and PNG/PDF
figures. Use `--no-traffic-scan` when only event files are available or when a
faster event-only analysis is required; this disables aircraft-exposure and
daily normalised metrics.

### 5. Generate random validation screenshots

This samples up to `N` unique current-state moments per colour from an existing
detector run and re-renders the full traffic snapshot.

```bash
python3 src/conflict_detection/generate_random_validation_snapshots.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --profile ENROUTE \
  --n 10 \
  --seed 42
```

Outputs go to `validation_screenshots/random_seed_42/` inside the run. Omit
`--seed` for a new random selection and pass `--n 20` to request up to 20
yellow, 20 orange and 20 red screenshots.

## Snapshot visualisation

[`src/acquisition/opensky_snapshot.py`](src/acquisition/opensky_snapshot.py)
queries the live OpenSky states API and creates one map. It is configured in
the script rather than through command-line arguments. Before running it,
change `AREA_MODE`, `SELECTED_BBOX_PRESET` (for example `GALICIA`) or
`SELECTED_PRESET`/`RADIUS_KM` near the top of the file:

```bash
python3 src/acquisition/opensky_snapshot.py
```

The default outputs are `data/raw/opensky/opensky_snapshot.csv`,
`results/figures/opensky_snapshot_map.png` and the relevant-pairs CSV in
`results/figures/`. Re-running the script replaces these current-snapshot
files, so copy or rename them when preserving a comparison.

## AIP reference data

These commands download current ENAIRE or Swedish LFV eAIP reference data and
filter it to the configured areas. `--source auto` chooses ENAIRE for Spanish
presets and LFV for the `SCAT_*` Swedish presets while preserving existing
entries when `--areas` is used.

```bash
python3 src/acquisition/download_waypoints.py \
  --source auto \
  --areas GALICIA \
  SCAT_ESMM_GOTHENBURG_ENROUTE \
  SCAT_ESMM_MALMO_ENROUTE \
  SCAT_ESOS_STOCKHOLM_ENROUTE \
  SCAT_ESOS_CENTRAL_ENROUTE

python3 src/acquisition/download_enaire_corridors.py \
  --area LECMSAN \
  --output src/acquisition/corridors.json
```

The Swedish source is LFV's official eAIP/AROWeb ENR 4.4. For historical
SCAT or OpenSky comparisons, provide an archived AIRAC HTML document with
`--source-html` or its direct URL with `--source-url`; current AIP data is not
automatically the historical waypoint network.

## Safety model

The default `ENROUTE` profile is configured in
[`src/acquisition/separation_profiles.json`](src/acquisition/separation_profiles.json).
Its research thresholds are 5 NM horizontal and 1,000 ft vertical. The model
also evaluates 1-, 2- and 5-minute constant-kinematics look-ahead intervals at
5-second steps. These are modelling parameters, not universal operational ATC
minima or safety advice.

## Repository structure

```text
data/
├── raw/                 # downloaded/source data; normally not committed
└── processed/           # regularised states and derived events
src/
├── acquisition/         # OpenSky, ENAIRE and map acquisition utilities
├── preprocessing/       # reusable 10-second state preparation
└── conflict_detection/  # event detection, validation and quality analysis
results/
└── figures/             # maps, plots and rendered validation outputs
```

See the README in each directory for the detailed script-level reference:
[`src/README.md`](src/README.md),
[`src/acquisition/README.md`](src/acquisition/README.md),
[`src/preprocessing/README.md`](src/preprocessing/README.md),
[`src/conflict_detection/README.md`](src/conflict_detection/README.md) and
[`results/README.md`](results/README.md).

## Data and licensing

Third-party datasets are not distributed under this repository's license. Raw
OpenSky, ENAIRE and other provider data remain subject to their original terms,
attribution requirements and redistribution restrictions. Do not commit large
raw or generated datasets unless their license and repository policy permit it.
