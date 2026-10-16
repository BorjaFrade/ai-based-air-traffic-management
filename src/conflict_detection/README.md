# OpenSky separation and safety analysis

This directory contains the post-processing stages that operate on the
regularised files in `data/processed/opensky_10s/`. None of these scripts
queries OpenSky directly.

The pipeline is:

```text
processed 10-second states
        ↓ detect_opensky_conflicts.py
yellow/orange/red event CSVs
        ├── analyze_atc_safety_quality.py
        └── generate_random_validation_snapshots.py
```

Run commands from the repository root. Historical timestamps are UTC.

## Safety model

[`../acquisition/separation_profiles.json`](../acquisition/separation_profiles.json)
contains the profiles and exact rules. The detector currently uses
`SEPARATION_PROFILE = "ENROUTE"` from the top of
[`detect_opensky_conflicts.py`](detect_opensky_conflicts.py).

For `ENROUTE`, the research minima are 5 NM horizontal and 1,000 ft vertical.
The detector evaluates the current state and constant-kinematics look-ahead at
1, 2 and 5 minutes, sampled every 5 seconds. The states are:

```text
NORMAL       outside the configured warning conditions
PROXIMITY    yellow warning state
HIGH_RISK    orange near-miss state
CONFLICT     red loss-of-separation state
```

These thresholds and labels are modelling choices for the thesis, not
operational ATC advice or universal regulatory minima.

## 1. Detect events

[`detect_opensky_conflicts.py`](detect_opensky_conflicts.py) reads monthly
10-second CSVs below `data/processed/opensky_10s/<area>/`. The required dates
are parsed as UTC. `--start` and `--end` are both inclusive; use an end value
aligned to the 10-second grid, normally `23:59:50`.

```bash
python3 src/conflict_detection/detect_opensky_conflicts.py \
  --area LECMSAN \
  --start "2019-01-01 00:00:00" \
  --end "2019-01-15 23:59:50"
```

For a complete January–February 2019 run:

```bash
python3 src/conflict_detection/detect_opensky_conflicts.py \
  --area LECMSAN \
  --start "2019-01-01 00:00:00" \
  --end "2019-02-28 23:59:50" \
  --chunksize 100000
```

The available `--area` key must match a processed output directory. Use a
smaller `--chunksize` when memory is limited; the default is 250,000 rows.

The run name is generated from the two timestamps. The example above creates:

```text
data/processed/opensky_separation/LECMSAN/runs/
└── 20190101T000000_20190228T235950/
    ├── yellow_proximity_events.csv
    ├── orange_high_risk_events.csv
    ├── red_loss_of_separation_events.csv
    ├── run_summary.csv
    └── daily_summary.csv
```

Only non-normal event rows are exported. A row remains in the file for the
worst state reached by the pair over the current state and the three
look-ahead intervals. Event CSVs include current geometry, aircraft
kinematics, current and predicted states, CPA/TCPA metrics, minimum separation
metrics and first predicted state occurrences.

The detector does not have an `--overwrite` option. A new date range creates a
new run directory; rerunning the same range recreates that run's CSV files.

## 2. Analyse safety quality

[`analyze_atc_safety_quality.py`](analyze_atc_safety_quality.py) reads the
detector outputs and groups consecutive observations of the same aircraft pair
into encounter episodes. It keeps current-state statistics separate from the
worst predicted event state.

```bash
python3 src/conflict_detection/analyze_atc_safety_quality.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --label "Manual control" \
  --profile ENROUTE
```

The output directory is:

```text
data/processed/opensky_separation/LECMSAN/runs/
└── 20190101T000000_20190228T235950/
    └── quality_analysis/manual_control/
```

Main CSV outputs are:

```text
quality_metrics_summary.csv
state_distribution_pair_current.csv
state_distribution_pair_event.csv
state_distribution_snapshot_current.csv
state_distribution_snapshot_event.csv
episodes.csv
daily_metrics.csv
aircraft_state_distribution_current.csv  # when traffic scan is enabled
```

Each figure is saved as both PNG and PDF. By default, the script also scans
`data/processed/opensky_10s/` to calculate aircraft exposure and daily
normalised metrics. Skip that potentially expensive step with:

```bash
python3 src/conflict_detection/analyze_atc_safety_quality.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --label "Manual control" \
  --no-traffic-scan
```

Other controls are `--episode-gap-s 60`, `--snapshot-step-s 10` and
`--chunksize 300000`. Use the same `--profile` value as the detector's active
profile when comparing metrics.

## 3. Generate random validation screenshots

[`generate_random_validation_snapshots.py`](generate_random_validation_snapshots.py)
selects up to `N` unique current-state timestamps per non-normal colour from an
existing detector run, reloads the full traffic snapshot and renders the
selected aircraft pair with the shared map renderer.

```bash
python3 src/conflict_detection/generate_random_validation_snapshots.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --profile ENROUTE \
  --n 10 \
  --seed 42
```

`--n 10` means up to 10 yellow, 10 orange and 10 red images, not 10 images in
total. Omit `--seed` for a different selection on each run. The output is:

```text
.../20190101T000000_20190228T235950/
└── validation_screenshots/
    └── random_seed_42/
        ├── selected_random_moments.csv
        ├── validation_report.csv
        └── *.png
```

The PNG dimensions are read from the selected area's `render` configuration in
`bbox_presets.json`. The script also overlays waypoints and ATS corridors when
those reference files are available.

## Input contract

The detector expects the preprocessor's monthly output, including at least:

```text
icao24, timestamp/state time, latitude, longitude,
baro_altitude or baro_altitude_ft, velocity or ground_speed,
true_track, vertical_rate
```

The preprocessor is the recommended producer because it normalises timestamps,
filters the exact polygon and preserves the metadata needed by the detector.
Do not feed raw Trino CSVs directly to this directory.
