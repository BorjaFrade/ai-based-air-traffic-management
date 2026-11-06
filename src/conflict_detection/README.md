# OpenSky separation and safety analysis

This directory contains local OpenSky analysis and rendering stages. The
detector and validation screenshots use regularised files in
`data/processed/opensky_10s/`; the sector-video script can also reconstruct
frames directly from downloaded raw envelope CSVs. None of these scripts
queries OpenSky directly.

The pipeline is:

```text
processed 10-second states
        ↓ detect_opensky_conflicts.py
yellow/orange/red event CSVs
        ├── analyze_atc_safety_quality.py
        └── generate_random_validation_snapshots.py

raw envelope CSVs or processed 10-second states
        └── render_opensky_sector_video.py → PNG frames → MP4
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

## SCAT detector

[`detect_scat_conflicts.py`](detect_scat_conflicts.py) consumes the daily
files produced by [`../preprocessing/preprocess_scat_10s.py`](../preprocessing/preprocess_scat_10s.py):

```bash
python3 src/conflict_detection/detect_scat_conflicts.py \
  --area SCAT_ESMM_GOTHENBURG_ENROUTE \
  --start "2016-10-15 00:00:00" \
  --end "2016-10-21 23:59:50"
```

SCAT event CSVs use the same yellow/orange/red safety states and look-ahead
metrics as OpenSky, but add per-aircraft context columns for the linked flight
plan and latest clearance, including CFL, assigned heading/speed, COPN/COPX,
route, active centre and predicted next fix.

The detector analyses FL290–FL410 by default. This avoids mixing terminal
traffic from the local `GG`, `MS` and `SE` sectors with the en-route conflict
experiment. Use `--all-flight-levels` for a terminal-area run, or set
`--min-fl` and `--max-fl` explicitly.

The output is written below
`data/processed/scat_separation/<AREA>/runs/<RUN_NAME>/`.

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

For SCAT use:

```bash
python3 src/conflict_detection/generate_scat_validation_snapshots.py \
  --area SCAT_ESMM_GOTHENBURG_GG \
  --run-name 20161015T000000_20161021T235950 \
  --n 10 \
  --seed 42
```

The renderer recalculates the selected pair and writes PNGs, a selected
moments CSV and `validation_report.csv`. SCAT consumes the same shared
`opensky_plot.py` visual configuration as OpenSky: satellite basemap, city and
aircraft labels, state colours, trajectory vectors and safety areas. The SCAT
route and current clearance context are added only as a validation footer.
Add `--no-basemap` for offline or reproducible rendering without tile
downloads.

## 4. Render a local sector video

[`render_opensky_sector_video.py`](render_opensky_sector_video.py) reads local
CSV files, re-evaluates the shared `ENROUTE` safety states at each frame,
draws trajectory endpoints at 2, 5, 10 and 20 minutes, then joins the PNG
frames into an H.264 MP4. The segment colours describe the worst predicted
state over 0–2, 2–5, 5–10 and 10–20 minutes. It does not need an existing
detector run or an OpenSky connection.
Waypoints and their labels remain visible. ATS corridors and corridor labels
are hidden by default to keep the map readable; add `--show-corridors` only
when those route overlays are needed.

With `--area LECMSAN_ENVELOPE`, the script uses the downloaded raw files under
`data/raw/opensky_sectors/LECMSAN_ENVELOPE/` and reconstructs 10-second
snapshots for the full rectangular envelope. It applies no LECMSAN polygon
filter. However, the existing 2019 raw CSVs were downloaded with FL290–410,
airborne-only and maximum 15-second last-contact-age filters. The video cannot
show traffic that was excluded during download. It uses the hourly
download cache when all needed hours are present, otherwise the monthly raw
CSV. With `--area LECMSAN`, it uses
the already processed CSVs in `data/processed/opensky_10s/LECMSAN/`, which
are limited to the sector polygon and FL290–410. `--source raw` or
`--source processed` can override this automatic choice.

For the complete UTC day of 15 January 2019:

```bash
python3 src/conflict_detection/render_opensky_sector_video.py \
  --area LECMSAN_ENVELOPE \
  --start "2019-01-15 00:00:00" \
  --end "2019-01-16 00:00:00" \
  --frame-step 10 --fps 10
```

The end is exclusive. This requests 8,640 frames and a 14-minute-24-second
video. Frames, `video.mp4`, `run_config.json` and `summary.json` go to
`results/opensky_sector_video/LECMSAN_ENVELOPE/20190115T000000_20190116T000000/`.
The map preset uses 3840 × 2160 pixels; a full day can take substantial time
and disk space. If interrupted, rerun the same command with `--resume`. For a
quick local check, use `--max-frames 3 --width 640 --height 360 --no-basemap`
and a separate `--out` directory. `--no-basemap` avoids map tile downloads.
The default satellite background uses Esri World Imagery tiles and needs
network access on the first frame. If the tiles cannot be fetched, rendering
stops instead of silently producing a dark-background video.
The terminal reports CSV chunks, the current UTC frame, completed and reused
PNGs, MP4 assembly progress, rendering rate and estimated time remaining.
Use `--log-every 10` for more frequent PNG/MP4 updates (default: 25 frames).
If a crowded frame cannot place all text labels without collisions, the
renderer retries only that frame without text labels while retaining aircraft
icons, trajectory vectors and safety colours. These instants are recorded in
`label_fallback_frames.json`; the video run continues. Existing PNGs are
reused with `--resume`.

The renderer writes `traffic_coverage.csv` with observed and estimated
aircraft counts for every frame. `track_gaps.csv` lists gaps for each aircraft,
including the last position and altitude before a track ends and the number
of frames filled. To inspect the source without making a video, add
`--audit-only` and select a separate `--out` directory.

For a gap bounded by two states of the same flight, the video interpolates a
geodesic position and barometric altitude at each 10-second frame. It derives
track, ground speed and vertical rate from the two endpoints. These estimates
enter the separation and safety-colour calculations and are drawn like other
aircraft; `state_source=GAP_ESTIMATE` remains in the internal data for audit.
The renderer reads up to one maximum-gap interval before and after the selected
video range so a gap crossing the first or last frame can also be filled when
both endpoint observations exist. This is retrospective interpolation and
uses later data; the resulting safety colours are estimates, not measurements.
The default maximum interval between endpoints is 30 minutes; change it with
`--max-interpolation-gap-seconds` (the older `--bridge-gap-seconds` name is
also accepted). Gaps with a changed callsign, implausible travel speed or
missing altitude are left empty. An aircraft with no later observation is
removed after its last valid state: no last-known marker or extrapolated
aircraft is drawn. A newly acquired position may still cause a visible jump
because the interpolated path is only an estimate. Use a new output directory
when changing these options.

The 2019 envelope CSVs cannot supply traffic outside FL290–410. To acquire
other airborne flight levels for a new video, rerun the downloader with
`--all-altitudes` and a bounded UTC interval. It stores the result separately
under `data/raw/opensky_sectors_all_altitudes/`; then pass that directory as
`--input-root` to this video script. This requires a new OpenSky Trino query.
The four-hour interval from 00:00 to 04:00 UTC on 15 January 2019, with one
hour of input context on each side, and the full UTC day of 23 June 2019 are
already available there. The January download improves coverage, but UAL91
has no new position after 02:09:42 UTC, while
BAW245 has a roughly 14-minute gap between fresh positions starting around
00:09:41 UTC. A direct Trino check without the 15-second contact-age filter
returned stale rows for up to five minutes, repeating the same coordinates;
these rows must not be treated as updated aircraft positions.

For the smaller polygon-only video, use `--area LECMSAN` instead. Its
processed directory currently contains January and February 2019. To
prepare another polygon-filtered month without querying OpenSky, run:

```bash
python3 src/preprocessing/preprocess_opensky_10s.py \
  --source-sector LECMSAN_ENVELOPE --analysis-area LECMSAN \
  --start-month 2019-03 --end-month 2019-03
```

## Input contract

The detector expects the preprocessor's monthly output, including at least:

```text
icao24, timestamp/state time, latitude, longitude,
baro_altitude or baro_altitude_ft, velocity or ground_speed,
true_track, vertical_rate
```

The preprocessor is the recommended producer for the detector because it
normalises timestamps, filters the exact polygon and preserves the metadata
needed by that analysis. The video script may read the raw envelope CSVs
directly and applies its own 10-second reconstruction.
