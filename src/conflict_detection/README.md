# OpenSky Separation Event Detector

`detect_opensky_separation_events_v2.py` analyses the monthly OpenSky sector
archive created under:

```text
data/raw/opensky_sectors/<BBOX_PRESET>/
```

and detects current or predicted pairwise separation problems.

The detector uses the same safety logic as `opensky_safety.py` and is intended
for the UNED Master's Thesis:

> **Air Traffic Management Using Artificial Intelligence Models**

It does **not** query OpenSky or Trino. It only reads previously downloaded CSV
files and produces derived scientific event tables.

---

## Safety states

The detector uses four states:

```text
GREEN   NORMAL
YELLOW  PROXIMITY
ORANGE  HIGH_RISK
RED     CONFLICT / LOSS OF SEPARATION
```

For the default `ENROUTE` profile:

```text
Minimum horizontal separation: 5 NM
Minimum vertical separation:   1000 ft
```

The experimental warning envelope is:

```python
WARNING_FACTOR = 1.50
```

therefore:

```text
Warning horizontal: 7.5 NM
Warning vertical:   1500 ft
```

The classification is:

```text
NORMAL
    outside the warning envelope

PROXIMITY
    inside the warning envelope
    but no regulatory minimum is infringed

HIGH_RISK
    inside the warning envelope
    and exactly one minimum is infringed

CONFLICT
    horizontal < 5 NM
    AND
    vertical < 1000 ft
```

Vertical separation is based on **barometric altitude**.

---

## Look-ahead

The detector evaluates:

```text
current state
0-1 min
1-2 min
2-5 min
```

The prediction uses the constant-kinematics model already implemented in
`opensky_safety.py`:

```text
current latitude / longitude
ground speed
track
vertical rate
```

and samples the future trajectory every:

```python
PREDICTION_STEP_SECONDS = 5
```

Each interval stores the worst state reached anywhere in that interval.

---

## Input structure

The detector expects the archive layout created by
`download_opensky_sector_archive.py`.

Example:

```text
data/raw/opensky_sectors/
└── GALICIA/
    ├── year=2024/
    │   ├── month=01/
    │   │   └── galicia_2024-01.csv
    │   ├── month=02/
    │   │   └── galicia_2024-02.csv
    │   └── ...
    └── year=2025/
```

The value supplied through:

```bash
--sector GALICIA
```

is the folder / BBOX-preset key.

---

## Output structure

Derived event files are written to:

```text
data/processed/opensky_separation/
```

with the same sector/year/month hierarchy:

```text
data/processed/opensky_separation/
└── GALICIA/
    └── year=2025/
        └── month=01/
            ├── galicia_2025-01_separation_proximity_events.csv
            ├── galicia_2025-01_separation_high_risk_events.csv
            └── galicia_2025-01_loss_of_separation_events.csv
```

The three files are:

```text
separation_proximity_events
    yellow / PROXIMITY

separation_high_risk_events
    orange / HIGH_RISK

loss_of_separation_events
    red / CONFLICT
```

A pair/timestamp is stored in the file corresponding to the **worst state**
reached now or during the 5-minute look-ahead.

For example, if the pair is currently yellow but becomes red in 3 minutes, the
row is written to:

```text
loss_of_separation_events.csv
```

while the row itself still records the complete state timeline.

---

## Basic usage

Analyse all available months for one sector:

```bash
python3 src/acquisition/detect_opensky_separation_events_v2.py \
    --sector GALICIA
```

Analyse one year:

```bash
python3 src/acquisition/detect_opensky_separation_events_v2.py \
    --sector GALICIA \
    --start-month 2025-01 \
    --end-month 2025-12
```

Analyse one month:

```bash
python3 src/acquisition/detect_opensky_separation_events_v2.py \
    --sector GALICIA \
    --start-month 2025-06 \
    --end-month 2025-06
```

Regenerate existing outputs:

```bash
python3 src/acquisition/detect_opensky_separation_events_v2.py \
    --sector GALICIA \
    --start-month 2025-06 \
    --end-month 2025-06 \
    --overwrite
```

---

# Main event data

Each event row contains the pair identity and current geometry:

```text
sector
timestamp
timestamp_utc

icao24_a
callsign_a
lat_a
lon_a
baro_altitude_ft_a

icao24_b
callsign_b
lat_b
lon_b
baro_altitude_ft_b

horizontal_sep_nm
vertical_sep_ft
```

It also stores the state timeline:

```text
state_current
worst_status_0_1min
worst_status_1_2min
worst_status_2_5min

event_state
event_colour
event_timeline
first_alert_window
first_worst_state_window
```

---

# Aircraft kinematics

For each aircraft the event row stores current horizontal and vertical motion.

Aircraft A:

```text
ground_speed_ms_a
ground_speed_kt_a
true_track_deg_a
vertical_rate_ms_a
vertical_rate_fpm_a
```

Aircraft B:

```text
ground_speed_ms_b
ground_speed_kt_b
true_track_deg_b
vertical_rate_ms_b
vertical_rate_fpm_b
```

This is useful because two pairs with the same current separation can have very
different future risk depending on their relative motion.

---

# Relative-motion metrics

The detector also calculates:

```text
relative_horizontal_speed_kt
horizontal_closing_rate_kt
vertical_closing_rate_fpm
```

Interpretation:

### `relative_horizontal_speed_kt`

Magnitude of the horizontal relative-velocity vector.

It is not the same as closing rate.

Two aircraft can have a high relative speed while moving mostly sideways with
respect to the current line of sight.

### `horizontal_closing_rate_kt`

Current rate at which horizontal range is changing.

```text
positive -> aircraft are closing
negative -> aircraft are opening
```

### `vertical_closing_rate_fpm`

Current rate at which absolute vertical separation is changing.

```text
positive -> vertical separation is decreasing
negative -> vertical separation is increasing
```

These fields are particularly useful for later machine-learning features.

---

# CPA / TCPA

The existing safety model already provides:

```text
cpa_nm
tcpa_s
vertical_sep_at_cpa_ft
```

`tcpa_s` is the predicted time to the horizontal closest point of approach
within the configured 5-minute horizon.

`cpa_nm` is the horizontal separation at that point.

`vertical_sep_at_cpa_ft` records the simultaneous vertical separation.

This is important because horizontal and vertical minima do not necessarily
occur at the same time.

---

# Minimum separation in each look-ahead interval

The V2 detector additionally calculates independent horizontal and vertical
minima for each interval.

## 0-1 minute

```text
min_horizontal_0_1min_nm
time_to_min_horizontal_0_1min_s
vertical_sep_at_min_horizontal_0_1min_ft

min_vertical_0_1min_ft
time_to_min_vertical_0_1min_s
horizontal_sep_at_min_vertical_0_1min_nm
```

## 1-2 minutes

```text
min_horizontal_1_2min_nm
time_to_min_horizontal_1_2min_s
vertical_sep_at_min_horizontal_1_2min_ft

min_vertical_1_2min_ft
time_to_min_vertical_1_2min_s
horizontal_sep_at_min_vertical_1_2min_nm
```

## 2-5 minutes

```text
min_horizontal_2_5min_nm
time_to_min_horizontal_2_5min_s
vertical_sep_at_min_horizontal_2_5min_ft

min_vertical_2_5min_ft
time_to_min_vertical_2_5min_s
horizontal_sep_at_min_vertical_2_5min_nm
```

---

## Why both dimensions are stored at each minimum

The horizontal minimum and vertical minimum are calculated independently.

For example:

```text
t = 100 s
horizontal separation = 3.8 NM
vertical separation   = 1800 ft

t = 180 s
horizontal separation = 7.2 NM
vertical separation   = 600 ft
```

The pair has:

```text
minimum horizontal = 3.8 NM
minimum vertical   = 600 ft
```

but those values **do not occur simultaneously**.

It would therefore be scientifically wrong to combine them and conclude that
the pair necessarily reaches:

```text
3.8 NM / 600 ft
```

For this reason, the CSV also stores:

```text
vertical_sep_at_min_horizontal_...
horizontal_sep_at_min_vertical_...
```

so the geometry at each minimum remains physically interpretable.

---

# Global 0-5 minute minima

The detector also records the best/worst geometric approach over the complete
look-ahead:

```text
min_horizontal_0_5min_nm
time_to_min_horizontal_0_5min_s
timestamp_min_horizontal_0_5min_utc
vertical_sep_at_min_horizontal_0_5min_ft

min_vertical_0_5min_ft
time_to_min_vertical_0_5min_s
timestamp_min_vertical_0_5min_utc
horizontal_sep_at_min_vertical_0_5min_nm
```

These fields make it easy to answer:

```text
How close will the pair get horizontally in the next five minutes?

When will that happen?

What will the vertical separation be at exactly that moment?

What is the minimum vertical separation in the next five minutes?

When will it occur?

What horizontal separation will exist at that exact moment?
```

---

# First predicted safety-state occurrence

For each pair the detector searches the 5-second prediction grid for the first
occurrence of each non-normal state:

```text
first_proximity_s
first_proximity_timestamp_utc

first_high_risk_s
first_high_risk_timestamp_utc

first_conflict_s
first_conflict_timestamp_utc
```

For example:

```text
first_proximity_s = 40
first_high_risk_s = 85
first_conflict_s  = 150
```

means:

```text
+40 s  -> enters yellow
+85 s  -> enters orange
+150 s -> enters red
```

under the current constant-kinematics prediction.

These variables are useful targets/features for later predictive models.

---

# Example conceptual event row

A pair could contain:

```text
timestamp_utc = 2025-06-01T12:30:00+00:00

callsign_a = IBE1234
callsign_b = RYR5678

horizontal_sep_nm = 8.1
vertical_sep_ft   = 1700

ground_speed_kt_a = 455
ground_speed_kt_b = 448

vertical_rate_fpm_a = 0
vertical_rate_fpm_b = -900

horizontal_closing_rate_kt = 720
vertical_closing_rate_fpm  = 900

state_current       = NORMAL
worst_status_0_1min = PROXIMITY
worst_status_1_2min = HIGH_RISK
worst_status_2_5min = CONFLICT

min_horizontal_0_5min_nm = 2.9
time_to_min_horizontal_0_5min_s = 210
vertical_sep_at_min_horizontal_0_5min_ft = 700

min_vertical_0_5min_ft = 500
time_to_min_vertical_0_5min_s = 260
horizontal_sep_at_min_vertical_0_5min_nm = 4.2

first_proximity_s = 45
first_high_risk_s = 95
first_conflict_s = 175
```

That single row contains much more information than a simple conflict flag.

---

# Prediction assumptions

All future metrics are predictions based on the current state vector.

The current model assumes constant:

```text
ground speed
track
vertical rate
```

for the next five minutes.

Therefore:

```text
current separation
current velocity
```

are measured OpenSky-derived quantities, while:

```text
look-ahead state
future minimum separation
future threshold-entry time
CPA/TCPA
```

are model-derived predictions.

This distinction should be preserved in the thesis.

---

# Streaming large monthly files

The detector does not load an entire month into memory.

By default:

```python
READ_CHUNK_ROWS = 250_000
```

The monthly CSV is streamed in chunks.

The detector buffers complete OpenSky hour partitions before splitting them
into timestamps. This prevents a snapshot from being accidentally split at a
pandas chunk boundary.

---

# Existing-output behaviour

By default:

```python
OVERWRITE_EXISTING = False
```

If all three monthly event CSVs already exist, that month is skipped.

Use:

```bash
--overwrite
```

when the detector configuration changes and the month must be recalculated.

This is especially important after changing:

```text
WARNING_FACTOR
PREDICTION_STEP_SECONDS
separation profile
look-ahead implementation
```

---

# Scientific interpretation

The three derived datasets have different meanings.

## Proximity

Useful for studying:

```text
traffic density
near interactions
early warning
precursors to conflict
```

## High risk

Useful for:

```text
one-dimension minimum infringements
transition from warning to actual risk
candidate controller intervention points
```

## Loss of separation

Useful for:

```text
confirmed simultaneous separation-minimum infringements
rare-event analysis
conflict-detection validation
```

For machine-learning work, it may later be useful to combine all three files
into a single event table while preserving `event_state` as the label.

---

# Files required

Place these in the same acquisition directory:

```text
detect_opensky_separation_events_v2.py
opensky_safety.py
separation_profiles.json
```

The monthly OpenSky sector archive must already exist under:

```text
data/raw/opensky_sectors/
```

---

# Recommended workflow

```text
download sector archive
        ↓
verify monthly CSV
        ↓
run separation detector
        ↓
inspect yellow/orange/red event CSVs
        ↓
validate a sample visually
        ↓
aggregate events
        ↓
build ML / statistical experiments
```

Before using the generated events as thesis labels, manually inspect a sample
of detections against the map/video visualisation to verify the end-to-end
pipeline.
