# Data acquisition

This folder contains the OpenSky data-acquisition and visualisation scripts used
in the UNED Master's Thesis **AI-Based Air Traffic Management**. They are
research tools, not operational ATC software.

## Current research configuration

The initial en-route experiment is configured for the RVSM cruise band:

```python
FILTER_CRUISE_FL_BAND = True
CRUISE_FL_MIN = 290
CRUISE_FL_MAX = 410
```

Vertical separation is evaluated using **barometric altitude**. The baseline
en-route minima remain:

```text
Horizontal: 5 NM
Vertical:   1000 ft
```

A confirmed loss of separation requires **both** minima to be infringed
simultaneously.

The warning envelope is experimental and deliberately separate from the
regulatory minima. The default is:

```python
WARNING_FACTOR = 1.50
```

so the initial warning envelope is 7.5 NM / 1500 ft. Values such as 1.25 can
be tested later as part of the sensitivity analysis.

## Four safety states

The visualisation now uses four clearly separated colours:

```text
GREEN   NORMAL       outside the warning envelope
YELLOW  PROXIMITY    inside the warning envelope, minima still respected
ORANGE  HIGH_RISK    one minimum infringed and the other inside the warning envelope
RED     CONFLICT     horizontal < 5 NM AND vertical < 1000 ft
```

Default colours are:

```text
GREEN   #00A651
YELLOW  #FFD400
ORANGE  #FF6F00
RED     #D7191C
```

The aircraft symbol and its safety circle always represent the **current**
state.

## Multi-horizon look-ahead

The old single trajectory arrow has been replaced by a segmented trajectory:

```python
LOOKAHEAD_MINUTES = [1, 2, 5]
PREDICTION_STEP_SECONDS = 5
```

The three segments represent:

```text
0-1 min
1-2 min
2-5 min
```

Each segment has its own arrowhead and is coloured with the **worst state
predicted anywhere inside that interval**, not only at the interval endpoint.

Prediction uses the current position, true track, ground speed and vertical
rate under a constant-kinematics assumption.

Current pairwise status is still calculated from the measured snapshot. The
future look-ahead is a research prediction layer.

## CPA / TCPA

`opensky_safety.py` also estimates:

```text
cpa_nm
tcpa_s
vertical_sep_at_cpa_ft
```

TCPA is first estimated with a local relative-motion model and the CPA distance
is then recomputed geodesically.

Historical source snapshots can remain at their native/source-oriented cadence
(e.g. 10 s), while the look-ahead safety calculation uses the configurable
5-second prediction step. This avoids unnecessarily multiplying the stored
historical dataset.

## Separation-event exports

`opensky_snapshot.py` writes the relevant pairwise cases to:

```text
results/figures/opensky_snapshot_separation.csv
```

The Trino frame workflow writes relevant current or predicted cases to:

```text
.../separation_events.csv
```

Typical fields include:

```text
icao24_a
icao24_b
callsign_a
callsign_b
horizontal_sep_nm
vertical_sep_ft
state_current
status_current
cpa_nm
tcpa_s
vertical_sep_at_cpa_ft
worst_status_0_1min
worst_status_1_2min
worst_status_2_5min
predicted_conflict
```

Only pairs that are non-normal now or in at least one look-ahead interval are
kept in the dedicated events export.

## Study-area presets

Study areas are shared through `bbox_presets.json`.

Three area types are supported.

### Rectangular area

```json
{
  "MY_RECTANGLE": {
    "name": "My rectangle",
    "type": "vertices",
    "vertices": {
      "lamin": 41.75,
      "lamax": 43.85,
      "lomin": -9.40,
      "lomax": -6.65,
      "extended": 1
    }
  }
}
```

### Centre + radius

```json
{
  "MY_CIRCLE": {
    "name": "My centre-radius area",
    "type": "center",
    "center": {
      "lat": 41.2974,
      "lon": 2.0833
    },
    "radius_km": 50
  }
}
```

### Arbitrary polygon

```json
{
  "MY_SECTOR": {
    "name": "ATC sector polygon",
    "type": "polygon",
    "polygon": {
      "vertices": [
        {"lat": 43.10, "lon": -9.20},
        {"lat": 43.70, "lon": -8.10},
        {"lat": 43.20, "lon": -6.90},
        {"lat": 42.20, "lon": -7.10},
        {"lat": 42.00, "lon": -8.60}
      ]
    },
    "render": {
      "width_px": 3840,
      "height_px": 2160
    }
  }
}
```

The coordinates above are only a **schema example**, not an operational sector.

For a polygon preset, OpenSky/Trino is queried with the smallest enclosing
rectangular BBOX, then the downloaded states are clipped **locally to the exact
polygon**. This keeps database queries efficient while allowing real ATC-sector
geometries to be used for the scientific analysis.

`bbox_presets.json` contains a disabled `_POLYGON_TEMPLATE`; keys beginning
with `_` and presets with `"enabled": false` are ignored by the scripts.

## Shared helper modules

### `area_presets.py`

Resolves rectangular, centre-radius and polygon presets. It also provides the
local point-in-polygon filter used by the snapshot, Trino, weekly downloader,
waypoint downloader, live collector and REST example.

### `opensky_safety.py`

Contains the common four-state pairwise classification, constant-kinematics
look-ahead, CPA/TCPA calculation and relevant-event filtering. This keeps the
live snapshot and historical Trino workflow on the same safety logic.

## `opensky_snapshot.py`

Downloads one current state-vector snapshot, saves the raw response, filters the
analysis set, performs current + predicted pairwise analysis and generates the
map.

The raw snapshot remains:

```text
data/raw/opensky/opensky_snapshot.csv
```

The map remains:

```text
results/figures/opensky_snapshot_map.png
```

For polygon presets, the exact polygon is also drawn on the map and aircraft
outside it are excluded from the analysis.

## `opensky_plot.py`

Reusable map renderer. It draws the satellite basemap, cities, aircraft,
current safety circles, labels and the three coloured look-ahead trajectory
segments.

## `download_opensky_trino_states.py`

Queries OpenSky Trino sequentially in short temporal blocks. A polygon study
area is sent to Trino as its enclosing BBOX and clipped locally after retrieval.

The raw historical CSV is downloaded once. Frames and video are generated
locally from that data; Trino is **not** queried once per visual frame.

The visual snapshot cadence remains independently configurable with:

```python
FRAME_STEP_SECONDS = 10
```

while look-ahead analysis uses:

```python
PREDICTION_STEP_SECONDS = 5
```

## `download_opensky_monday_states.py`

Downloads the OpenSky Weekly 24 Hours scientific files and filters each global
hourly source file locally. Study areas now come from the same
`bbox_presets.json`, including polygon presets.

The source dataset itself has its published cadence; the downloader does not
invent intermediate state vectors.

## `download_waypoints.py`

Downloads the public waypoint source and applies the same exact study-area
filter. For a polygon preset, only waypoints inside the polygon are retained.

Regenerate after adding a new polygon:

```bash
python3 src/acquisition/download_waypoints.py --areas MY_SECTOR
```

## `opensky_collect.py`

Live collection now uses the shared study-area presets and defaults to a
5-second polling interval. Polygon results are clipped locally after the
rectangular OpenSky REST request.

Always check current OpenSky rate limits before a long live collection.

## `opensky_historical_rest.py`

Historical REST demonstration. It also uses the shared study-area resolver, so a
polygon preset can be selected without editing raw latitude/longitude limits.

## Deferred work

The following ideas remain deliberately outside this code revision:

- synthetic aircraft/conflict injection;
- AI training and conflict-resolution policy;
- training on one sector and validation on different sectors;
- airport SID/STAR/IAC optimisation;
- dynamic ATC sector handover logic.

The new polygon preset architecture is intended to make a later real-sector
implementation straightforward without hard-coding administrative regions.

## Dependencies

In addition to the project requirements, the historical Trino workflow requires
`pyopensky`, `python-dotenv`, `imageio` and a working ffmpeg backend for MP4
generation.

Do not commit OpenSky credentials. `credentials.env` should remain outside
version control.


## Waypoint overlay

The traffic maps can now plot the aviation waypoints stored in
[`waypoints.json`](waypoints.json).

Enable them with:

```python
SHOW_WAYPOINTS = True
SHOW_WAYPOINT_LABELS = True
```

Waypoints are drawn as **white squares with a black edge**, which visually
distinguishes them from the **white circular city markers**. Their names use
the same collision-aware label-placement system used for cities and aircraft.

The main style controls are:

```python
WAYPOINT_MARKER = "s"
WAYPOINT_MARKER_SIZE = 34
WAYPOINT_FACE_COLOR = "white"
WAYPOINT_EDGE_COLOR = "black"
WAYPOINT_EDGE_WIDTH = 0.9
WAYPOINT_LABEL_COLOR = "white"
```

`opensky_snapshot.py` and `download_opensky_trino_states.py` load the
waypoints once for the selected study area and pass them to
`opensky_plot.render_map()`. For polygon presets, the waypoint set is also
clipped to the exact polygon.

If an old `waypoints.json` uses a different top-level key from the current
`bbox_presets.json`, the shared loader falls back to deduplicating the
available waypoints and filtering them geometrically to the active study area.

When a waypoint name cannot be placed without overlapping another label or an
aircraft symbol, its square marker is still drawn and only the text label is
skipped.
