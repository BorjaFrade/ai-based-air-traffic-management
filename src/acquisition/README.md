**# Data acquisition**



This folder contains the OpenSky data-acquisition and visualisation scripts used

in the UNED Master's Thesis **\*\*AI-Based Air Traffic Management\*\***. They include live snapshots, historical visualisation workflows and long historical sector archives.

They are research tools, not operational ATC software.



**## Current research configuration**



The initial en-route experiment is configured for the RVSM cruise band:



\`\`\`python

FILTER_CRUISE_FL_BAND = True

CRUISE_FL_MIN = 290

CRUISE_FL_MAX = 410

\`\`\`



Vertical separation is evaluated using **\*\*barometric altitude\*\***. The baseline

en-route minima remain:



\`\`\`text

Horizontal: 5 NM

Vertical:   1000 ft

\`\`\`



A confirmed loss of separation requires **\*\*both\*\*** minima to be infringed

simultaneously.



The warning envelope is experimental and deliberately separate from the

regulatory minima. The default is:



\`\`\`python

WARNING_FACTOR = 1.50

\`\`\`



so the initial warning envelope is 7.5 NM / 1500 ft. Values such as 1.25 can

be tested later as part of the sensitivity analysis.



**## Four safety states**



The visualisation now uses four clearly separated colours:



\`\`\`text

GREEN   NORMAL       outside the warning envelope

YELLOW  PROXIMITY    inside the warning envelope, minima still respected

ORANGE  HIGH_RISK    one minimum infringed and the other inside the warning envelope

RED     CONFLICT     horizontal < 5 NM AND vertical < 1000 ft

\`\`\`



Default colours are:



\`\`\`text

GREEN   #00A651

YELLOW  #FFD400

ORANGE  #FF6F00

RED     #D7191C

\`\`\`



The aircraft symbol and its safety circle always represent the **\*\*current\*\***

state.



**## Multi-horizon look-ahead**



The old single trajectory arrow has been replaced by a segmented trajectory:



\`\`\`python

LOOKAHEAD_MINUTES = [1, 2, 5]

PREDICTION_STEP_SECONDS = 5

\`\`\`



The three segments represent:



\`\`\`text

0-1 min

1-2 min

2-5 min

\`\`\`



Each segment has its own arrowhead and is coloured with the \*\*worst state

predicted anywhere inside that interval\*\*, not only at the interval endpoint.



Prediction uses the current position, true track, ground speed and vertical

rate under a constant-kinematics assumption.



Current pairwise status is still calculated from the measured snapshot. The

future look-ahead is a research prediction layer.



**## CPA / TCPA**



\`opensky_safety.py\` also estimates:



\`\`\`text

cpa_nm

tcpa_s

vertical_sep_at_cpa_ft

\`\`\`



TCPA is first estimated with a local relative-motion model and the CPA distance

is then recomputed geodesically.



Historical source snapshots can remain at their native/source-oriented cadence

(e.g. 10 s), while the look-ahead safety calculation uses the configurable

5-second prediction step. This avoids unnecessarily multiplying the stored

historical dataset.



**## Separation-event exports**



\`opensky_snapshot.py\` writes the relevant pairwise cases to:



\`\`\`text

results/figures/opensky_snapshot_separation.csv

\`\`\`



The Trino frame workflow writes relevant current or predicted cases to:



\`\`\`text

.../separation_events.csv

\`\`\`



Typical fields include:



\`\`\`text

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

\`\`\`



Only pairs that are non-normal now or in at least one look-ahead interval are

kept in the dedicated events export.



**## Study-area presets**



Study areas are shared through \`bbox_presets.json\`.



Three area types are supported.



**### Rectangular area**



\`\`\`json

{

&#x20; "MY_RECTANGLE": {

&#x20;   "name": "My rectangle",

&#x20;   "type": "vertices",

&#x20;   "vertices": {

&#x20;     "lamin": 41.75,

&#x20;     "lamax": 43.85,

&#x20;     "lomin": -9.40,

&#x20;     "lomax": -6.65,

&#x20;     "extended": 1

&#x20;   }

&#x20; }

}

\`\`\`



**### Centre + radius**



\`\`\`json

{

&#x20; "MY_CIRCLE": {

&#x20;   "name": "My centre-radius area",

&#x20;   "type": "center",

&#x20;   "center": {

&#x20;     "lat": 41.2974,

&#x20;     "lon": 2.0833

&#x20;   },

&#x20;   "radius_km": 50

&#x20; }

}

\`\`\`



**### Arbitrary polygon**



\`\`\`json

{

&#x20; "MY_SECTOR": {

&#x20;   "name": "ATC sector polygon",

&#x20;   "type": "polygon",

&#x20;   "polygon": {

&#x20;     "vertices": [

&#x20;       {"lat": 43.10, "lon": -9.20},

&#x20;       {"lat": 43.70, "lon": -8.10},

&#x20;       {"lat": 43.20, "lon": -6.90},

&#x20;       {"lat": 42.20, "lon": -7.10},

&#x20;       {"lat": 42.00, "lon": -8.60}

&#x20;     ]

&#x20;   },

&#x20;   "render": {

&#x20;     "width_px": 3840,

&#x20;     "height_px": 2160

&#x20;   }

&#x20; }

}

\`\`\`



The coordinates above are only a **\*\*schema example\*\***, not an operational sector.



For a polygon preset, OpenSky/Trino is queried with the smallest enclosing

rectangular BBOX, then the downloaded states are clipped \*\*locally to the exact

polygon\*\*. This keeps database queries efficient while allowing real ATC-sector

geometries to be used for the scientific analysis.



\`bbox_presets.json\` contains a disabled \`\_POLYGON_TEMPLATE\`; keys beginning

with \`\_\` and presets with \`"enabled": false\` are ignored by the scripts.



**## Shared helper modules**



**### \`area_presets.py\`**



Resolves rectangular, centre-radius and polygon presets. It also provides the

local point-in-polygon filter used by the snapshot, Trino, weekly downloader,

waypoint downloader, live collector and REST example.



**### \`opensky_safety.py\`**



Contains the common four-state pairwise classification, constant-kinematics

look-ahead, CPA/TCPA calculation and relevant-event filtering. This keeps the

live snapshot and historical Trino workflow on the same safety logic.



**## \`opensky_snapshot.py\`**



Downloads one current state-vector snapshot, saves the raw response, filters the

analysis set, performs current + predicted pairwise analysis and generates the

map.



The raw snapshot remains:



\`\`\`text

data/raw/opensky/opensky_snapshot.csv

\`\`\`



The map remains:



\`\`\`text

results/figures/opensky_snapshot_map.png

\`\`\`



For polygon presets, the exact polygon is also drawn on the map and aircraft

outside it are excluded from the analysis.



**## \`opensky_plot.py\`**



Reusable map renderer. It draws the satellite basemap, cities, aircraft,

current safety circles, labels and the three coloured look-ahead trajectory

segments.



**## \`download_opensky_trino_states.py\`**



Queries OpenSky Trino sequentially in short temporal blocks. A polygon study

area is sent to Trino as its enclosing BBOX and clipped locally after retrieval.



The raw historical CSV is downloaded once. Frames and video are generated

locally from that data; Trino is **\*\*not\*\*** queried once per visual frame.



The visual snapshot cadence remains independently configurable with:



\`\`\`python

FRAME_STEP_SECONDS = 10

\`\`\`



while look-ahead analysis uses:



\`\`\`python

PREDICTION_STEP_SECONDS = 5

\`\`\`



**## \`download_opensky_monday_states.py\`**



Downloads the OpenSky Weekly 24 Hours scientific files and filters each global

hourly source file locally. Study areas now come from the same

\`bbox_presets.json\`, including polygon presets.



The source dataset itself has its published cadence; the downloader does not

invent intermediate state vectors.



**## \`download_waypoints.py\`**



Downloads the public waypoint source and applies the same exact study-area

filter. For a polygon preset, only waypoints inside the polygon are retained.



Regenerate after adding a new polygon:



\`\`\`bash

python3 src/acquisition/download_waypoints.py --areas MY_SECTOR

\`\`\`



**## \`opensky_collect.py\`**



Live collection now uses the shared study-area presets and defaults to a

5-second polling interval. Polygon results are clipped locally after the

rectangular OpenSky REST request.



Always check current OpenSky rate limits before a long live collection.



**## \`opensky_historical_rest.py\`**



Historical REST demonstration. It also uses the shared study-area resolver, so a

polygon preset can be selected without editing raw latitude/longitude limits.


## `download_opensky_sector_archive.py`

For long historical downloads, use the dedicated sector archive downloader instead of the
frame/video workflow.

Its purpose is to download OpenSky Trino state vectors for one configured study sector
between two UTC dates and store them as **one CSV per calendar month**, without generating
PNG frames or MP4 video.

The script is designed for long, resumable archive jobs such as several years of traffic.

### Basic usage

```bash
python3 src/acquisition/download_opensky_sector_archive.py \
    --sector LECMSAN \
    --start 2016-01-01 \
    --end 2026-01-01
```

The start timestamp is inclusive and the end timestamp is exclusive.

Therefore:

```text
--start 2016-01-01
--end   2026-01-01
```

covers traffic from the beginning of 2016 through the end of 2025.

### Sector definition

`--sector` must match a key in `bbox_presets.json`.

The downloader uses the shared `area_presets.py` resolver, so rectangular,
centre-radius and arbitrary polygon presets are supported.

For a polygon sector, Trino is queried with the **smallest enclosing rectangular
BBOX**, then the returned state vectors are clipped **locally to the exact polygon**.

```text
              Trino query BBOX
        ┌────────────────────────┐
        │       ╱────────╲       │
        │      ╱  SECTOR  ╲      │
        │     ╱            ╲     │
        │     ╲            ╱     │
        │      ╲──────────╱      │
        └────────────────────────┘
                   ↓
             local clipping
                   ↓
          exact polygon sector
```

This keeps the database query simple and spatially restricted while allowing the
scientific analysis to use real ATC-sector geometry.

### Cruise-band filtering

The default configuration is aligned with the initial en-route experiment:

```python
FILTER_FL_BAND = True
FL_MIN = 290
FL_MAX = 410
KEEP_ONLY_AIRBORNE = True
```

The FL290-FL410 filter is based on **barometric altitude** and is pushed into the
Trino query.

Set:

```python
FILTER_FL_BAND = False
```

if a later experiment requires all altitude levels.

### Monthly output structure

The archive is stored under:

```text
data/raw/opensky_sector_archive/
```

For example:

```text
data/raw/opensky_sector_archive/
└── LECMSAN/
    ├── year=2016/
    │   ├── month=01/
    │   │   ├── lecmsan_2016-01.csv
    │   │   ├── lecmsan_2016-01.manifest.json
    │   │   └── _window_cache/
    │   ├── month=02/
    │   │   ├── lecmsan_2016-02.csv
    │   │   ├── lecmsan_2016-02.manifest.json
    │   │   └── _window_cache/
    │   └── ...
    ├── year=2017/
    └── ...
```

The final research dataset is therefore naturally partitioned by:

```text
sector / year / month
```

### Resumable downloads

Every completed Trino query window is immediately cached locally.

Example:

```text
_window_cache/
├── window_20250101T000000Z_20250101T010000Z.csv
├── window_20250101T010000Z_20250101T020000Z.csv
└── window_20250101T020000Z_20250101T030000Z.csv
```

If the process is interrupted by a network failure, laptop restart, temporary
authentication issue or `Ctrl+C`, rerun the **same command**.

Already completed windows are skipped and only missing windows are queried.

An empty query result is also cached, so an already-checked empty period is not
requested again on the next run.

When all windows belonging to a month have been completed, they are assembled into
the final monthly CSV.

### Monthly manifest

Each completed monthly CSV has a matching:

```text
*.manifest.json
```

The manifest records the main reproducibility information, including:

```text
sector key
sector name
area type
UTC start/end
query-window duration
number of windows
number of rows
FL filter
airborne filter
BBOX
polygon vertices, when applicable
creation timestamp
```

Keep these manifests together with the monthly CSV files for thesis reproducibility.

### Trino performance strategy

Large date ranges are a **job definition**, not a single Trino query.

For example:

```bash
--start 2016-01-01 --end 2026-01-01
```

does **not** send a ten-year query to Trino.

The downloader splits the job into short sequential windows that never cross the
OpenSky `hour` partition boundary.

With:

```python
QUERY_WINDOW_MINUTES = 60
```

the effective sequence is:

```text
14:00 -> 15:00
15:00 -> 16:00
16:00 -> 17:00
...
```

The current downloader deliberately uses:

```text
Concurrent Trino queries: 1
```

Do not wrap the script in multiprocessing or launch many concurrent copies against
the same OpenSky account.

The Trino call also uses:

```python
cached=True
```

through:

```python
PYOPENSKY_CACHED = True
```

in addition to the downloader's own persistent per-window CSV cache.

### Slow-query handling

The default query window is:

```python
QUERY_WINDOW_MINUTES = 60
```

If a sector/hour request repeatedly becomes slow, reduce it to:

```python
QUERY_WINDOW_MINUTES = 30
```

or:

```python
QUERY_WINDOW_MINUTES = 15
```

The code still guarantees that a query never crosses an `hour` partition boundary.

The downloader prints a warning when a query takes more than:

```python
SLOW_QUERY_WARNING_SECONDS = 300
```

i.e. five minutes.

This is a signal to reduce the requested time window before continuing with a large
historical job.

The script also includes:

```python
POLITE_DELAY_SECONDS = 2.0
```

between sequential queries. This is a conservative local courtesy delay rather than
an OpenSky quota.

Always review the current OpenSky Trino performance documentation before launching a
large archive job:

```text
https://openskynetwork.github.io/opensky-api/trino.html#tools-performance-considerations
```

### Safe first tests

Do not start with the complete ten-year archive.

A useful first test is one day:

```bash
python3 src/acquisition/download_opensky_sector_archive.py \
    --sector LECMSAN \
    --start 2025-01-01 \
    --end 2025-01-02
```

For an even more conservative first run:

```bash
python3 src/acquisition/download_opensky_sector_archive.py \
    --sector LECMSAN \
    --start 2025-01-01 \
    --end 2025-01-02 \
    --max-windows 6
```

This executes at most six **new** Trino windows.

Run the same command again to continue from the cached progress.

A recommended validation sequence for a new sector is:

```text
1 hour
  ↓
6 hours
  ↓
24 hours
  ↓
1 week
  ↓
1 month
  ↓
1 year
  ↓
full historical interval
```

For each stage, check:

```text
average query duration
maximum query duration
rows returned per hour
CSV size
known/missing periods
polygon clipping
```

### Ten-year archive example

Once a representative short test and at least one full month have been validated:

```bash
python3 src/acquisition/download_opensky_sector_archive.py \
    --sector LECMSAN \
    --start 2016-01-01 \
    --end 2026-01-01
```

There is no need to keep the ten-year dataset in RAM. Each completed month becomes an
independent CSV and the job can be safely resumed whenever required.

For experiments used in the thesis, preserve:

```text
bbox_presets.json
exact sector polygon
monthly CSV files
monthly manifests
script version / Git commit
pyOpenSky version
archive download date
known OpenSky data-availability gaps
```

This allows the historical experiment to be reproduced even if the sector definitions,
tooling or OpenSky archive evolve later.





**## Deferred work**



The following ideas remain deliberately outside this code revision:



\- synthetic aircraft/conflict injection;

\- AI training and conflict-resolution policy;

\- training on one sector and validation on different sectors;

\- airport SID/STAR/IAC optimisation;

\- dynamic ATC sector handover logic.



The new polygon preset architecture is intended to make a later real-sector

implementation straightforward without hard-coding administrative regions.



**## Dependencies**



In addition to the project requirements, the historical Trino workflow requires

\`pyopensky\`, \`python-dotenv\`, \`imageio\` and a working ffmpeg backend for MP4

generation.



Do not commit OpenSky credentials. \`credentials.env\` should remain outside

version control.





**## Waypoint overlay**



The traffic maps can now plot the aviation waypoints stored in

[\`waypoints.json\`]\(waypoints.json).



Enable them with:



\`\`\`python

SHOW_WAYPOINTS = True

SHOW_WAYPOINT_LABELS = True

\`\`\`



Waypoints are drawn as **\*\*white squares with a black edge\*\***, which visually

distinguishes them from the **\*\*white circular city markers\*\***. Their names use

the same collision-aware label-placement system used for cities and aircraft.



The main style controls are:



\`\`\`python

WAYPOINT_MARKER = "s"

WAYPOINT_MARKER_SIZE = 34

WAYPOINT_FACE_COLOR = "white"

WAYPOINT_EDGE_COLOR = "black"

WAYPOINT_EDGE_WIDTH = 0.9

WAYPOINT_LABEL_COLOR = "white"

\`\`\`



\`opensky_snapshot.py\` and \`download_opensky_trino_states.py\` load the

waypoints once for the selected study area and pass them to

\`opensky_plot.render_map()\`. For polygon presets, the waypoint set is also

clipped to the exact polygon.



If an old \`waypoints.json\` uses a different top-level key from the current

\`bbox_presets.json\`, the shared loader falls back to deduplicating the

available waypoints and filtering them geometrically to the active study area.



When a waypoint name cannot be placed without overlapping another label or an

aircraft symbol, its square marker is still drawn and only the text label is

skipped.
