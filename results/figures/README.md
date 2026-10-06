# Figures and maps

This folder stores visual outputs generated from the aircraft state-vector data.
The OpenSky maps in this folder (for example, `opensky_bcn_map.png`,
`opensky_galicia_map.png` and `opensky_spain_north_map.png`) are produced by
`src/acquisition/opensky_snapshot.py` with different geographic configurations.

## What the map represents

The map shows a geographic snapshot of aircraft detected by OpenSky inside the
configured area of interest. It combines:

- aircraft positions from the OpenSky state-vector response;
- a geographic basemap, when the map-tile service is available;
- aircraft labels and movement vectors based on heading and ground speed;
- three-digit true-track labels at the tip of each movement vector, measured
  clockwise from geographic north (for example, `072°` or `250°`);
- altitude, vertical-speed and ground/airborne information used in the display;
- pairwise proximity or conflict-warning annotations according to the configured
  horizontal and vertical research thresholds.

These annotations are analytical visualisations only. They are not an operational
separation assessment and must not be interpreted as air-traffic-control advice.

## How images are obtained

From the repository root, run:

```bash
python src/acquisition/opensky_snapshot.py
```

The script queries the OpenSky states API, processes the response and saves the
default map as `results/figures/opensky_snapshot_map.png`. The corresponding
processed snapshot is saved as `data/raw/opensky/opensky_snapshot.csv` for
traceability. When several configurations are being compared, rename or copy
each generated PNG immediately—for example, to `opensky_bcn_map.png`,
`opensky_galicia_map.png` or `opensky_spain_north_map.png`—so later runs do not
overwrite the preserved figures.

The geographic scope is controlled in the script. `AREA_MODE="BBOX"` uses
`CUSTOM_BBOX`; `AREA_MODE="CENTER_RADIUS"` derives a bounding box around the
selected airport preset and radius. Re-running the script replaces the current
snapshot and map, so preserve renamed copies when comparing different runs.
