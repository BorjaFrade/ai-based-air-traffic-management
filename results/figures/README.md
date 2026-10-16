# Figures and maps

This directory contains rendered maps, plots and other visual outputs. The
figures are analytical research outputs, not operational separation displays.

## Live OpenSky map

Configure the area in `src/acquisition/opensky_snapshot.py` and run:

```bash
python3 src/acquisition/opensky_snapshot.py
```

The script queries the live states API and writes:

```text
opensky_snapshot_map.png
opensky_snapshot_separation.csv
```

The default map area is selected by `AREA_MODE` and either
`SELECTED_BBOX_PRESET` or `SELECTED_PRESET`/`RADIUS_KM`. Re-running replaces
the current-snapshot files. Rename or copy them immediately when preserving a
specific area or observation time.

## Historical quality figures

Historical PNG/PDF figures are created inside the detector run by:

```bash
python3 src/conflict_detection/analyze_atc_safety_quality.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --label "Manual control" \
  --profile ENROUTE
```

The numbered outputs include current pair-state distributions, non-green rate,
worst-horizon event state, snapshot time distributions, daily metrics, event
episodes and episode duration. Each plot is written as `.png` and `.pdf`.

## Validation screenshots

To render selected yellow/orange/red moments from an existing run:

```bash
python3 src/conflict_detection/generate_random_validation_snapshots.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --profile ENROUTE \
  --n 10 \
  --seed 42
```

The output directory contains `selected_random_moments.csv`,
`validation_report.csv` and up to 30 PNGs. The requested pixel dimensions come
from the selected area's `render` block in `src/acquisition/bbox_presets.json`.

## Interpretation

Maps combine aircraft state vectors, movement vectors, labels, optional
waypoints/corridors and the configured research safety model. Horizontal and
vertical thresholds, warning colours and prediction horizons come from
`src/acquisition/separation_profiles.json`. The figures must therefore be
reported together with the area, UTC observation interval, profile and source
dataset; they must not be presented as operational ATC advice.
