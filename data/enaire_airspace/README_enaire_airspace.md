# ENAIRE Spanish ATC Airspace Map — 8K, FL290–FL410

This test version focuses the sector map on the cruise/RVSM study band used in
the TFM and improves the dense labels visible around Galicia, Madrid and
Barcelona.

## Vertical filtering

By default:

```python
FILTER_BY_FL_RANGE = True
FL_MIN = 290
FL_MAX = 410
```

Only sector records whose published vertical range intersects FL290–FL410 are
drawn.

The script parses the ENAIRE vertical-limit fields defensively and adds:

```text
LOWER_FT_PARSED
UPPER_FT_PARSED
FL_RANGE_MATCH
```

to the exported metadata.

If no usable vertical limits are found in the downloaded ENAIRE sector schema,
the script prints a warning and automatically keeps all sectors instead of
silently producing an empty map.

Unknown vertical limits can be retained with:

```python
KEEP_SECTORS_WITH_UNKNOWN_VERTICAL_LIMITS = True
```

The default is `False`.

## Dense labels

Sector labels remain compact ENAIRE identifiers, e.g.:

```text
LECMSAO
LECMSAB
LECMSAI
```

Crowding is detected in display pixels:

```python
LABEL_CLUSTER_DISTANCE_PX = 180
```

Normal labels are white:

```python
NORMAL_LABEL_COLOR = "#FFFFFF"
SECTOR_LABEL_FONTSIZE = 8.0
```

Crowded labels are larger and yellow:

```python
CROWDED_LABEL_COLOR = "#FFD400"
CROWDED_LABEL_FONTSIZE = 10.5
```

They are slightly displaced around their sector-centre anchors and can use
short leader lines:

```python
MOVE_CROWDED_LABELS = True
ADD_LABEL_LEADER_LINES = True
```

This is intended to make dense areas readable without losing the connection
between each label and its polygon.

## Resolution

The map remains true 8K UHD:

```text
7680 x 4320 px
```

with Esri World Imagery as the satellite basemap.

## Installation

From this directory, install the dedicated geospatial dependencies:

```bash
python3 -m pip install -r requirements_enaire_airspace.txt
```

## Run

```bash
python3 data/enaire_airspace/download_enaire_airspace_map.py
```

The script has no command-line parameters. Edit the configuration constants at
the top of `download_enaire_airspace_map.py` before running. For example, use
`FILTER_BY_FL_RANGE = True`, `FL_MIN = 290`, `FL_MAX = 410` for the RVSM band,
or set `FORCE_DOWNLOAD = True` to refresh the cached ENAIRE GeoJSON files.

Run dates are not passed to the script: the source is the current ENAIRE ArcGIS
REST layer at execution time. Record the execution date and archive the raw
GeoJSON files when the map is used in the thesis.

Existing ENAIRE GeoJSON data are reused while:

```python
FORCE_DOWNLOAD = False
```

## Important

This is a **band-intersection map**: a sector is included if any part of its
published vertical interval intersects FL290–FL410.

It is not the same as a single-level slice such as FL350. A later version can
generate separate maps at FL290, FL310, FL330, FL350, FL370, FL390 and FL410
if that is useful for checking vertical sectorisation changes.
