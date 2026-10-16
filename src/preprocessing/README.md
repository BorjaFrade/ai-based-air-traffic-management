# OpenSky preprocessing

[`preprocess_opensky_10s.py`](preprocess_opensky_10s.py) converts the monthly
CSV archive produced by
[`download_opensky_sector.py`](../acquisition/download_opensky_sector.py) into
reusable, chronological 10-second UTC state files. It does not detect
conflicts; that is the following stage in `src/conflict_detection/`.

## Inputs and outputs

Input files are discovered below:

```text
data/raw/opensky_sectors/<source-sector>/year=YYYY/month=MM/*.csv
```

The source-sector directory must contain files named like
`<name>_YYYY-MM.csv`. The output is written to:

```text
data/processed/opensky_10s/<analysis-area>/year=YYYY/month=MM/
├── <name>_YYYY-MM_10s.csv
├── <name>_YYYY-MM_10s.manifest.json
└── daily_cache/                 # removed unless --keep-daily-cache is used
```

Raw files are never modified. The default reconstruction uses a 10-second
observation grid, permits short interpolation/extrapolation gaps, filters to
the selected polygon and keeps the FL290–FL410 cruise band configured in the
script.

## Command-line options

```text
--source-sector NAME    Raw archive directory; default LECMSAN_ENVELOPE
--analysis-area NAME    Polygon preset; default LECMSAN
--start-month YYYY-MM   First month, inclusive (required)
--end-month YYYY-MM     Last month, inclusive (required)
--chunksize N           CSV rows read per chunk; default 250000
--overwrite             Rebuild outputs that already exist
--keep-daily-cache      Keep resumable daily intermediate files
```

## Examples

Process January and February 2019:

```bash
python3 src/preprocessing/preprocess_opensky_10s.py \
  --source-sector LECMSAN_ENVELOPE \
  --analysis-area LECMSAN \
  --start-month 2019-01 \
  --end-month 2019-02
```

Rebuild only January with a smaller memory footprint:

```bash
python3 src/preprocessing/preprocess_opensky_10s.py \
  --source-sector LECMSAN_ENVELOPE \
  --analysis-area LECMSAN \
  --start-month 2019-01 \
  --end-month 2019-01 \
  --chunksize 100000 \
  --overwrite
```

Retain the daily cache for inspection or a later resume:

```bash
python3 src/preprocessing/preprocess_opensky_10s.py \
  --source-sector LECMSAN_ENVELOPE \
  --analysis-area LECMSAN \
  --start-month 2019-01 \
  --end-month 2019-02 \
  --keep-daily-cache
```

The `--analysis-area` value must name a polygon preset in
[`../acquisition/bbox_presets.json`](../acquisition/bbox_presets.json). A
bounding-box/envelope preset is appropriate for `--source-sector`; the exact
polygon is applied during preprocessing.
