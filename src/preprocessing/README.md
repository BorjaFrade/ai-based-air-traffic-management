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

## SCAT weekly ZIP archives

[`preprocess_scat_10s.py`](preprocess_scat_10s.py) is the corresponding
adapter for the SCAT dataset. It streams the flight JSON files from the 13
weekly ZIPs and keeps the temporal context that OpenSky does not provide:
flight-plan updates, controller clearances, active control centre and
predicted route. It writes chronological daily files under
`data/processed/scat_10s/<AREA>/date=YYYY-MM-DD/`.

The area presets `SCAT_ESMM_GOTHENBURG_GG`, `SCAT_ESMM_MALMO_MS` and
`SCAT_ESOS_STOCKHOLM_SE` are envelopes of the real local SCAT sectors
documented in `airspace.json`, not arbitrary city-radius boxes. The matching
`*_ENROUTE` presets use the real ESMM/4, ESMM/8, ESOS/6 and ESOS/3 sector
envelopes for the first-pass FL290–FL410 analysis.

The preprocessor keeps source surveillance rows at all levels. The SCAT
detector applies FL290–FL410 by default, matching the OpenSky cruise-band
experiment; use `--all-flight-levels` for an airport/terminal analysis.

After preprocessing, validate the feet/metres, m/s/kt and vertical-rate
conversions with:

```bash
python3 src/preprocessing/validate_scat_units.py \
  --area SCAT_ESMM_GOTHENBURG_ENROUTE
```

Smoke test one hundred flights from one week:

```bash
python3 src/preprocessing/preprocess_scat_10s.py \
  --area SCAT_ESMM_GOTHENBURG_GG \
  --week 20161015_20161021 \
  --max-flights 100 \
  --overwrite
```

Process the complete downloaded dataset for one area:

```bash
python3 src/preprocessing/preprocess_scat_10s.py \
  --area SCAT_ESOS_STOCKHOLM_SE \
  --overwrite
```

The script uses an external chronological merge for each daily file, so the
conflict detector can stream snapshots without loading a whole week into
memory. Use `--keep-intermediate` when auditing the unsorted per-archive
parts.
