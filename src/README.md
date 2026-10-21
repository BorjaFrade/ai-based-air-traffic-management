# Source code

The `src/` directory contains the reusable research pipeline. Run the commands
from the repository root so that the scripts resolve `data/` and `results/`
relative to the project.

## Modules

- [`acquisition/`](acquisition/): OpenSky and ENAIRE downloads, live snapshots,
  safety logic and map rendering. See [`acquisition/README.md`](acquisition/README.md).
- [`preprocessing/`](preprocessing/): converts monthly raw OpenSky archives into
  regular 10-second UTC state files. See [`preprocessing/README.md`](preprocessing/README.md).
- [`conflict_detection/`](conflict_detection/): detects current/predicted
  separation events, creates validation screenshots and calculates quality
  metrics. See [`conflict_detection/README.md`](conflict_detection/README.md).
- [`scat_analysis/`](scat_analysis/): exploratory SCAT analysis of controller
  interventions, continuous interaction features, provisional counterfactual
  labels and the final OpenSky-rule comparison. See
  [`scat_analysis/README.md`](scat_analysis/README.md).
- `trajectories/`, `visualization/` and `models/`: reserved for later thesis
  stages; they are not currently complete command-line pipelines.

## Data flow

```text
src/acquisition/download_opensky_sector.py
        ↓
data/raw/opensky_sectors/<source-sector>/year=YYYY/month=MM/*.csv
        ↓
src/preprocessing/preprocess_opensky_10s.py
        ↓
data/processed/opensky_10s/<analysis-area>/year=YYYY/month=MM/*_10s.csv
        ↓
src/conflict_detection/detect_opensky_conflicts.py
        ↓
data/processed/opensky_separation/<area>/runs/<run-name>/
        ↓
quality analysis, validation screenshots and thesis figures
```

The raw downloader uses a bounding-box/envelope preset because that is the
efficient query area. The preprocessor then applies the exact polygon preset
used by the analysis, for example `LECMSAN` inside `LECMSAN_ENVELOPE`.

## Quick command reference

```bash
# Download January and February 2019 (end is exclusive)
python3 src/acquisition/download_opensky_sector.py \
  --sector LECMSAN_ENVELOPE \
  --start "2019-01-01 00:00:00" \
  --end "2019-03-01 00:00:00"

# Build regular 10-second files (months are inclusive)
python3 src/preprocessing/preprocess_opensky_10s.py \
  --source-sector LECMSAN_ENVELOPE \
  --analysis-area LECMSAN \
  --start-month 2019-01 \
  --end-month 2019-02

# Detect events (end is inclusive and should be 10-second aligned)
python3 src/conflict_detection/detect_opensky_conflicts.py \
  --area LECMSAN \
  --start "2019-01-01 00:00:00" \
  --end "2019-02-28 23:59:50"
```

Configuration is shared through `src/acquisition/bbox_presets.json`,
`separation_profiles.json`, `waypoints.json` and `corridors.json`. The
separation profile is loaded by the detector and by the validation/rendering
scripts, so record the profile name and version when reporting results.
