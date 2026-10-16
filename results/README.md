# Results

`results/` contains rendered figures and tables produced by the acquisition,
validation and analysis scripts. Large raw and processed datasets belong under
`data/`, not here.

## Current snapshot

Run the live snapshot script from the repository root:

```bash
python3 src/acquisition/opensky_snapshot.py
```

It writes the current map to `results/figures/opensky_snapshot_map.png` and
exports the current relevant-pair table alongside it. The geographic area and
radius are configured in `src/acquisition/opensky_snapshot.py` before running.

## Historical safety analysis

The detector writes event CSVs and run summaries under:

```text
data/processed/opensky_separation/<area>/runs/<run-name>/
```

The quality-analysis script then writes CSV summaries and PNG/PDF figures to:

```text
data/processed/opensky_separation/<area>/runs/<run-name>/quality_analysis/<label>/
```

For example:

```bash
python3 src/conflict_detection/analyze_atc_safety_quality.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --label "Manual control" \
  --profile ENROUTE
```

Random validation maps are stored below the same run directory:

```bash
python3 src/conflict_detection/generate_random_validation_snapshots.py \
  --area LECMSAN \
  --run-name 20190101T000000_20190228T235950 \
  --profile ENROUTE \
  --n 10 \
  --seed 42
```

This creates `validation_screenshots/random_seed_42/` with the selected moments,
validation report and PNG maps.

## Provenance

Every result should record, at minimum:

- source provider and dataset version or AIRAC amendment;
- area key and polygon/bounding-box configuration;
- UTC start and end timestamps;
- detector and quality-analysis command lines;
- separation profile and configuration version;
- whether the result uses current state or predicted event state.

Generated outputs are reproducible only when these inputs and configuration
values are preserved. See [`figures/README.md`](figures/README.md) for the
rendered-image conventions.
