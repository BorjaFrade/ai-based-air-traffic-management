# SCAT exploratory analysis

This pipeline is the exploratory phase of the TFM. It uses the Swedish Civil
Air Traffic Control (SCAT) records to study what precedes an operational ATC
intervention. It does not define conflict from the existence of a clearance,
and it does not use the OpenSky GREEN/YELLOW/ORANGE/RED detector until stage 10.

The central target is:

```text
ATC_INTERVENTION = 1  actionable clearance/intervention exists at t
ATC_INTERVENTION = 0  matched comparable control situation without clearance
```

`ATC_INTERVENTION` is not `CONFLICT`. A clearance may be issued for
sequencing, coordination, efficiency, weather, transfer of control or many
other operational reasons.

## Recommended order

Run the first five stages with the full dataset only after a smoke test on one
week or a limited number of flights. The raw ZIPs are streamed and never
extracted or modified. The current TFM execution is deliberately staged: the
feature build is being completed first for the archive week
`20161015_20161021`, and stages 06–10 will consume that scoped table before
we decide whether to expand to all 13 weeks.

```bash
python3 src/scat_analysis/01_audit_scat.py
python3 src/scat_analysis/02_extract_scat_traffic.py
python3 src/scat_analysis/03_extract_scat_clearances.py
python3 src/scat_analysis/04_build_clearance_windows.py
python3 src/scat_analysis/05_build_interaction_features.py
```

## Resource-safe staged execution

Stages 04 and 05 are designed to stay below the available RAM. Stage 04
streams state rows and writes per-week checkpoint shards. Stage 05 partitions
inputs into one-hour blocks, processes focal rows in chunks, and writes a
checkpoint after every target chunk. The active first-week run uses:

```bash
python3 -u src/scat_analysis/05_build_interaction_features.py \
  --target-chunk-size 5000 \
  --input-batch-size 50000 \
  --time-block-hours 1 \
  --output-root data/processed/scat/05_features \
  --resume \
  --stop-after-week 20161015_20161021
```

`--stop-after-week` finalises `interaction_features.parquet` after the named
week and exits before processing the next week. The resulting feature table is
therefore intentionally scoped to that week; `run_config.json` records the
scope and `checkpoint.json` records the completed blocks. To resume an
interrupted run, use the same command. To later build all 13 weeks, start a
new run with `--overwrite` and omit `--stop-after-week`.

The intermediate partition files and raw feature shards are retained so a
failure can be resumed without rebuilding stages 01–04. Do not delete
`05_features/checkpoints`, `05_features/input_partitions` or
`05_features/checkpoint.json` while the staged run is active.

Smoke-test example:

```bash
python3 src/scat_analysis/01_audit_scat.py \
  --weeks scat20161015_20161021.zip \
  --max-flights-per-archive 100

python3 src/scat_analysis/02_extract_scat_traffic.py \
  --weeks scat20161015_20161021.zip \
  --max-flights-per-archive 100
```

Every stage has `--help`, records a `run_config.json` and updates:

```text
data/processed/scat/00_READ_ME_FIRST.md
```

## Outputs

1. `01_audit_scat.py` inspects the real nested JSON structure, clearance
   fields, special airspace/weather members, timestamps and sampling.
2. `02_extract_scat_traffic.py` writes one row per native surveillance plot to
   `02_traffic/scat_states.parquet` and keeps predicted-trajectory points in a
   sidecar Parquet file.
3. `03_extract_scat_clearances.py` writes one row per actionable change in
   `fpl_clearance`. It records heading, level, speed, previous/new values and
   provenance, and sets only `ATC_INTERVENTION=1`.
4. `04_build_clearance_windows.py` builds ten-minute pre/post windows and
   matched controls. Rows after the event are retained for retrospective work
   but marked `predictor_eligible=False`.
5. `05_build_interaction_features.py` calculates continuous separation,
   relative kinematics, CPA/TCPA, trends, traffic density and a time-consistent
   proxy comparison with TopSky predicted points. It supports chunk-level
   checkpoints and a week-scoped finalisation mode. No safety threshold is
   used.

Stages 06–10 consume these artifacts. They must preserve temporal splitting,
avoid duplicate windows across train/test, and use the cautious provisional
labels `UNKNOWN`, `LIKELY_INTERACTION_RELATED` and
`LIKELY_NON_INTERACTION_OPERATIONAL` only after the counterfactual analysis.

Once the first-week feature table exists, run stages 06–10 in order against
the default `05_features/interaction_features.parquet`. Because the table is
already scoped to one week, these analyses will remain first-week analyses
without reloading the other 12 weeks:

```bash
python3 src/scat_analysis/06_explore_interventions.py --overwrite
python3 src/scat_analysis/07_explain_interventions.py --overwrite
python3 src/scat_analysis/08_bayesian_network.py --overwrite
python3 src/scat_analysis/09_counterfactual_clearance_analysis.py --overwrite
python3 src/scat_analysis/10_compare_opensky_rules.py --overwrite
```

Stage 10 is still only a comparison with the existing OpenSky rule states;
none of stages 01–09 turns `ATC_INTERVENTION` into a conflict label.

## Important interpretation details

- `clearance_cfl=360` with `clearance_cfl_unit=F` means FL360, nominally
  36,000 ft; `A` means altitude in feet.
- The state extractor attaches the latest flight-plan/clearance record at or
  before each surveillance plot. It does not claim a new controller order was
  issued every five seconds.
- CPA, TCPA, closing rates and separation are continuous explanatory
  variables, not labels.
- A very high predictive metric is a leakage warning until the temporal split,
  window grouping and post-clearance exclusion have been audited.

