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

## Current execution scope

Stages 01–04 have already been run on the 13-week source dataset and must be
preserved, not rerun. From stage 05 onward the methodological pilot is limited
to `archive_week=20161015_20161021` (`FIRST_WEEK_PILOT`). No other week may be
processed without explicit authorization. Stage 05 is complete for that week;
its existing interaction table was filtered by exact primary matching IDs,
then stage 05b produced one pre-event row per window. Stage 06 has been run
on those windows. Separately authorized cruise-only Stages 07 and 09 are now complete;
do not run Stages 08 or 10 or another SCAT week without separate authorization.

The altitude-stratified Stage 06 iteration first audited and created
`window_features_v2_20161015_20161021.parquet` from the existing primary
pair table. V2 validates the measured-FL-to-feet conversion, re-anchors phase
and centre, and adds within-30-NM neighbour diagnostics while preserving all
20,194 windows. `06b_explore_altitude_bands.py` keeps matched pairs intact,
compares altitude/level bands separately and produces an OpenSky processing
plan. Its principal domain is FL290–FL410; lower bands are exploratory.
Physical tails and within-band matching failures are documented, not hidden.
`06c_finalize_altitude_deliverables.py` is a lightweight, repeatable reporting
pass over those existing first-week outputs. It adds the requested matching
report, robust top-ten tables, cruise descriptive schema, explicit clearance
zeroes and OpenSky availability classes without rerunning any earlier stage.

The final pre-Stage-07 cruise validation creates a **separate**
`CRUISE_MATCHING_V2` from the existing V1 cruise intervention IDs and
clearance-free first-week controls. `04c_cruise_matching_v2.py` performs
context-only, 1:1 without-replacement caliper matching; it does not rerun
Stages 01–05. `06d_validate_cruise_matching_v2.py` reuses the existing
window/pair features for bearing and future-CPA quality sensitivities,
V1-versus-V2 descriptive checks, and the final gate. The filtered feature
file preserves the original V2 columns and adds circular-bearing and
future-CPA derived fields. The methodological decision is
`READY_FOR_STAGE07_CRUISE`, which was a methodological gate rather than
permission. Stage 07 was later separately authorized and executed through
`07_explain_interventions_cruise.py`; the legacy Stage 07 was not executed.
The cruise script uses the 4,098-pair V2 cohort, a pair/focal-flight-connected
chronological train/validation/test split with embargo, an explicit predictor
allowlist, pair-conditioned statistical analysis and predeclared SCAT Full,
OpenSky-compatible and minimal-physical model families. It writes only to
`07_explainability_cruise/`, plus its separate report. The recommendation is
`READY_FOR_COUNTERFACTUAL_ANALYSIS`; it does not authorize Stage 08/09/10.

Stage 09 was subsequently authorized separately and run with
`09_counterfactual_clearance_analysis_cruise.py` on exactly 4,098 V2
interventions. The 4,098 matched controls calibrate finite-horizon
constant-kinematic model error (TRAIN only for materiality thresholds), not
ATC counterfactual labels. Native surveillance states and clearances are
read-only inputs; 60/120/300-s horizons are primary and 600 s is sensitivity.
No safety separation threshold, Stage-07 prediction or OpenSky rule is used
for the three provisional labels. The conservative result is
`INSUFFICIENT_COUNTERFACTUAL_EVIDENCE`, chiefly because most events remain
`UNKNOWN` and long-horizon coverage/sensitivity are weak. The Stage 09 script
uses resumable day/block checkpoints; the legacy Stage 09 was not run.
Its completed artifacts live in `data/processed/scat/09_counterfactual_cruise/`;
inspect the [report](../../data/processed/scat/PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md)
before any explicit rerun. `--overwrite` regenerates only Stage 09 outputs
while reusing validated native-state/day checkpoints; it is not needed to
read the existing result.

See [the original pilot report](../../data/processed/scat/PILOT_STAGE06_REPORT.md),
[the altitude-stratified report](../../data/processed/scat/PILOT_STAGE06_ALTITUDE_STRATIFIED_REPORT.md),
[the final cruise matching V2 report](../../data/processed/scat/PILOT_CRUISE_MATCHING_V2_REPORT.md),
[the Stage 07 cruise explainability report](../../data/processed/scat/PILOT_STAGE07_CRUISE_EXPLAINABILITY_REPORT.md),
[the Stage 09 cruise counterfactual report](../../data/processed/scat/PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md),
[the OpenSky plan](../../data/processed/scat/OPENSKY_PROCESSING_PLAN_FROM_SCAT.md)
and [the task log](../../data/processed/scat/PILOT_STAGE06_PROGRESS.md).

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

`--stop-after-week` is now mandatory and must equal the pilot week. The
existing feature table is complete; the command above is retained only as
reproducibility documentation. Do not use `--overwrite` on the completed stage
05 output. Its checkpoint and run configuration record the finished pilot.

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

Later stages consume these artifacts. They must preserve temporal splitting,
avoid duplicate windows across train/test, and use the cautious provisional
labels `UNKNOWN`, `LIKELY_INTERACTION_RELATED` and
`LIKELY_NON_INTERACTION_OPERATIONAL` only after the counterfactual analysis.

The original Stage 06 consumed only
`05_features/window_features_20161015_20161021.parquet`, with one pre-event
window per observation. Its clustering used only intervention windows;
clearance type was joined afterward. The legacy Stage 07 and Stage 08 still
default to the same pilot window table; the legacy Stages 09–10 default to the
pilot primary pair table, because their retrospective analyses require
trajectories. All have week guards, but their methods and resource use need
review before they are run. The reviewed cruise-only Stage 09 implementation
is separate from that legacy entrypoint.

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

