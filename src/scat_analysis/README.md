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
on those windows. Separately authorized cruise-only Stage 07, Stage 09 V1 and
Stage 09 V2 are now complete;
Stage 10 has now been separately authorized and run on that same cruise cohort.
Do not run Stage 08 or another SCAT week without separate authorization.

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

The separately authorized `09b_clearance_aware_counterfactual_cruise.py`
preserves V1 as a read-only `KINEMATIC_COUNTERFACTUAL_BASELINE`. V2 joins the
immediately preceding/current/next actionable SCAT clearances, rejects old or
incoherent prior instructions, and compares previous-command CF-A against
V1 kinematic CF-B. It reuses V1's native-state cache, candidate pairs,
censoring and TRAIN-control error calibration. Only 60/120 s determine V2
labels; 300/600 s are diagnostic. No LEVEL pair shows a CF-A/CF-B difference
above control error at the primary horizons. V2 yields 1 provisional
interaction-related, 192 non-interaction and 3,905 UNKNOWN events; the decision
remains `INSUFFICIENT_COUNTERFACTUAL_EVIDENCE`. Its separate outputs and
resumable day checkpoints are in `09_counterfactual_cruise_v2/`; neither V1
nor earlier stages were recomputed.

The reformulated `10_compare_rule_geometry_with_atc_intervention_cruise.py`
compares continuous traffic geometry and predeclared fixed/predicted rules with
recorded ATC intervention in the 4,098 matched cruise pairs. It reads the
existing Stage 05 native pair rows in batches to keep paired separation/CPA
coherent, reuses Stage 07 splits and saved TEST predictions, and writes only
its own `10_rule_geometry_comparison_cruise/` outputs and
`PILOT_STAGE10_RULE_GEOMETRY_VS_ATC_REPORT.md`. Neither intervention nor any
rule/Stage 09 label is conflict ground truth. The legacy
`10_compare_opensky_rules.py` was preserved and not run. The exact production
OpenSky dynamic colour-state logic cannot be reconstructed from the available
pair aggregates; a static profile proxy is explicitly separated from that
unavailable comparator. The Stage 10 decision is `REVISE_RULE_COMPARISON`;
the bounded OpenSky quality/feature-yield pilot remains `UNDETERMINED` and
full OpenSky processing is not authorized.

See [the original pilot report](../../data/processed/scat/PILOT_STAGE06_REPORT.md),
[the altitude-stratified report](../../data/processed/scat/PILOT_STAGE06_ALTITUDE_STRATIFIED_REPORT.md),
[the final cruise matching V2 report](../../data/processed/scat/PILOT_CRUISE_MATCHING_V2_REPORT.md),
[the Stage 07 cruise explainability report](../../data/processed/scat/PILOT_STAGE07_CRUISE_EXPLAINABILITY_REPORT.md),
[the Stage 09 cruise counterfactual report](../../data/processed/scat/PILOT_STAGE09_CRUISE_COUNTERFACTUAL_REPORT.md),
[the Stage 09 V2 clearance-aware report](../../data/processed/scat/PILOT_STAGE09_V2_CLEARANCE_AWARE_REPORT.md),
[the Stage 10 geometry/intervention report](../../data/processed/scat/PILOT_STAGE10_RULE_GEOMETRY_VS_ATC_REPORT.md),
[the OpenSky plan](../../data/processed/scat/OPENSKY_PROCESSING_PLAN_FROM_SCAT.md)
and [the task log](../../data/processed/scat/PILOT_STAGE06_PROGRESS.md).

## Stages 20–21: clearance attribution and maneuver screening

These stages examine whether the association between encounter geometry and
ATC clearances persists when unique clearances and contiguous encounters are
counted separately. They also provide an initial retrospective screen for
maneuvers. They write only to the directory specified by `--out` and do not
change earlier stage outputs.

- `stage20_audit_clearances.py` joins pair-at-time observations to unique
  HEADING/LEVEL clearances. It reports counts by color, preceding and following
  orders, attribution of one order to multiple pairs, and contiguous non-green
  episodes.
- `stage21_maneuver_screening.py` compares observed heading/altitude changes
  and separation with a simplified inertial, no-maneuver projection. This is
  a retrospective screen, not a causal counterfactual.
- `scat_stage_io.py` provides shared CSV/Parquet reading, column discovery,
  and UTC timestamp normalization. `stage20_export_pairs.py` replays the Stage
  16 calculation on processed surveillance and predicted-route inputs to
  export every route-predictor pair snapshot. Keep these scripts together in
  `src/scat_analysis/`.

Run the examples below from the repository root. They require `pandas` and
`numpy`; Parquet input also requires `pyarrow`. DuckDB is not required:

```bash
python3 -m pip install --upgrade pandas numpy pyarrow
```

### Find and inspect the inputs

The original unique-clearance table is
`data/processed/scat/03_clearances/scat_clearances.parquet`. Stage 16 itself
saved aggregates and a sample, so use the separate full export in
`results/stage20/pairs/`. Each `pairs_<week>.parquet` contains the nominal
week's route-predictor snapshots, both original `flight_key` values, source
members, `m_min`, `tau_sec`, frozen Stage 19 color, and the Stage 16 300-second
label. The export is generated with:

```bash
python3 src/scat_analysis/stage20_export_pairs.py \
  --out results/stage20/pairs --weeks all
```

This replay reads the existing Stage 02 surveillance and prediction Parquet
files, follows the Stage 16 120-second snapshot, 20-minute horizon, 10-second
prediction grid, FL290–410 and 350 NM population, and retains only the 91
nominal days used in Stage 19. It does not overwrite Stage 16–19 results.
Inspect the canonical input before auditing:

```bash
python3 src/scat_analysis/stage20_audit_clearances.py \
  --pairs results/stage20/pairs \
  --clearances data/processed/scat/03_clearances/scat_clearances.parquet \
  --inspect \
  --out results/stage20
```

Inspection prints the columns and three sample rows per input and creates
`results/stage20/mapping_example.json`. The auditor requires this canonical
pair schema and automatically recognizes the Stage 03 `clearance_epoch` as
event time. It restricts Stage 03's unique clearance IDs to the Stage 13/14
eligible orders used by Stage 16, excluding speed-only and other ineligible
records. It reports calibration and test weeks separately without refitting
the Stage 19 color thresholds.

### Stage 20: pilot and full audit

```bash
python3 src/scat_analysis/stage20_audit_clearances.py \
  --pairs results/stage20/pairs/pairs_20161015_20161021.parquet \
  --clearances data/processed/scat/03_clearances/scat_clearances.parquet \
  --out results/stage20_pilot_week

python3 src/scat_analysis/stage20_audit_clearances.py \
  --pairs results/stage20/pairs \
  --clearances data/processed/scat/03_clearances/scat_clearances.parquet \
  --out results/stage20
```

The first command audits one complete week; the second covers all 13 weeks.
The output files are:

| File | Contents |
| --- | --- |
| `stage20_summary.json` | Totals, rules, mappings, and warnings |
| `stage20_by_color.csv` | Pair-at-time observations and attributed ATC events by color |
| `stage20_episodes.csv` | Contiguous non-green episodes grouped by pair and time gap |
| `stage20_non_green_observations.csv.gz` | Non-green observations with preceding/following ATC events |
| `stage20_clearance_attribution.csv.gz` | One row per unique clearance, including how many observations of each color attribute it |

Review `stage20_summary.json`, `stage20_by_color.csv`, and
`stage20_episodes.csv` first. The completed export contains 21,567,488
pair-at-time observations and 8,434 red ones over 13 weeks. Its color counts
and Stage 16 order labels match the saved Stage 19 tables exactly. See the
[Stage 20 audit report](../../results/stage20/STAGE20_AUDIT_REPORT.md) for the
attribution results and interpretation.

### Stage 21: observed trajectories and maneuver screening

Stage 21 reads the complete Stage 20 episode, observation and clearance
tables, plus the original Stage 02 surveillance points and Stage 03 clearance
records. The surveillance altitude column is `altitude_ft`; pass `--alt-unit ft`
explicitly. Inspect the input schemas, then run a reproducible calibration
pilot and freeze its rules for the full run:

```bash
python3 src/scat_analysis/stage21_maneuver_screening.py \
  --stage20 results/stage20 \
  --tracks data/processed/scat/02_traffic/scat_states.parquet \
  --clearances data/processed/scat/03_clearances/scat_clearances.parquet \
  --alt-unit ft --inspect

python3 src/scat_analysis/stage21_maneuver_screening.py \
  --stage20 results/stage20 \
  --tracks data/processed/scat/02_traffic/scat_states.parquet \
  --clearances data/processed/scat/03_clearances/scat_clearances.parquet \
  --alt-unit ft --pilot --pilot-week 20161015_20161021 \
  --pilot-per-stratum 20 --seed 42 --out results/stage21/pilot

python3 src/scat_analysis/stage21_maneuver_screening.py \
  --stage20 results/stage20 \
  --tracks data/processed/scat/02_traffic/scat_states.parquet \
  --clearances data/processed/scat/03_clearances/scat_clearances.parquet \
  --alt-unit ft --pilot-summary results/stage21/pilot/stage21_summary.json \
  --out results/stage21
```

The pilot draws 20 calibration episodes from each of six strata: red with no
order, prior only, later only, or both; orange; and yellow. The full run keeps
calibration and test separate. Both use a 20-minute horizon, synchronized
10-second samples, a maximum 30-second interpolation gap, and at least 80%
coverage with both endpoints observed. Missing future points are never
extrapolated. A conclusion that predicted loss was not observed requires
complete coverage of the 20-minute grid. The outputs are
`stage21_summary.json`, `stage21_by_color.csv`,
`stage21_encounters.csv`, `stage21_clearance_maneuvers.csv`,
`stage21_counterfactuals.csv`, `STAGE21_VALIDATION_REPORT.md`, and example
figures in `results/stage21/`. See the
[Stage 21 validation report](../../results/stage21/STAGE21_VALIDATION_REPORT.md)
for denominators, red episodes without linked orders, maneuver screening and
publication limits.

### Interpretation and data checks

- Stage 20 attributes a clearance to every pair containing that aircraft
  within the time window. This does not establish that the order responded to
  that encounter. Repeated observations and unique clearance events are
  reported separately. Red/orange labels are predictions, not realized
  incidents.
- Stage 21 compares Stage 16's predicted minimum at the first snapshot of an
  episode's highest color with observed horizontal and vertical separation at
  the same subsequent instants. Its pre-order, constant-velocity projection is
  a screening reference only. A compatible maneuver and a larger observed
  separation do not establish that the order prevented a conflict.
- These outputs are neither calibrated risk estimates nor verified resolved
  conflicts. Before interpreting full-run counts, check coverage of the 91
  nominal days and confirm that the Stage 16–18 FL290–410, 350 NM, and 120 s
  filters were applied. Stage 20 audits attribution; it does not rebuild pairs
  or correct historical selections.
- Stage 21 marks encounters with insufficient synchronized surveillance as
  excluded from observed-minimum estimates. A route string may indicate
  planned level information, but it cannot establish when a turn or level
  change was planned. All maneuver and inertial results need operational
  review before publication.

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

