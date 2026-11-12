# Reproducing the CEAS figures

Run from the repository root with Python 3 and `numpy`, `pandas`, `pyarrow`,
`matplotlib`, `scipy`, and `scikit-learn` installed:

```bash
python3 paper/figure_scripts/benchmark.py
python3 paper/figure_scripts/audit_supplements.py
python3 paper/figure_scripts/coverage_causes.py
python3 paper/figure_scripts/generate.py --figure all
```

`--figure 1` through `--figure 8` regenerate individual figures. The generator
writes PNG panels to `paper/figures/` and never changes Stage 1–21 results.
`benchmark.py` writes its cohort, predictions, scores, reliability curves,
bootstrap intervals, paired weekly differences, alert burden and decision to
`results/benchmarks/`. The random seed is **20261112**. Its approved protocol
is recorded in `docs/decisiones_protocolo.md`.

To reconstruct the approved green-inclusion sensitivities without overwriting
the primary results, run:

```bash
python3 paper/figure_scripts/benchmark.py --green-limit 3 --out results/benchmarks_green3
python3 paper/figure_scripts/benchmark.py --green-limit inf --prepare-only --out results/benchmarks_green_all
python3 paper/figure_scripts/sensitivity_green_all.py
python3 paper/figure_scripts/benchmark.py --gap 240 --out results/benchmarks_gap240
python3 paper/figure_scripts/benchmark.py --gap 480 --out results/benchmarks_gap480
python3 paper/figure_scripts/sensitivity_broad_orders.py
```

The canonical manuscript is `paper/ceas_manuscript_figures.tex`. It is updated
with verified values from the primary benchmark. Do not rerun model selection
on test weeks or edit the frozen Stage 16–21 inputs.
The no-limit green sensitivity freezes the primary M4 regularisation choice
and reports only M0 and M4, since its 3.39 million green episodes make a new
full leave-one-week-out model grid costly.

The manuscript and plotted labels describe the operations without numbered
pipeline stages. Stage identifiers in this README and the code are retained
only to locate the exact input files and reproduce the outputs.

## Inputs and parameters

| Figure | Source | Key parameters |
| --- | --- | --- |
| 1 | Stage 03, 16–21 scripts and Stage 19 split manifest | Nine calibration and four test weeks; workflow only |
| 2 | `data/processed/scat/19_calib_selection/tables/heat_vs_green_R.csv`, `heat_pairs_R.csv`, `audit.json` | Route predictor; 5-minute label; density-adjusted lift against **final green** calculated on calibration weeks |
| 3 | Stage 20 pair export, Stage 21 encounter and manoeuvre tables, Stage 02 surveillance and predicted-trajectory Parquets | Encounter **33**, week **20161015_20161021**, anchor **2016-10-16 12:58:00 UTC**, 20-minute/10-second horizon; no interpolation across gaps over 30 s; no extrapolation |
| 4 | Stage 20 colour counts, Stage 19 `scheme_tiers.csv`, Stage 16 daily cell Parquets | 300-s later eligible-order label; 400 whole-day bootstrap replicates for rate intervals; reported Stage 19 day-bootstrap intervals for lift |
| 5 | Stage 20 clearance-attribution CSV, canonical Stage 20 pair Parquets, Stage 03 clearance Parquet | Exact original clearance epochs; strict `(T,T+300 s]` association; all 13 weeks |
| 6 | Stage 20 episodes CSV | Test red episodes; prior 600 s before episode start; later 300 s after episode observations |
| 7 | Stage 21 encounters CSV | All 13 weeks; observed minimum evaluable at ≥80% synchronized 10-s grid coverage and both endpoints; absence claim requires 100% coverage |
| 8 | `results/benchmarks/metrics.csv`, `reliability.csv` | One episode, one vote; 400 paired whole-day bootstrap draws; M0–M5 |

### Example encounter

Encounter 33 contains source flight keys
`scat20161015_20161021:100048` and
`scat20161015_20161021:102654`. It has two linked eligible orders, both after
the anchor. The plotted Stage 16 route forecast is replayed from the latest
available surveillance states and prediction updates at the anchor. Its minimum
`m=0.3942360998` at `s=210 s` matches the stored Stage 21/Stage 16 values.
The synchronized observed minimum is `m=1.0`; all 121 grid points are covered.
Vertical lines indicate the two issue times and their Stage 21 compatible
manoeuvre-onset proxies. Neither temporal association nor compatibility proves
that the orders resolved this encounter.

### Validation built into the generator

- Stage 16 daily counts (restricted to each nominal archive week) must exactly
  equal Stage 20 pair-instant and positive-label counts for every split/colour.
- Stage 20 attribution must contain 51,950 unique eligible orders, 47,339 with
  a link and 3,176,638 order–pair-instant links, with linked-order median 65 and
  90th percentile 108.
- The Figure 5 distinct-pair reconstruction checks the exact number of linked
  pair-instants for **each** of the 51,950 orders against Stage 20. Fractional
  timestamps from Stage 03 are required: the rounded `time_utc` string in the
  attribution CSV changes matches at 300-second boundaries.
- Red test episode categories must sum to 1,863; Stage 21 colour coverage and
  red/orange outcome partitions must match the saved tables.
- The Figure 3 forecast minimum and observed minimum must reproduce the stored
  encounter values, with complete observed grid coverage.

## Pending comparison

Figure 8 plots the completed M0–M5 held-out comparison. M6 was excluded by
author decision because a conflict probability does not directly estimate the
recorded-order outcome. The primary outcome is an eligible order in the 300 s
after episode start. The any-observation outcome is reported separately to show
the effect of unequal exposure. Non-green episodes retain their frozen Stage 20
IDs; green-only runs exclude green observations within 360 s of a non-green
observation for the same pair. This is a conservative reconciliation rule, not
a claim that removed green observations were risk free.

`audit_supplements.py` reproduces the selection funnel, compares eligible and
broader orders by week/type/matched flight level, and classifies observable
surveillance-coverage patterns. It cannot infer whether a missing
end state arose from area exit, archive boundary, track gap or another cause.
`coverage_causes.py` scans the original Stage 02 individual tracks and writes
one evidence category per unevaluable episode to
`results/benchmarks/coverage_evidence_detail.csv`. It distinguishes the last
observed altitude band, missing starts, archive boundaries and track endings.
Without a Swedish airspace polygon, an in-band track end remains geographically
indeterminate.
The trajectory-predictor sensitivities in the task list remain outstanding.
The broader order sensitivity relabels the same episodes with all 908,165
extracted heading/level records and writes to `results/benchmarks_broad_orders/`.
The fixed
120/300/600-s outcome-window and 300/600-s prior-window variants are produced by
`sensitivity_windows.py` and summarised in `results/benchmarks/window_sensitivity.csv`.

## Compilation

The official December 2024 Springer Nature `sn-jnl.cls` is stored in `paper/`
next to the manuscript. Its [official download page](https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940)
provides the template package. From `paper/`, compile with a TeX engine that
supports the class, for example:

```bash
cd paper
tectonic ceas_manuscript_figures.tex
```

The manuscript uses an in-file bibliography, so BibTeX is not needed.
