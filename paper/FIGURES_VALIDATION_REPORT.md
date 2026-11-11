# CEAS figure validation report

Validation date: 11 November 2026. The supplied single-file LaTeX draft was
preserved as `ceas_manuscript_original.tex`; the figure-integrated copy is
`ceas_manuscript_figures.tex`.

| Fig. | Generated file (PDF and matching PNG) | Data source | Validation | Notes |
| --- | --- | --- | --- | --- |
| 1 `fig:workflow` | `figures/fig01_workflow.*` | Stage 03, 16–21 scripts and Stage 19 split manifest | Pass | Observed and calculated streams distinguished; M0–M6 marked planned. |
| 2 `fig:colors` | `figures/fig02_color_calibration.*` | Stage 19 `heat_vs_green_R.csv`, `heat_pairs_R.csv` | Pass | Nine calibration weeks only; five-minute lift uses traffic-density-specific **final green** rates, not exploratory far-pair reference. The 1–1.2 band is green. |
| 3 `fig:example` | `figures/fig03_encounter_example.*` | Stage 02 surveillance and predictions; Stage 20/21 encounter and manoeuvre tables | Pass | Encounter 33, week 20161015_20161021; two linked orders; complete 121-point observed grid. Replayed forecast minimum 0.3942360998 at 210 s agrees with Stage 21. |
| 4 `fig:lift` | `figures/fig04_clearance_rates.*` | Stage 16 daily cells, Stage 19 final tiers, Stage 20 colour counts | Pass | Stage 16/20 per-colour counts agree exactly. Rate intervals recomputed by 400 whole-day bootstrap replicates; lift intervals taken from Stage 19. Different estimands are labelled. |
| 5 `fig:multiplicity` | `figures/fig05_clearance_multiplicity.*` | Stage 03 exact clearance epochs; Stage 20 attribution and pair Parquets | Pass | Every one of 51,950 per-order link totals matches Stage 20 after reconstructing distinct pairs. Counts, median, and 90th percentile verified. |
| 6 `fig:episodeorders` | `figures/fig06_red_episode_orders.*` | Stage 20 episodes and colour counts | Pass | 642 + 213 + 187 + 821 = 1,863 test red episodes; 42.2% and 45.9% use distinct denominators. |
| 7 `fig:coverage` | `figures/fig07_surveillance_coverage.*` | Stage 21 encounters and by-colour summary | Pass | 19,905/39,351 yellow, 5,802/9,039 orange, 1,744/6,051 red evaluable. One red evaluable episode has partial coverage and cannot support an absence claim. |
| 8 `fig:benchmarks` | **Not generated** | No M0–M6 result table or benchmark run | Pending | Existing placeholder and label retained; no values inferred or test-week experiments run. |

## Numeric and temporal checks

- Total pair-instants: **21,567,488** over the 91 nominal days. Calibration
  contains **14,710,608** pair-instants with **1,928,099** later-order labels;
  test contains **6,856,880** with **930,069** labels. Stage 20 reports zero
  disagreements with original Stage 16 300-s binary labels.
- Four-class calibration and test rates reproduce the Stage 20 table:
  green **13.04595% / 13.50107%**, yellow **20.40271% / 20.72653%**,
  orange **28.23356% / 30.13831%**, red **38.13690% / 42.22649%**.
  The percentage bars represent pair-instants with an eligible heading/level
  order in the next 300 s, not independent encounters, conflicts or incidents.
- Attribution: **51,950** distinct eligible orders, **47,339** with at least
  one pair-instant, **3,176,638** order–pair-instant relations. For linked
  orders the median is **65** and the 90th percentile is **108**.
- Example 33 uses anonymised flight keys
  `scat20161015_20161021:100048` and
  `scat20161015_20161021:102654`, anchored at
  **2016-10-16 12:58:00 UTC**. Its full 20-minute Stage 16 route forecast was
  replayed from only states and prediction updates available by the anchor.
  Observed separation was evaluated at simultaneous UTC times on a 10-s grid.
  The Stage 21 interpolation limit of 30 s and no-extrapolation rule were used;
  this example has no surveillance gap on that grid.
- Stage 21 evaluated **27,451** of **54,441** non-green encounters. Among
  evaluable orange episodes, **470** had observed `m<1` and **5,332** had
  complete coverage without observed `m<1`. Among evaluable red episodes,
  the analogous counts are **703**, **1,040**, and **one** partial-coverage
  residual. These categories do not establish false alerts or causal effects.

## Documented manuscript discrepancies and corrections

1. **Primary predictor altitude.** The supplied draft said that the Stage 16
   route predictor retains current altitude. `stage16_pair_hazard.py` and the
   replayed example show that it follows the predicted `afl_value` profile
   where a usable route exists, with constant-motion fallback otherwise.
   The integrated draft corrects this verifiable methodological statement.
2. **Test green reference.** `stage17_colour_schemes.py::tier_stats` calculates
   density-specific green rates inside each split. The test lift therefore
   uses test green reference rates re-estimated on the four test weeks. This
   is descriptive evaluation, not a frozen-rate prospective prediction. The
   draft contained a `TBC` request to confirm this; the integrated copy states
   the verified rule.
3. **The 61 no-linked-order red test episodes.** The original draft noted
   `46 + 13 = 59`, leaving two unexplained. Direct row-level inspection of
   `stage21_encounters.csv` found that episode IDs **12232** and **37261**
   have no Stage 03 order in the preceding 30 minutes but do have one in the
   episode/five-minute follow-up. The integrated text gives this reconciliation.
4. **Uncertainty wording.** The Stage 19 final-tier file contains
   day-bootstrap intervals for *lift*, not for the plotted raw rates. Figure 4
   derives rate intervals anew from daily Stage 16 cell counts using 400
   whole-day draws. It does not use binomial intervals.
5. **Model comparison.** The manuscript protocol leaves green-episode
   inclusion, the M6 error model and the decision threshold unresolved. No
   M0–M6 held-out result exists. Figure 8 remains a clearly labelled
   placeholder.

## Render and integration checks

All seven PDF figures were generated as vectors and paired with 300-dpi PNGs.
The figure PNGs were visually reviewed for clipping, overlap, readable axes,
consistent class colours and grayscale patterns. The integrated manuscript
compiled with the official December 2024 Springer Nature `sn-jnl.cls` using
Tectonic. The resulting **22-page PDF** was rendered and figure pages 6, 9,
11–15 and benchmark page 19 were visually inspected. The compiler log has no
`Overfull`, missing-character, undefined-reference or LaTeX-error messages.
Figures 1–7 have intact captions and labels; the sole remaining figure
placeholder is Figure 8. Ordinary underfull-box warnings remain from the
template's pagination and line breaking.

The manuscript remains an internal scientific draft with author and policy
`TBC` items and pending M0–M6 results. No Stage 1–21 output was modified.

## Reader-facing terminology revision

The integrated manuscript and its figures use descriptions of the operations
instead of internal numbered stage identifiers. The clearance-eligibility
paragraph now states the actual level- and heading-order filters verified in
the preprocessing scripts, including the 10-s surveillance age limit, the
FL290–FL410 band and the 50,774 + 1,176 event counts. Numbered stage names
remain only in reproduction documentation, source filenames and the preserved
original draft. The revised 22-page PDF was recompiled and checked for any
remaining numbered-stage text, overflow or unresolved references.
