# Encounter-level benchmark: validation record

Protocol choices fixed on 12 November 2026 and recorded locally in `docs/decisiones_protocolo.md` before the first run of the benchmark; the file is not versioned. The test weeks were scored repeatedly during development, so results are exploratory held-out analyses. Source inputs are the frozen Stage 03 and Stage 16–21 outputs. Scripts are in `paper/figure_scripts/`; new outputs are under `results/benchmarks/` and sensitivity-specific directories. The four test weeks were scored with calibration-fitted models and thresholds.

## Primary cohort and target

- **67,367 episodes:** 12,926 green-only, 39,351 yellow, 9,039 orange and 6,051 red. The non-green counts exactly reproduce Stage 20; 1,863 red episodes fall in test.
- **21,281 test episodes.** Primary outcome: at least one of the 51,950 eligible orders in the fixed `(T0,T0+300 s]` window from episode start. A second outcome uses a 300-s window after any observation.
- Green-only runs require `m_min < 2` and exclude observations within 360 s of an audited non-green observation for the same pair. This preserves existing non-green IDs. The all-observation outcome is calculated from individual observed green instants, not by treating the interval between episode start and end as continuously exposed.
- Primary episode-weighted test rates for fixed versus any-observation windows: green 16.2%/26.0%, yellow 21.5%/30.5%, orange 27.7%/41.0%, red 30.9%/45.9%. These are eligible-order rates, not conflict rates.

## Held-out comparison

M4 was selected among M2–M4 by minimum leave-one-week-out calibration Brier score. All models use the same cohort and fixed outcome. M4 uses spline geometry, preceding-hour episode-start count, prior eligible order and log episode duration; its density feature does not use future episode starts.

**Interpretation limit:** maximum episode colour, minimum geometry and duration use the completed episode. They may include observations later than the start-based order window. The scores compare retrospective associations on held-out weeks; they do not measure a deployable prediction available at episode start. A decision-time model requires a separate prospective feature construction.

| Weight | M0 Brier | M4 Brier | Paired difference | 95% day-bootstrap interval | SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| One episode, one vote | 0.171235 | 0.169040 | −0.002196 | [−0.002977, −0.001489] | 0.000396 |
| Pair-instant weight | 0.148185 | 0.146677 | −0.001508 | [−0.002048, −0.000956] | 0.000295 |

The one-SE decision rule favours M4 for predicting subsequent eligible orders. Four paired week-level differences are all negative but remain descriptive. The test alert burden is 17,338 distinct non-green episodes across 605 hours containing at least one such episode (28.7 per active hour). Figure 8 reads `metrics.csv` and `reliability.csv`; it plots no simulated values.

## Audits and sensitivities

- Selection funnel: 1,019,799 actionable rows → 908,165 heading/level records → 51,950 eligible orders (50,774 level, 1,176 heading). `audit_supplements.py` checks those counts and the eligible/broader week, type and matched-flight-level table.
- Coverage evidence: of 6,051 red episodes, 1,744 are evaluable. Among 4,307 unevaluable red episodes, 479 have an aircraft last observed outside FL290–FL410, 3,823 have an individual track ending before the 20-minute endpoint while last observed inside that band, and five lack an observed start. Across all 26,990 unevaluable episodes, the track-level audit found no endpoint bracketed by a later observation across a gap. Without an operational Swedish airspace polygon, an in-band track end cannot be confirmed as a geographic exit.
- Fixed later window 120 s: red rate 13.7% falls below orange 14.4%; colour monotonicity is lost. M4 remains selected and improves Brier by 0.00161. At 600 s the ordering returns and M4 improves Brier by 0.00220. A 300-s prior window retains M4 with an improvement of 0.00213.
- Episode gaps of 240 and 480 s rebuild both green and non-green runs. Test colour rates remain ordered and M4 remains calibration-selected; episode-weighted Brier improvements are 0.00156 and 0.00251, respectively. The non-green counts appropriately differ from the frozen 360-s audit.
- Relabelling the same episodes with all 908,165 extracted heading/level orders changes the test target: green/yellow/orange/red fixed-window rates become 23.6%/36.7%/58.3%/34.5%. Red falls below orange. M4 remains selected and improves episode-weighted Brier by 0.04181 versus M0, but this magnitude is not directly comparable to the eligible-order target.
- Green limit `m_min < 3`: 83,959 green episodes, M4 remains selected; its episode-weighted Brier improvement is 0.00138. With no green limit, 3,390,948 green episodes are constructed; test green order rate is 15.8%. The primary M0/M4 specifications refitted on this enlarged calibration cohort score 0.13355/0.13247 (difference −0.00108). The no-limit run freezes M4's primary regularisation choice to avoid a very large new model-selection grid.

## Remaining work before submission

Alternative trajectory-predictor sensitivities have not been run. The track-level follow-up audit distinguishes altitude-band exits, missing starts, archive boundaries and track endings in-band, but cannot confirm geographic exits without a polygon or identify why a track ended in-band. Consequently the manuscript reports observable evidence and does not claim a missingness correction. The M6 conflict-probability model was intentionally deferred by the approved protocol. Author and policy TBC items, release tagging and Zenodo deposition still need the authors' final submission workflow.
