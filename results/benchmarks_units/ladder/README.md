# Lift ladder by unit of analysis

The four test weeks (28 days), the **6,856,880** pair-instants of the Stage 20 export and only the **51,950** eligible clearances were analysed. The prior check read the existing outputs of `units_table6.py` and `units_binary.py`: Table 6 still reproduces 1.53/2.23/3.11 per pair-instant, 1.37/1.75/1.96 per episode from the start and 0.89/1.19/1.33 from any observation. The ladder code does not rewrite those CSV files.

## Reading the steps

| Step | Unit, colour and window | Red lift, instantaneous traffic (95% CI) | Positive `m < 1` lift (95% CI) |
| --- | --- | ---: | ---: |
| 0 | All observations; instantaneous colour; 300 s after `t` | 3.11 (2.92–3.32) | 2.48 (2.36–2.60) |
| 0b | As 0, with weight `1/n_e` per observation | 2.66 (2.52–2.82) | 2.26 (2.14–2.36) |
| 1 | First observation of each episode; its colour and window | 2.03 (1.91–2.16) | 1.80 (1.72–1.89) |
| 2 | One episode; maximum colour; 300 s from the start | 1.94 (1.83–2.06) | 1.81 (1.74–1.89) |
| 3 | As 2, with 300 s from the first instant of the maximum | 2.28 (2.16–2.42) | 1.85 (1.77–1.93) |

The additional reference `2_any` (300 s after **any** observation) gives, with the common instantaneous stratifier, lifts of **0.88/1.18/1.32** for yellow/orange/red. When the whole ladder is repeated with the hourly episode stratifier, red is **3.12** at step 0, **1.95** at step 2 and **2.29** at step 3; the qualitative conclusion does not change. All numbers, including the less favourable ones, the binary thresholds `c=1, 1.5, 2`, both stratifications and `2_any`, are in `ladder_rates_lifts.csv`.

The paired ratios of the red lift are: `0b/0` **0.86** (0.83–0.88), `1/0` **0.65** (0.62–0.69), `2/0` **0.62** (0.59–0.66), `3/2` **1.18** (1.12–1.24) and `3/0` **0.73** (0.70–0.77). Retrospective anchoring at `T_max` increases the red lift with respect to step 2, but for the binary screens `c=1.5` and `c=2` it reduces it: **1.51 → 1.46** and **1.45 → 1.42** respectively, under instantaneous traffic. `ladder_ratios.csv` contains all the ratios and the differences `log(L_B) − log(L_A)` with CIs.

**Difference with respect to Table 6.** The original `green_all` cohort excluded **60,700** green test observations located 360 s or less from non-green observations of the same pair. The current requirement that every observation belong to an episode led to keeping those green observations in green runs of their own. The result is **1,090,998** green episodes, **6,297 more** than in Table 6; the **17,338** non-green test episodes keep their original construction (54,441 over the 13 weeks). This raises the green episode rate and gives a red lift of **1.94** with instantaneous stratification and **1.95** with hourly stratification, against the published **1.96**. The complete common cohort was explicitly given priority for the table and the figure. No observation or episode of that cohort was excluded. Step 1 and the episode-level steps have fewer **evaluation rows** because they count one per episode; the CSV distinguishes `n_obs` (rows evaluated), `underlying_obs` (observations represented) and `n_episodes`.

## Computation decisions

- Pairs are identified by week, flight key and source member. Green and non-green runs are formed separately; a run is cut when the next instant of the same pair is **more** than 360 s away. In this way every observation has a single episode and the non-green ones remain as in Stage 20.
- `T0` is the first observation of the run. The maximum colour follows the order green < yellow < orange < red. `T_max` is the first instant with that maximum; in the binary schemes it is the first `m_min < c`, or `T0` for negative episodes. Ties at the same timestamp follow the original order of the Parquet file. In **1,107,130 of 1,108,336** test episodes, `T_max = T0`; in the other **1,206** it is later. The `T_max` date of the 17,338 non-green test episodes coincides exactly with `anchor_time_utc` of Stage 21. Step 3 is **retrospective**: the maximum is known only after the evolution of the episode has been observed.
- The label per observation is an eligible clearance in `(t,t+300 s]`; the labels of steps 1 and 2 use the label of the first observation, step 3 that of `T_max`, and `2_any` the logical maximum over all observations. The binary positive group is `m_min < c` and the negative group is everything else, including yellow and `1 <= m_min < 1.2` where applicable. A binary episode is positive if any observation is positive.
- The main stratifier is the global number of eligible aircraft in FL290–FL410 at the reference instant, reconstructed with the state selection of Stage 16 and checked against its exact daily counts: `[0,15)`, `[15,25)`, `[25,35)` and `[35,∞)`. The `episode_hour` repetition counts episode starts in the hour before the reference instant, with the frozen calibration cut-offs **2012 and 2939**. Both use the **same** definition within each ladder, also in the pair-instant steps.
- The lift divides the observed positives by those expected when the green (or negative) rate is applied, stratum by stratum, to the weights of the class. `positives` is the weighted numerator; in 0b it can be fractional. `positive_rows` additionally keeps the number of unweighted positive labels. The CIs are the 2.5th and 97.5th percentiles of **400** replicates with seed **20261112**, resampling the same 28 whole days for all steps and computing each ratio and difference of logarithms within the same replicate.

## Values drawn in Figure 11

Both panels use the common instantaneous stratifier. Each cell shows the lift (95% CI).

| Step | Yellow | Orange | Red | Binary `m < 1` |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.53 (1.49–1.59) | 2.23 (2.11–2.34) | 3.11 (2.92–3.32) | 2.48 (2.36–2.60) |
| 0b | 1.69 (1.63–1.76) | 2.06 (1.94–2.18) | 2.66 (2.52–2.82) | 2.26 (2.14–2.36) |
| 1 | 1.39 (1.34–1.45) | 1.71 (1.62–1.80) | 2.03 (1.91–2.16) | 1.80 (1.72–1.89) |
| 2 | 1.36 (1.30–1.42) | 1.74 (1.64–1.83) | 1.94 (1.83–2.06) | 1.81 (1.74–1.89) |
| 3 | 1.36 (1.30–1.42) | 1.74 (1.64–1.83) | 2.28 (2.16–2.42) | 1.85 (1.77–1.93) |

The PNG files are `paper/figures/fig11a_ladder_four_class.png` (**1001 × 872 px**) and `paper/figures/fig11b_ladder_binary.png` (**1001 × 877 px**), both at 300 dpi. `ladder_table.tex` is a four-class `booktabs` fragment with both stratifications and the `2_any` reference.

## Inputs, execution and outputs

Inputs: `data/processed/scat/19_calib_selection/tables/audit.json`; `data/processed/scat/02_traffic/scat_states.parquet`; `data/processed/scat/16_pair_hazard/tables/cells_*.parquet`; `results/stage20/pairs/pairs_*.parquet`; `results/stage20/stage20_episodes.csv`; `results/stage20/stage20_non_green_observations.csv.gz`; `results/stage21/stage21_encounters.csv`; `results/benchmarks_green_all/episodes.parquet`; and the existing CSV files `results/benchmarks_units/table6_reproduction.csv` and `binary_rates_lifts.csv` for the prior check.

Exact command from the repository root:

```bash
python3 paper/figure_scripts/units_ladder.py
python3 paper/figure_scripts/generate.py --figure 11
```

Measured time: **24.0 s** for the computation and the CSV/LaTeX files; **9.5 s** for the figure. `diagnostics.json` keeps the weekly breakdown and the computation time. The `fig11()` function only reads `ladder_rates_lifts.csv`; it checks that red at step 0 reproduces 3.11 and that red at step 2 keeps the agreed **1.94 of the common cohort**. It does not recompute the lifts.
