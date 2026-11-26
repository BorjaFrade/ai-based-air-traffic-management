# Robustness by unit and clearance multiplicity

Run from the repository root, with `python3`, in this order:

```bash
python3 paper/figure_scripts/units_table6.py
python3 paper/figure_scripts/units_binary.py
python3 paper/figure_scripts/units_multiplicity.py
```

The first command checks the 24 rates and lifts published in Table 6 and stops the analysis if any of them changes at the displayed rounding. All of them were reproduced. It uses the four test weeks of the Stage 19 manifest, the following 300 s, and a green class unrestricted in `m_min`.

| Table 6 column | File and function | Main source |
| --- | --- | --- |
| Pair-instant | `units_table6.py:pair_instant_column`; calls `stage17_colour_schemes.tier_stats` | `16_pair_hazard/tables/cells_*.parquet`, nominal days |
| Episode, 300 s from start | `units_table6.py:episode_columns`, label `y_fixed` | `results/benchmarks_green_all/episodes.parquet` |
| Episode, 300 s from any observation | `units_table6.py:episode_columns`, label `y_exposure` | The same Parquet file |

The standardisation uses the observed number of positives divided by the number expected if each stratum had the rate of the reference group. There are four traffic strata per pair-instant and three per episode. `table6_reproduction.csv` contains the unrounded values. The CIs of the pair-instant column in that CSV use the 400 replicates and the seed **20261112** requested here; they may differ from the old Stage 17 CIs, which were computed with a different random sequence.

## Short text for the results

In the four test weeks, `m_min < 1` identifies 8,823 pair-instants, of which 2,974 receive an eligible clearance in the following 300 s (33.71%). Among the 6,848,057 negatives the rate is 13.54%, a crude ratio of **2.49** and a standardised lift of **2.48** (95% CI 2.36–2.60). In the common cohort of episodes, the lift falls to **1.83** (1.75–1.90) for the window from the start and to **1.25** (1.21–1.29) for the window from any observation. Their paired ratios with respect to the pair-instant lift are **0.74** (0.70–0.77) and **0.50** (0.49–0.52). When the positive runs are rebuilt with the binary rule, the ratios are 0.75 (0.72–0.78) and 0.45 (0.44–0.46).

| `c` | Episodes | Window | Pair-instant lift | Episode lift | Episode/pair-instant lift (95% CI) |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Common | Start | 2.48 | 1.83 | 0.74 (0.70–0.77) |
| 1 | Common | Any observation | 2.48 | 1.25 | 0.50 (0.49–0.52) |
| 1 | Binary runs | Start | 2.48 | 1.86 | 0.75 (0.72–0.78) |
| 1 | Binary runs | Any observation | 2.48 | 1.11 | 0.45 (0.44–0.46) |
| 1.5 | Common | Start | 1.56 | 1.51 | 0.97 (0.94–1.00) |
| 1.5 | Common | Any observation | 1.56 | 1.17 | 0.75 (0.73–0.76) |
| 1.5 | Binary runs | Start | 1.56 | 1.47 | 0.94 (0.91–0.97) |
| 1.5 | Binary runs | Any observation | 1.56 | 1.07 | 0.69 (0.67–0.70) |
| 2 | Common | Start | 1.54 | 1.44 | 0.93 (0.91–0.95) |
| 2 | Common | Any observation | 1.54 | 1.02 | 0.66 (0.65–0.67) |
| 2 | Binary runs | Start | 1.54 | 1.43 | 0.93 (0.90–0.95) |
| 2 | Binary runs | Any observation | 1.54 | 1.04 | 0.68 (0.67–0.68) |

Each ratio and each difference of logarithms is computed **within the same replicate**, with the same 28 resampled days at both levels. `binary_rates_lifts.csv` gives `n`, positives, rates and CIs by class, threshold, unit, window and episode variant. `binary_lift_ratios.csv` gives the ratios and the logarithmic differences with CIs. The negatives include yellow and the green band `1 <= m_min < 1.2` when they do not meet the rule `m_min < c`. The binary-runs variant rebuilds the positive episodes from all observations with `m_min < c` and a maximum gap of 360 s; it keeps as the negative reference the episodes of the common cohort whose minimum is `>= c`. This preserves the unrestricted green of Table 6 before reclassifying it.

The multiplicity of the full scenario at 350 NM reproduces **47,339** of 51,950 clearances with links, **4,611** without links and **3,176,638** links. Each linked clearance labels a median of **65** pair-instants (p90 **108**) and **33** distinct aircraft pairs (p90 **47**). Restricting to non-green observations at 350 NM leaves **18,668** linked clearances and **38,729** pair-instant links; with `m_min < 1`, **6,939** and **11,129**. The nine scenario-by-radius combinations are in `multiplicity_pairs.csv`; `multiplicity_by_clearance.csv.gz` keeps the counts of each clearance and allows the distributions to be audited. The counts at 350 NM match Stage 20 **row by row**; the episode links at 350 NM match Stage 21 row by row.

| Initial radius | Clearances linked to non-green episodes | Episode–clearance links | Clearances in >1 episode | In >1 colour |
| ---: | ---: | ---: | ---: | ---: |
| 350 NM | 28,324 | 45,463 | 11,474 (40.5%) | 6,383 |
| 100 NM | 24,405 | 35,136 | 8,015 (32.8%) | 4,056 |
| 50 NM | 20,680 | 27,730 | 5,587 (27.0%) | 2,598 |

`multiplicity_episodes.csv` includes the zeros among the 51,950 clearances. The episode links follow the Stage 21 definition: a clearance in the 600 s before the start or in the 300 s after any observation. For this reason, the total number of clearances linked to episodes can exceed that of pair-instant links, which uses only later clearances. For each radius, **the observations are filtered by initial horizontal separation before the episodes are built** and the same 360 s gap is then applied. The initial separation is reconstructed from the last surveillance states used by Stage 16, with its azimuthal equidistant projection. The means, medians and p90 values in `multiplicity_pairs.csv` use as denominator the clearances with at least one link in each scenario and radius; clearances without links are reported separately.

## Inputs, decisions and time

- Inputs: `data/processed/scat/19_calib_selection/tables/audit.json`; `data/processed/scat/16_pair_hazard/tables/cells_*.parquet`; `results/benchmarks_green_all/episodes.parquet`; `results/stage20/pairs/pairs_*.parquet`; `results/stage20/stage20_clearance_attribution.csv.gz`; `data/processed/scat/02_traffic/scat_states.parquet`; `data/processed/scat/03_clearances/scat_clearances.parquet`.
- The clearance links use the original fractional epoch of Stage 03, because the `time_utc` text of Stage 20 is rounded to seconds and alters the cases at the 300 s edge. Distinct aircraft pairs are counted by flight identity and source member, without duplicating temporal observations.
- Bootstrap: 400 resamples of **whole days**, seed **20261112**; 2.5th and 97.5th percentiles. The original traffic strata of each unit are kept. For the new binary runs, the stratum comes from the hourly count of episode starts of the frozen cohort at the time of the new start and from the calibration cut-offs (2012, 2939).
- Measured run time: reproduction of Table 6 ~0.3 s; binary comparison **5.3 s**; multiplicity **87.0 s** (including the loading of states and the reconstruction of radii). Local platform, Python 3.
