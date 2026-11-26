# Robustness of the lift ladder

Command: `python3 paper/figure_scripts/units_robustness.py`.
Total measured time: 68.7 s. Seed 20261112; 400 whole-day bootstrap replicates in each run.
B0: maximum differences with respect to the published CSV files: rates/lifts 0; ratios 0 (tolerance 1e-12).

The calibration weeks are read from `data/processed/scat/19_calib_selection/tables/audit.json`. The thresholds of the four levels were fitted on these weeks; the binary screen m<1 has no fitted parameters.
The green/negative reference is re-estimated on the weeks of each run. The instantaneous FL290–FL410 strata [0,15), [15,25), [25,35), [35,∞) are used. Ratios and differences of logarithms are computed within each common replicate.
On test with gap 360, the frozen checks of Stage 20/21, Table 6 and the population of 6,856,880 are applied. In the other scenarios, the checks are: unique assignment of observations, total weight of Step 0b equal to the number of episodes, and first red maximum / first m<1 equal to k_cls=1. The stratum counts are checked against Stage 16 in each week.

| Run | Class | 0 | 1 | 2 | 3 | 1/0 | 2/0 |
|---|---|---|---|---|---|---|---|
| test_gap360 | red | 3.11 (2.92–3.32) | 2.03 (1.91–2.16) | 1.94 (1.83–2.06) | 2.28 (2.16–2.42) | 0.65 (0.62–0.69) | 0.62 (0.59–0.66) |
| test_gap360 | positive | 2.48 (2.36–2.60) | 1.80 (1.72–1.89) | 1.81 (1.74–1.89) | 1.85 (1.77–1.93) | 0.73 (0.69–0.76) | 0.73 (0.70–0.76) |
| gap240 | red | 3.11 (2.92–3.32) | 2.01 (1.89–2.14) | 1.96 (1.86–2.08) | 2.28 (2.16–2.42) | 0.65 (0.61–0.69) | 0.63 (0.60–0.67) |
| gap240 | positive | 2.48 (2.36–2.60) | 1.79 (1.71–1.88) | 1.82 (1.74–1.90) | 1.84 (1.77–1.91) | 0.72 (0.69–0.76) | 0.73 (0.70–0.77) |
| gap480 | red | 3.11 (2.92–3.32) | 2.03 (1.91–2.15) | 1.92 (1.81–2.03) | 2.28 (2.17–2.42) | 0.65 (0.62–0.69) | 0.62 (0.58–0.65) |
| gap480 | positive | 2.48 (2.36–2.60) | 1.81 (1.73–1.89) | 1.81 (1.74–1.89) | 1.84 (1.77–1.92) | 0.73 (0.70–0.76) | 0.73 (0.70–0.76) |
| calibration_gap360 | red | 2.91 (2.80–3.05) | 1.97 (1.84–2.08) | 1.80 (1.71–1.90) | 2.11 (2.03–2.21) | 0.68 (0.64–0.71) | 0.62 (0.59–0.64) |
| calibration_gap360 | positive | 2.38 (2.32–2.45) | 1.72 (1.65–1.78) | 1.74 (1.67–1.80) | 1.76 (1.70–1.82) | 0.72 (0.70–0.74) | 0.73 (0.71–0.75) |

| Run | Observations | Green episodes | Non-green episodes | Red obs. | Red episodes |
|---|---:|---:|---:|---:|---:|
| test_gap360 | 6,856,880 | 1,090,998 | 17,338 | 2,605 | 1,863 |
| gap240 | 6,856,880 | 1,093,142 | 18,041 | 2,605 | 1,863 |
| gap480 | 6,856,880 | 1,090,243 | 17,005 | 2,605 | 1,860 |
| calibration_gap360 | 14,710,608 | 2,320,017 | 37,103 | 5,829 | 4,188 |

## Individual commands and times

For a new run: `python3 paper/figure_scripts/units_robustness.py`. After an interruption: `python3 paper/figure_scripts/units_robustness.py --resume`.

- B0: `python3 paper/figure_scripts/units_ladder.py --out /private/tmp/units_ladder_B0` (the orchestrator uses a unique temporary folder).
- gap240: `python3 paper/figure_scripts/units_ladder.py --gap 240 --weeks test --out results/benchmarks_units/robustness/gap240 --instant-only --main-schemes`.
- gap480: `python3 paper/figure_scripts/units_ladder.py --gap 480 --weeks test --out results/benchmarks_units/robustness/gap480 --instant-only --main-schemes`.
- calibration_gap360: `python3 paper/figure_scripts/units_ladder.py --gap 360 --weeks calib --out results/benchmarks_units/robustness/calibration_gap360 --instant-only --main-schemes`.

Computation times (s): test_gap360=31.7, gap240=19.7, gap480=19.7, calibration_gap360=43.2.

The specific weeks are given in the `weeks` column of each CSV.
