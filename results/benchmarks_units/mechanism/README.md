# Mechanism of the lift by position

Command: `python3 paper/figure_scripts/units_mechanism.py`.
Measured time: 21.8 s. Seed 20261112; 400 whole-day bootstrap replicates (28 days).

Episodes: green and non-green runs kept separate, gap >360 s. `k_ep` and `k_cls` start at 1; ties follow the order of the Parquet file. `T0` is the first instant of the episode.
The rate is positives/n_obs and has its own CI. `lift_ref0` uses the reference of Step 0 by instantaneous stratum; `lift_refEp` uses the reference of Step 3, one row per episode at its anchor. The CIs are the 2.5th and 97.5th percentiles of the day resampling. The row proportions are computed within each class.
`episode_length.csv` includes distributions by length and duration (last minus first instant, s), as well as the number of positive observations per non-green episode; `share_episodes` uses the total of each class, week and section. The start label uses (T0,T0+300 s]. The time to clearance uses the first eligible clearance in (t,t+300 s]. For episodes, the position indicates how many observations precede the first later clearance; the quotient divides by the length.

| Class | Variant | Bin | lift_ref0 (IC) | lift_refEp (IC) |
|---|---|---|---|---|
| red | k_ep | 1 | 2.38 (2.22–2.54) | 2.03 (1.91–2.16) |
| red | k_ep | 2 | 3.46 (3.10–3.88) | 2.95 (2.65–3.28) |
| red | k_ep | 3 | 3.89 (3.48–4.27) | 3.32 (2.99–3.63) |
| red | k_ep | 4+ | 4.19 (3.74–4.69) | 3.57 (3.23–3.96) |
| red | k_cls | 1 | 2.68 (2.53–2.84) | 2.28 (2.16–2.42) |
| red | k_cls | 2 | 4.05 (3.67–4.41) | 3.45 (3.14–3.75) |
| red | k_cls | 3 | 4.87 (4.25–5.54) | 4.15 (3.62–4.65) |
| red | k_cls | 4+ | 2.64 (0.00–6.16) | 2.25 (0.00–5.30) |
| positive | k_ep | 1 | 2.11 (2.01–2.23) | 1.80 (1.72–1.89) |
| positive | k_ep | 2 | 2.52 (2.40–2.65) | 2.14 (2.04–2.27) |
| positive | k_ep | 3 | 2.74 (2.48–3.01) | 2.34 (2.13–2.55) |
| positive | k_ep | 4+ | 3.15 (2.91–3.44) | 2.69 (2.49–2.90) |
| positive | k_cls | 1 | 2.17 (2.08–2.27) | 1.85 (1.77–1.93) |
| positive | k_cls | 2 | 2.49 (2.33–2.63) | 2.12 (1.99–2.25) |
| positive | k_cls | 3 | 2.86 (2.58–3.15) | 2.44 (2.21–2.69) |
| positive | k_cls | 4+ | 3.43 (3.14–3.75) | 2.92 (2.69–3.17) |
