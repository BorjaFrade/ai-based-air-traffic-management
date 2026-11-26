# Ablation of M4 (T11)

Frozen cohort: 67,367 episodes; 21,281 in test. Outcome: eligible clearance in `(T0,T0+300 s]`. One episode, one vote. Nine calibration weeks for the selection of C by leave-one-week-out; four test weeks for scoring.

The full M3 and M4 reproduce exactly the Brier and ΔBrier values saved in `results/benchmarks/metrics.csv` (asserts in `ablation.py`). The CIs are percentiles of 400 day-level bootstrap replicates with seed 20261112, and use the same day indices in all paired differences.

| Model | Brier | Log score | Slope | Δ vs M0 [95 % CI] | Δ vs M3 [95 % CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 | 0.169617 | 0.520616 | 0.8986 | -0.001619 [-0.002321, -0.000925] | +0.000000 [+0.000000, +0.000000] |
| M3+density | 0.169653 | 0.520721 | 0.8917 | -0.001583 [-0.002349, -0.000872] | +0.000036 [-0.000081, +0.000130] |
| M3+prior | 0.169608 | 0.520724 | 0.8923 | -0.001627 [-0.002334, -0.000913] | -0.000008 [-0.000061, +0.000054] |
| M3+duration | 0.169105 | 0.518909 | 0.9437 | -0.002130 [-0.002825, -0.001473] | -0.000512 [-0.000666, -0.000343] |
| M3+density+prior | 0.169645 | 0.520818 | 0.8861 | -0.001590 [-0.002363, -0.000876] | +0.000029 [-0.000089, +0.000129] |
| M4 | 0.169040 | 0.518949 | 0.9337 | -0.002196 [-0.002977, -0.001489] | -0.000577 [-0.000764, -0.000377] |

Increment of M4 over M3: **-0.000577** in Brier score.
Density + prior clearance without duration: **+0.000029**.
Duration added after those two variables: **-0.000605**.
These are conditional differences between refitted models, not a causal attribution nor an additive decomposition of individual effects.

## Differences by test week

| Model | Week | Δ vs M0 | Δ vs M3 |
| --- | --- | ---: | ---: |
| M3 | 20170107_20170113 | -0.000324 | +0.000000 |
| M3 | 20170401_20170407 | -0.002937 | +0.000000 |
| M3 | 20170624_20170630 | -0.000196 | +0.000000 |
| M3 | 20170916_20170922 | -0.002653 | +0.000000 |
| M3+density | 20170107_20170113 | -0.000438 | -0.000114 |
| M3+density | 20170401_20170407 | -0.003062 | -0.000124 |
| M3+density | 20170624_20170630 | +0.000085 | +0.000281 |
| M3+density | 20170916_20170922 | -0.002647 | +0.000006 |
| M3+prior | 20170107_20170113 | -0.000339 | -0.000015 |
| M3+prior | 20170401_20170407 | -0.003014 | -0.000076 |
| M3+prior | 20170624_20170630 | -0.000232 | -0.000036 |
| M3+prior | 20170916_20170922 | -0.002578 | +0.000075 |
| M3+duration | 20170107_20170113 | -0.001224 | -0.000900 |
| M3+duration | 20170401_20170407 | -0.003277 | -0.000340 |
| M3+duration | 20170624_20170630 | -0.000763 | -0.000567 |
| M3+duration | 20170916_20170922 | -0.003037 | -0.000384 |
| M3+density+prior | 20170107_20170113 | -0.000449 | -0.000125 |
| M3+density+prior | 20170401_20170407 | -0.003130 | -0.000192 |
| M3+density+prior | 20170624_20170630 | +0.000053 | +0.000250 |
| M3+density+prior | 20170916_20170922 | -0.002580 | +0.000073 |
| M4 | 20170107_20170113 | -0.001358 | -0.001034 |
| M4 | 20170401_20170407 | -0.003465 | -0.000527 |
| M4 | 20170624_20170630 | -0.000664 | -0.000468 |
| M4 | 20170916_20170922 | -0.003126 | -0.000473 |

## Limit of interpretation

Duration and the geometric maximum are known only when the episode ends. This ablation compares retrospective associations, not prediction available in real time at T0.
