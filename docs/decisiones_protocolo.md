# Decisions made before test evaluation

Date of author approval: **12 November 2026** (Europe/Madrid).
These decisions were fixed before running the new episode benchmarks on the four test weeks. The Stage 16–21 inputs and the split into nine calibration weeks and four test weeks remain frozen.

1. **Primary cohort:** green episodes with predicted minimum `m_min < 2`; the variants with no limit and with `m_min < 3` will be reported as sensitivity analyses. The non-green episodes from the original audit are retained as a reconciliation reference.
2. **Scope:** M6 is removed from the main experiment and deferred to future work. Its conflict probability does not directly estimate the probability of a subsequent clearance. The primary comparison is M0–M5.
3. **Decision rule:** the best continuous model among M2–M4 will be preferred if its Brier score improvement over M0 exceeds **one standard error** of the difference, estimated with a day-level bootstrap. Otherwise, M0 may be recommended as an interpretable summary. Model complexity and hyperparameters will be selected using calibration data only.

The primary outcome is defined as at least one eligible order in `(T0, T0 + 300 s]`, where `T0` is the episode start; the association window from any observation within the episode is retained as an exposure analysis. Windows of 120 and 600 s will be evaluated as sensitivity analyses. This definition prevents longer episodes from having a longer primary window by design.

The user approved these three decisions in this session. Thresholds must not be reselected after inspecting the test data.
