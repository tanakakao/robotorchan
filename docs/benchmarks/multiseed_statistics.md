# Multi-seed statistical evaluation

Phase 24 extends the existing normal-approximation curve summaries with
**seed-level percentile bootstrap confidence intervals**.

`bootstrap_curves({seed: metric_curve, ...})` returns pointwise mean and
bootstrap confidence bounds. The bootstrap resamples **whole independent
seed trajectories**, rather than individual evaluation steps, preserving
within-trajectory dependence. Identical `seed` and `n_resamples` values
reproduce the same interval.

`compare_paired_bootstrap(first, second)` computes first-minus-second
curves for matching seed identifiers and resamples the paired differences.
`probability_positive` is the empirical fraction of bootstrap mean
differences above zero. It is **not** a Bayesian posterior probability
or a formal p-value.

Use the Phase 23 `FairComparisonProtocol` to validate identical
experimental settings and initial observations before comparing curves.
Report the number of independent seeds, interval confidence, bootstrap
resamples, metric orientation, and evaluation budget.

Important limitations:

- Bounds are **pointwise**, not simultaneous bands.
- Percentile bootstrap can be unreliable with very few independent seeds.
- A single-seed interval collapses to a point and provides no uncertainty estimate.
- Bootstrap intervals alone do not correct for multiple comparisons.
- Missing or non-finite metric curves are rejected by the shared curve validator.
