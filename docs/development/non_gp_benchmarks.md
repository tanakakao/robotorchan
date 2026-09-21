# Non-GP surrogate benchmark

`benchmarks/non_gp_surrogates.py` provides a small deterministic regression benchmark for the
non-GP surrogate families implemented in Phases 4--12.

The benchmark compares Random Forest, Extra Trees, bootstrap Gradient Boosting, and bootstrap
Histogram Gradient Boosting on the same nonlinear synthetic function. It reports predictive RMSE,
mean empirical posterior standard deviation, fitting time, and posterior evaluation time.

The numbers are diagnostics rather than a model ranking. The ensemble posterior semantics differ
between tree ensembles and bootstrap boosting, so posterior standard deviations should not be
interpreted as calibrated uncertainty without a separate calibration study.

Run it from the repository root with:

```bash
python benchmarks/non_gp_surrogates.py
```

The benchmark uses a fixed seed and a modest data size so it can also serve as a regression harness.
The test suite checks reproducibility and finite metrics, but deliberately does not assert that one
surrogate must outperform another.
