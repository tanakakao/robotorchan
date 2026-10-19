# Heterogeneous E2E Phase 3 — Regression GP validation

Phase 3 tests the two regression outputs independently against the fixed
Phase 2 synthetic data-generating process.

The integration test constructs the existing robotorchan `SingleTaskGP`
with BoTorch `Normalize(d=3)` and `Standardize(m=1)`, a known observation
variance of `0.02**2`, and fits using the native BoTorch
`fit_gpytorch_mll(model.make_mll())` path.

For both Strength and Conductivity, it verifies retained raw training data,
posterior mean and variance shape, finiteness, nonnegative posterior variance,
and holdout prediction error. A separate test verifies native batched
posterior sampling with `batch=2`, `q=3`, and four posterior draws.

The fixed random generators and CPU float64 configuration make the test
repeatable. The holdout MSE threshold is a broad smoke-test guard, not a
claim of statistically calibrated uncertainty or optimizer superiority.

Run:

```bash
pytest -q tests/benchmarks/test_heterogeneous_regression_e2e.py
```

Classifier training, semantics, acquisition composition, and closed-loop
optimization remain in subsequent phases.
