# Phase 16: Regression surrogate variation

Compare the compatibility of exact `SingleTaskGP` and approximate `SingleTaskVariationalGP` with the same semantic qEI acquisition and candidate optimization pipeline. Both models train on the same noisy Strength observations from the heterogeneous synthetic benchmark.

The exact model fits its marginal likelihood; the variational model trains with Adam and its `VariationalELBO` using eight inducing points. Tests check raw training data retention, finite native BoTorch posteriors, batched qEI evaluations, and bounded candidate generation for q=1 and q=2.

**Limitations:** This is a model-family integration test, not an accuracy or runtime comparison. Variational fitting uses a short fixed iteration budget and may be underconverged. Classification, multitask, mixed categorical inputs, and high-dimensional surrogate variants require separate compatibility tests.

Run `pytest -q tests/benchmarks/test_heterogeneous_regression_variations_e2e.py`.
