# Phase 11: ProbabilityObjective end-to-end validation

Fit the binary Pass classifier using Bernoulli variational inference. Interpret either class probability (class 0 or class 1) as a maximization objective via ProbabilityObjective and the existing semantic acquisition bridge.

The test evaluates predictive probabilities on a reproducible finite candidate pool, checks the bridge against the classifier's predict_proba contract, and selects the pool candidate with the highest predicted objective value. If supported by the model capability registry, it also validates epistemic probability samples. A separate negative test rejects a regression output as a classification probability objective.

**Scope:** This is probability-objective scoring and finite-pool candidate selection, not native qEI/qNEI/qEHVI. A classification probability is not treated as a Gaussian latent posterior, and no unsupported mixed-family joint MC acquisition is constructed. Selection does not imply true improvement or probability calibration.

Run `pytest -q tests/benchmarks/test_heterogeneous_probability_objective_e2e.py`.
