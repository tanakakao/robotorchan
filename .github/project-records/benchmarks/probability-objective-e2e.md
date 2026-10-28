# Historical probability-objective verification

## Heterogeneous ProbabilityObjective E2E validation (historical Phase 11)

The binary Pass GP classifier is fitted with Bernoulli variational
inference. `ProbabilityObjective` interprets either predicted class-0
or class-1 probability as a **maximization objective**, rather than
using Pass as a feasibility constraint. The semantic acquisition bridge
scores a reproducible finite candidate pool against the classifier's
`predict_proba` contract, then selects the candidate with the highest
predicted objective value. Where the model capability registry permits,
the test also checks epistemic probability samples. A negative test
rejects regression outputs used as classification probability objectives.

This E2E test uses **learned** probabilities, whereas the analytic
benchmark problems above use `evaluate_truth(X)` for known probability
truth and `sample_pass_labels(X, generator)` for stochastic labels.
These are distinct data and evaluation contracts.

**Scope:** Finite-pool probability-objective scoring is **not** native
qEI/qNEI/qEHVI. Classification probabilities are not Gaussian latent
posteriors, and the test does not construct unsupported mixed-family
joint Monte Carlo acquisitions. Candidate selection does not establish
true improvement or classifier probability calibration.

Run `pytest -q tests/benchmarks/test_heterogeneous_probability_objective_e2e.py`.
