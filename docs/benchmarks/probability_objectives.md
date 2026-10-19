# Probability objectives

Phase 18 adds two benchmark problems where the probability of Pass
is an **optimization objective**, not a feasibility constraint:

- `pass_probability_objective`: maximize the probability of Pass.
- `yield_probability_tradeoff`: maximize deterministic Yield and
  probability of Pass jointly.

The ground-truth probability is

```text
p(Pass | x) = 0.1 + 0.8 * exp(-8 * ((x0 - 0.7)^2 + (x1 - 0.3)^2))
```

The single-objective optimum is `p=0.9` at `x=(0.7, 0.3)`.
For multiobjective evaluation, Yield is
`1-(x0-0.2)^2-(x1-0.7)^2`, with reference point `(0, 0)`.

`BenchmarkProblem.evaluate_truth(X)` returns analytic probabilities.
`sample_pass_labels(X, generator)` separately draws stochastic 0/1
Bernoulli labels for classifier training, using an explicit PyTorch RNG.

The benchmark runner currently evaluates analytic probability truth,
**not** stochastic Bernoulli labels: the label sampler is an independent
training-data helper. This distinction avoids misrepresenting labels as
calibrated probability observations. A learned classifier should
provide predictive probabilities through its own API before those
probabilities are used as acquisition objectives.

Existing `simple_regret_curve` and `hypervolume_curve` apply to the
truth-valued objective. The phase does not claim to validate
classification model calibration or qEHVI composition.

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
