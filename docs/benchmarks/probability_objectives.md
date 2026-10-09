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
