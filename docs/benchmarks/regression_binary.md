# Single regression objective with binary Pass constraint

The `strength_pass` benchmark combines one continuous objective
(maximize Strength) with one binary Pass/Fail outcome constraint.

- Strength: `1 - (x0-0.8)^2 - (x1-0.3)^2`
- Pass margin: `g(X) = x1 - 0.25 - 0.5*x0`
- Binary label: `strength_pass_labels(X) = (g(X) >= 0)`
- Known constrained optimum: `X=(0.66, 0.58)`, Strength `0.902`

The benchmark stores the **deterministic signed margin** as the truth
constraint for feasibility metrics. A classification model should
instead train on the canonical 0/1 labels returned by
`strength_pass_labels`; these must not be confused with latent GP
posterior samples or predictive Pass probabilities.

`register_regression_binary_problems(registry)` registers the problem.
`run_benchmark` supports seeded evaluation, and existing
`simple_regret_curve` and `cumulative_feasibility_rate` support
truth-based evaluation.

This phase adds a reusable benchmark fixture and evaluation tests,
not a trained classifier or a constrained qEI/qNEI acquisition
composition. Model training and probabilistic feasibility should be
benchmarked separately against this known ground truth.
