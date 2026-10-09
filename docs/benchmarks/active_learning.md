# Active learning boundary benchmarks

Phase 20 adds two deterministic binary classification ground truths:

- `circle_boundary`: a closed level-set boundary centered at `(0.5, 0.5)`.
- `wave_boundary`: a nonlinear sinusoidal decision boundary.

Both operate on `[0, 1]^2`. Signed boundary margins are available
from `circle_margin(X)` and `wave_margin(X)`. A nonnegative margin
defines class 1, including the boundary; negative margins define class 0.
Use `active_learning_labels(X, boundary="circle"|"wave")` to
generate canonical 0/1 training labels.

The existing `BenchmarkProblem` contract requires an objective.
These problems use a **constant zero dummy objective** and expose
boundary truth as one outcome constraint. Optimization regret is
therefore **not an active-learning performance measure**.

To compare active-learning strategies, use a fixed, independently
generated reference set and track after each acquisition budget:

- `classification_accuracy(labels, predictive_probabilities)`
- `brier_score(labels, predictive_probabilities)`
- `boundary_mae(true_margin, predicted_margin)` when a signed
  margin predictor is available

Classification accuracy uses a 0.5 probability threshold. The Brier
score evaluates predictive probability calibration and discrimination,
not just class predictions. Boundary MAE measures margin prediction
error, not geometric Hausdorff distance.

A random candidate strategy and seeded runner provide a reproducible
smoke test. This phase does **not** implement model training, predictive
entropy, BALD, boundary-straddle acquisition, or learning-curve
orchestration; those need an active-learning experiment runner.
