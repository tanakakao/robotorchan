# Heterogeneous benchmark problems (Phase 10)

The `strength_conductivity_pass` benchmark has two continuous regression
objectives and one binary classification target:

- **Strength**: maximize a smooth continuous response.
- **Conductivity**: maximize a competing smooth continuous response.
- **Pass**: a binary feasibility label, 1 for Pass and 0 for Fail.

Both input variables lie in [0, 1]. The deterministic ground-truth
feasibility rule is `x1 - 0.25 - 0.5*x0 >= 0`.

```python
from robotorchan.benchmarks import (
    strength_conductivity_pass,
    strength_conductivity_pass_labels,
)

problem = strength_conductivity_pass()
X = problem.bounds.mean(dim=0).unsqueeze(0)
regression_targets = problem.evaluate_truth(X)  # (q, 2)
binary_targets = strength_conductivity_pass_labels(X)  # (q, 1), values 0 or 1
ground_truth_margin = problem.evaluate_constraints(X)  # (q, 1)
```

## Separation of model targets and feasibility truth

`BenchmarkProblem` stores continuous objectives and signed feasibility
margins, consistent with the existing `g(X) >= 0` outcome-constraint
contract. The binary labels are obtained through an **explicit separate
function**. The signed margin must not be passed off as a classification
target. The labels are deterministic, and the boundary is Pass.

This phase provides the test problem and labels; it does **not** claim
integration with classifier posterior probabilities, semantic adapters,
constrained qEHVI/qNEHVI, or model-training pipelines. Those require
dedicated end-to-end integration tests in a later phase.
