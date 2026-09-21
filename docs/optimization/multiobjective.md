# Multi-objective acquisition functions

Phase 9 keeps standard multi-objective Bayesian optimization on BoTorch's native acquisition
APIs. robotorchan models are tested against these APIs rather than hidden behind local aliases.

## Hypervolume improvement

For noiseless or effectively deterministic observations, prefer
`qLogExpectedHypervolumeImprovement` (qLogEHVI). For noisy observations, prefer
`qLogNoisyExpectedHypervolumeImprovement` (qLogNEHVI).

```python
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)

acqf = qLogExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    partitioning=partitioning,
)
```

The reference point must be dominated by the objective values whose hypervolume should count.
robotorchan does not guess this value because its meaning is problem-specific.

## qLogNParEGO

For scalarization-based multi-objective optimization, BoTorch provides `qLogNParEGO`.
It is useful when hypervolume-based optimization becomes inconvenient or a decomposition-based
search is preferred.

```python
from botorch.acquisition.multi_objective.parego import qLogNParEGO

acqf = qLogNParEGO(
    model=model,
    X_baseline=train_X,
    scalarization_weights=weights,
)
```

Weights define one scalarized subproblem. Repeated BO iterations normally vary them to explore
the Pareto frontier.

## Multi-objective Knowledge Gradient

BoTorch also provides hypervolume Knowledge Gradient for non-myopic multi-objective BO.
It remains a native BoTorch path rather than a robotorchan wrapper. Because its fantasy-tree
optimization is materially heavier than EHVI/NEHVI, use it when the value of lookahead
justifies the additional cost.

## Scope

Phase 9 covers the standard multi-objective acquisition interface. Constraint composition,
custom objectives, and multi-fidelity hypervolume lookahead are orthogonal concerns and remain
with their dedicated integration paths.
