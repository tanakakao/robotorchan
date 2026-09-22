# Multi-objective acquisition functions

This guide keeps standard multi-objective Bayesian optimization on BoTorch's native acquisition
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

## Constraints and custom objectives

The registry's `supports_constraints` flag describes a supported BoTorch composition path. It
does not mean robotorchan creates constraint callables, feasibility conventions, objectives, or
reference points automatically. These remain problem-specific BoTorch inputs.

qLogEHVI, qLogNEHVI, and qLogNParEGO are registered as multi-objective paths. qLogNParEGO was
already runtime-tested in robotorchan and is now represented in the capability registry as well,
so documentation, executable coverage, and recommendation metadata agree.

## Hypervolume Knowledge Gradient

Hypervolume Knowledge Gradient remains a BoTorch-native future integration item. The current
repository does not contain executable HVKG integration coverage, so robotorchan must not present
it as runtime-validated merely because BoTorch exposes the algorithm. Adding it should include
model compatibility, fantasy semantics, and candidate-optimization tests before registry support
is declared.

## Scope

The validated standard multi-objective surface covers qLogEHVI, qLogNEHVI, and qLogNParEGO.
Constraint composition and custom objectives remain explicit BoTorch inputs. Multi-fidelity
hypervolume lookahead and HVKG are not part of the current validated contract.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/07_multiobjective.md).
