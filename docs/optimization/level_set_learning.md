# Level-set and boundary learning

This guide separates **continuous-response level-set estimation** from generic regression active
learning and from classification.

The task is to learn the input-space boundary where a continuous latent response reaches a
specified level `target`.

## Straddle

`Straddle` uses the score

`beta * posterior_std - abs(posterior_mean - target)`.

It therefore prefers points that are both uncertain and plausibly close to the requested
level set.

```python
from robotorchan.acquisition import Straddle

acqf = Straddle(model, target=0.0, beta=1.96)
```

`target` is the response level and `beta` controls the uncertainty contribution. Both names
are part of the level-set API rather than classification terminology.

## BoundaryVariance

`BoundaryVariance` emphasizes posterior variance near the target boundary. Proximity is
measured relative to posterior standard deviation, so points confidently far from the target
receive little value.

```python
from robotorchan.acquisition import BoundaryVariance

acqf = BoundaryVariance(model, target=critical_value)
```

`Straddle`, `BoundaryVariance`, and `RandomizedStraddle` currently support q=1. For multi-output posteriors, `output_index` must be
specified explicitly. Structured tensor outputs must be scalarized before use.

## Scope

These methods operate on continuous regression posteriors. They do not introduce a classifier
or classification-specific uncertainty criterion. `RandomizedStraddle` is a public randomized level-set acquisition; `BoundaryVariance` remains
explicitly identified as a robotorchan-specific heuristic.


For the underlying theory and provenance distinctions, see
[Level-set Acquisition](../theory/acquisition/10_level_set.md).
