# Level-set and boundary learning

Phase 6 separates **continuous-response level-set estimation** from generic regression active
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

Both acquisitions currently support q=1. For multi-output posteriors, `output_index` must be
specified explicitly. Structured tensor outputs must be scalarized before use.

## Scope

These methods operate on continuous regression posteriors. They do not introduce a classifier
or classification-specific uncertainty criterion. Randomized Straddle remains an advanced
extension rather than being folded into the deterministic core prematurely.
