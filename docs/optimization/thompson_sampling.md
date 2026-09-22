# Thompson sampling and posterior sampling

This guide separates **finite candidate-set selection** from continuous acquisition optimization.

For a finite candidate pool, robotorchan provides `select_thompson_candidates`. It delegates
posterior sampling and maximization to BoTorch's `MaxPosteriorSampling` and only adds a small,
explicit contract for candidate shape and sampling count.

```python
from robotorchan.acquisition import select_thompson_candidates

selected = select_thompson_candidates(model, choices, num_samples=4)
```

This is useful when the feasible domain is already enumerated, including many discrete or
mixed-variable workflows. With `replacement=False` (the default), a batch contains distinct
rows from the supplied candidate set.

## Why this is not an acquisition-function wrapper

Thompson sampling is naturally expressed as posterior sampling followed by selection of a
sample maximizer. BoTorch already supplies the sampling primitive, so robotorchan does not
introduce a synthetic Thompson acquisition class or duplicate posterior sampling logic.

## Continuous domains

Continuous Thompson sampling additionally requires a strategy for maximizing a sampled
function over bounds. That is an optimization concern and is deliberately not hidden inside
the finite-pool helper. High-dimensional strategies such as trust-region or embedding-based
search may require different sampled-function optimization paths.

## Compatibility

The helper requires a BoTorch-compatible model posterior. Compatibility therefore follows
the model/posterior contract rather than a GP-specific type check. Model families with
special posterior semantics should be validated by focused integration tests before being
listed as supported.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/12_selection_guide.md).
