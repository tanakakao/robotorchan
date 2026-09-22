# Information-theoretic acquisition functions

This guide uses BoTorch's native information-theoretic Bayesian-optimization acquisitions rather
than wrapping or re-exporting them.

## Max-value Entropy Search

For information gain about the optimum value, use BoTorch directly:

```python
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

acqf = qMaxValueEntropy(model=model, candidate_set=candidate_set)
```

`candidate_set` discretizes the design space used to sample maximum values. The model is
single-output unless an appropriate posterior transform is supported by the chosen native
acquisition.

## GIBBON

For a cheaper lower-bound approximation with practical batch support:

```python
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

acqf = qLowerBoundMaxValueEntropy(model=model, candidate_set=candidate_set)
```

robotorchan integration-tests these native classes but does not create aliases or wrappers.

## Randomized Straddle

Level-set estimation has a separate information-driven extension,
`RandomizedStraddle`. Following Inatsu et al. (2024), it replaces the manually tuned
Straddle confidence parameter with a draw from a chi-squared distribution with two degrees
of freedom. The sampled value is used as `sqrt(beta_t)` in the Straddle score.

```python
from robotorchan.acquisition import RandomizedStraddle

generator = torch.Generator().manual_seed(0)
acqf = RandomizedStraddle(model, target=0.0, generator=generator)
```

Supplying a generator makes experiments reproducible. One random confidence coefficient is held fixed throughout a candidate-selection round so
acquisition optimization sees a stable objective. Call `resample()` to start a new round with a
new coefficient.

robotorchan does not add a local MES/PES/JES implementation when BoTorch already provides the
required information-theoretic BO machinery.


For entropy, mutual information, MES, and GIBBON theory, see
[Information-theoretic Acquisition](../theory/acquisition/05_information_theoretic.md).
`RandomizedStraddle` itself is covered in
[Level-set Acquisition](../theory/acquisition/10_level_set.md).
