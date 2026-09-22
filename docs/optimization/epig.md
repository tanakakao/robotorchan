# Expected Predictive Information Gain

This guide adds regression EPIG as a robotorchan-specific active-learning acquisition.
EPIG targets predictive usefulness: a candidate is valuable when observing it is expected to
reduce uncertainty for predictions drawn from a target input distribution.

For Gaussian regression, mutual information between a candidate observation and a target
observation has a closed form. robotorchan averages this quantity over a finite target set,
optionally using target-distribution weights.

```python
from robotorchan.acquisition import ExpectedPredictiveInformationGain

acqf = ExpectedPredictiveInformationGain(
    model,
    target_X,
    observation_noise=0.01,
    target_weights=weights,
)
scores = acqf(candidate_X)
```

The initial implementation intentionally supports q=1 and single-output Gaussian regression.
It does not pretend to be the classification/ensemble EPIG estimator from the broader active
learning literature. Batch EPIG requires joint conditioning semantics and belongs in a future
extension rather than an independent-score approximation.

The target set is part of the acquisition definition. Using a representative deployment/test
distribution makes EPIG prediction-oriented; using an arbitrary pool changes the question the
acquisition optimizes.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/09_active_learning.md).
