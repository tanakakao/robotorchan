# Regression active learning

This guide adds uncertainty-based active learning for **continuous regression outputs**.
Classification-specific acquisitions remain outside the current scope.

## Pointwise uncertainty

robotorchan provides two small q=1 acquisition functions:

- `PosteriorVariance`: maximize posterior variance;
- `PosteriorStd`: maximize posterior standard deviation.

Both follow BoTorch's `AcquisitionFunction` interface and can therefore be passed to
`optimize_acqf` for single-candidate selection.

```python
from robotorchan.acquisition import PosteriorVariance

acqf = PosteriorVariance(model)
```

For multi-output posteriors, specify `output_index`. robotorchan does not silently sum or
average outputs because that would impose an application-specific utility.

## Batch integrated variance reduction

For batch pure exploration, use BoTorch's native
`qNegIntegratedPosteriorVariance` directly:

```python
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance

acqf = qNegIntegratedPosteriorVariance(model=model, mc_points=target_points)
```

This criterion values a batch by its expected reduction of posterior uncertainty over the
integration points. robotorchan does not wrap or re-export it.

## Scope

These acquisitions are regression active-learning tools. Level-set and boundary learning
are handled separately, while predictive-information methods such as EPIG are documented separately.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/09_active_learning.md).


## Probabilistic non-GP surrogates

Moment-based regression AL is not restricted to Gaussian processes. A non-GP model may use
`PosteriorVariance` and `PosteriorStd` when its BoTorch posterior exposes statistically meaningful
marginal moments. `NGBoostSurrogate` is the reference distributional non-GP path: its Gaussian
predictive distribution supplies these moments directly.

This does not make every Gaussian-posterior acquisition valid. In particular,
`ExpectedPredictiveInformationGain` requires the joint Gaussian covariance contract, while plain
NGBoost only exposes independent predictive marginals. BALD is also outside this contract because
predictive variance alone does not identify epistemic and aleatoric uncertainty separately.
