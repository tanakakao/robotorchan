# Regression active learning

Phase 5 adds uncertainty-based active learning for **continuous regression outputs**.
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
are handled separately, while predictive-information methods such as EPIG are introduced in
later phases.
