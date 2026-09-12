# Mixed-input model support

robotorchan keeps mixed continuous/categorical variants beside their corresponding model families where practical. The public convention is `MixedXxx(..., cat_dims=[...])`; a separate collection of mixed-model files is intentionally avoided.

## Implementation policy

The preferred implementation is a real categorical covariance based on `CategoricalKernel` whenever the model architecture allows that covariance to participate in the model's actual fitting and prediction path without changing the upstream mathematical semantics.

Internal one-hot encoding is a compatibility fallback for model families whose covariance is constructed or evaluated in a way that makes native categorical-kernel injection unsafe or unavailable. Users still interact with the original mixed feature space; training, posterior evaluation, conditioning, and other supported model operations must apply the encoding internally and consistently.

**One-hot support should not be treated as the desired permanent endpoint when native categorical covariance is feasible.** Except for model architectures where a true categorical kernel is structurally impossible or would require changing the underlying model definition, Mixed wrappers should be eligible to migrate to a native categorical kernel in the future while preserving their public `MixedXxx(..., cat_dims=[...])` API.

## Current support

| Model family | Mixed-input support | Public model |
| --- | --- | --- |
| Single-task exact GP | Native mixed kernel | `MixedSingleTaskGP` |
| ICM multi-task GP | Native mixed kernel | `MixedMultiTaskGP` |
| Kronecker multi-task GP | Native mixed kernel | `MixedKroneckerMultiTaskGP` |
| Variational single-task GP | Native mixed kernel | `MixedSingleTaskVariationalGP` |
| Single-task multi-fidelity GP | Native mixed design kernel + BoTorch fidelity kernels | `MixedSingleTaskMultiFidelityGP` |
| Robust relevance-pursuit single-task GP | Native mixed kernel + robust outlier likelihood | `MixedRobustRelevancePursuitSingleTaskGP` |
| Model list | Composes mixed and non-mixed children directly | `ModelListGP` |
| Fully Bayesian SAAS single-task GP | Internal one-hot fallback | `MixedSaasFullyBayesianSingleTaskGP` |
| Fully Bayesian SAAS multi-task GP | Internal one-hot fallback | `MixedSaasFullyBayesianMultiTaskGP` |
| Additive MAP-SAAS single-task GP | Internal one-hot fallback | `MixedAdditiveMapSaasSingleTaskGP` |
| Ensemble MAP-SAAS single-task GP | Internal one-hot fallback | `MixedEnsembleMapSaasSingleTaskGP` |

## Internal one-hot behavior

Mixed wrappers that use the fallback path still expose raw categorical columns to callers:

```python
from robotorchan.models import MixedSaasFullyBayesianSingleTaskGP

model = MixedSaasFullyBayesianSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[1, 3],
)

posterior = model.posterior(raw_mixed_X)
```

The model owns the category vocabulary and encoded representation. The original caller tensor remains available through `raw_train_X`. Prediction inputs are encoded using the same stored vocabulary, and unseen categories are rejected explicitly unless a future model documents another policy.

## Fully Bayesian SAAS

BoTorch's fully Bayesian SAAS models evaluate covariance in the JAX/NumPyro sampling path used by NUTS and only construct the GPyTorch covariance after MCMC samples have been produced. Injecting a GPyTorch categorical kernel only after sampling would therefore make fitting and prediction inconsistent. The current Mixed SAAS wrappers use internal one-hot encoding so the same numeric representation is seen throughout NUTS and posterior evaluation.

A future true categorical-kernel implementation would require matching categorical covariance semantics in both the JAX/NumPyro sampling path and the GPyTorch model loaded from MCMC samples. Until that exists, one-hot is the structurally safe fallback.

## MAP-SAAS

BoTorch MAP-SAAS builds specialized SAAS covariance modules internally rather than exposing the normal mixed covariance injection surface used by robotorchan's native Mixed exact-GP wrappers. Phase 9 therefore uses a model-owned one-hot `InputTransform` for `MixedAdditiveMapSaasSingleTaskGP` and `MixedEnsembleMapSaasSingleTaskGP`.

Using an `InputTransform` rather than manually encoding only constructor and posterior tensors keeps the transformation attached to the BoTorch model and allows supported conditioning / fantasy workflows to reuse the same representation. If BoTorch later exposes a mathematically faithful way to combine MAP-SAAS sparsity priors with native categorical covariance, these wrappers should migrate to that implementation without changing the public `cat_dims` interface.

## Robust relevance pursuit

BoTorch's robust relevance-pursuit model exposes its data `covar_module` directly. Its specialized behavior is implemented by wrapping the observation likelihood with sparse outlier noise and by dispatching relevance pursuit during `fit_gpytorch_mll`; the data covariance itself remains replaceable.

`MixedRobustRelevancePursuitSingleTaskGP` therefore uses the normal robotorchan native mixed covariance rather than one-hot encoding. The continuous, categorical, and interaction terms participate directly in fitting and prediction, while the upstream robust likelihood and relevance-pursuit fitting path remain unchanged. The model's `to_standard_model()` path also retains the same mixed covariance module so the specialized fitting dispatch does not silently fall back to a continuous-only kernel.
