# Nonstationary Gaussian-process models

robotorchan implements exact GP models with input-dependent latent
process smoothness. These models use a Gibbs covariance kernel and
must not be confused with heteroskedastic observation-noise models.

## Covariance and local lengthscales

The `GibbsKernel` uses an affine, positive lengthscale field:

```text
ell(x) = softplus(a + B x) + lengthscale_floor
```

The resulting Gibbs covariance allows smoothness to vary across the
input space. The kernel learns its intercept and slope parameters
jointly with the GP model. `lengthscale_floor` must be positive.

## Implemented model families

| Model | Input structure | Local-lengthscale diagnostic |
| --- | --- | --- |
| `NonstationarySingleTaskGP` | Continuous single-task | `local_lengthscale(X)` |
| `MixedNonstationarySingleTaskGP` | Mixed single-task, `cat_dims` | Tuple of additive and interaction continuous-branch lengthscales |
| `NonstationaryMultiTaskGP` | Long-format task feature | Data-feature lengthscales, excluding task identity |
| `MixedNonstationaryMultiTaskGP` | Long-format task + mixed data | Tuple of continuous-branch lengthscales |
| `NonstationaryKroneckerMultiTaskGP` | Block-design multi-output | `local_lengthscale(X)` on data features |

Mixed models apply the Gibbs kernel to continuous data features and
combine it with categorical covariance. A mixed model requires at
least one continuous data dimension. `task_feature` and `cat_dims`
identify distinct structural features and must not overlap.

## Example: single-task nonstationary GP

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import NonstationarySingleTaskGP

train_X = torch.rand(24, 2, dtype=torch.double)
train_Y = torch.sin(6 * train_X[:, :1])

model = NonstationarySingleTaskGP(
    train_X,
    train_Y,
    lengthscale_floor=1e-3,
)
fit_gpytorch_mll(model.make_mll())

test_X = torch.rand(5, 2, dtype=torch.double)
posterior = model.posterior(test_X)
local_scales = model.local_lengthscale(test_X)
```

The exact-GP variants support the usual `make_mll()` fitting path,
BoTorch posterior interface, and raw training-data retention.
The local-lengthscale diagnostic reports learned kernel structure;
it is not a direct estimate of observation noise or causal regime
boundaries.

## Modeling boundaries

- **Heteroskedastic GP:** input-dependent observation variance.
- **Nonstationary GP:** input-dependent latent covariance/smoothness.
- **Uncertain-input GP:** uncertain training input coordinates.
- **Risk-aware acquisition:** decision-time objective aggregation.

The implemented family uses affine lengthscale fields. Neural or
GP-driven lengthscale fields and change-point kernels are not implied
by these APIs. Validate performance against stationary baselines,
particularly when data are sparse.

See [Robust and noise models](robust_noise.md) and
[Model selection](../getting_started/model_selection.md).
