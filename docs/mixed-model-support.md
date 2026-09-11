# Mixed-input model support

robotorchan keeps mixed continuous/categorical variants beside their corresponding model families where practical. The public convention is `MixedXxx(..., cat_dims=[...])`; a separate collection of mixed-model files is intentionally avoided.

## Current support

| Model family | Mixed-input support | Public model |
| --- | --- | --- |
| Single-task exact GP | Native mixed kernel | `MixedSingleTaskGP` |
| ICM multi-task GP | Native mixed kernel | `MixedMultiTaskGP` |
| Kronecker multi-task GP | Native mixed kernel | `MixedKroneckerMultiTaskGP` |
| Variational single-task GP | Native mixed kernel | `MixedSingleTaskVariationalGP` |
| Single-task multi-fidelity GP | Native mixed design kernel + BoTorch fidelity kernels | `MixedSingleTaskMultiFidelityGP` |
| Model list | Composes mixed and non-mixed children directly | `ModelListGP` |
| Fully Bayesian SAAS single-task GP | One-hot categorical preprocessing | `SaasFullyBayesianSingleTaskGP` |
| Fully Bayesian SAAS multi-task GP | One-hot categorical preprocessing | `SaasFullyBayesianMultiTaskGP` |

## Fully Bayesian SAAS

BoTorch's fully Bayesian SAAS models evaluate covariance in the JAX/NumPyro sampling path used by NUTS and only construct the GPyTorch covariance after MCMC samples have been produced. For this reason, robotorchan does not pretend that the ordinary GPyTorch mixed-kernel helper can provide a native categorical kernel during NUTS.

The supported practical path is to one-hot encode categorical design variables before constructing the SAAS model. The encoded columns are then used consistently for both NUTS fitting and posterior prediction.

For example:

```python
import torch
from robotorchan.models import SaasFullyBayesianSingleTaskGP

continuous = torch.rand(12, 2, dtype=torch.double)
category = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
one_hot = torch.nn.functional.one_hot(category, num_classes=3).to(torch.double)
train_X = torch.cat([continuous, one_hot], dim=-1)
train_Y = torch.rand(12, 1, dtype=torch.double)

model = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
```

Use the same category ordering and number of one-hot columns for candidate / posterior inputs. For multi-task SAAS, one-hot encode only the design variables and keep the long-format task feature as its own single column.

A future native categorical-kernel SAAS implementation would require a dedicated mixed JAX/NumPyro covariance and matching GPyTorch loading logic. That is intentionally separate from the current wrapper API.
