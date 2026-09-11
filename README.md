# robotorchan

`robotorchan` is an extension library for [BoTorch](https://botorch.org/) focused on Bayesian optimization, active learning, experimental design, and a consistent model-facing API.

## Design goals

- **BoTorch-native**: accept and return standard BoTorch / PyTorch objects whenever possible.
- **Thin wrappers where useful**: existing BoTorch models may be wrapped when that adds consistent robotorchan behavior such as raw-data retention or `make_mll()`.
- **Research-friendly**: make experimental acquisition functions and optimization utilities easy to test.
- **Composable**: keep models, acquisition functions, objectives, transforms, and optimizers separable.
- **Tested**: numerical behavior and tensor shapes are covered by automated tests.
- **Clean-room implementation**: robotorchan is implemented from scratch from public algorithms and public APIs. It does not copy source code from prior private/internal projects.

## Model conventions

Current wrappers are:

- `robotorchan.models.SingleTaskGP`
- `robotorchan.models.MixedSingleTaskGP`
- `robotorchan.models.SingleTaskMultiFidelityGP`
- `robotorchan.models.MultiTaskGP`
- `robotorchan.models.KroneckerMultiTaskGP`
- `robotorchan.models.ModelListGP`

Supervised single-model wrappers add the common robotorchan model surface where applicable:

- `raw_train_X`
- `raw_train_Y`
- `raw_train_Yvar`
- `raw_data`
- `supports_mll`
- `make_mll()`

The wrappers preserve the upstream constructor surface and delegate predictive behavior, kernels, transforms, conditioning, and model-specific semantics to BoTorch. For `KroneckerMultiTaskGP`, `raw_train_Yvar` is `None` because the upstream constructor does not expose a `train_Yvar` argument.

`ModelListGP` is a container of independent child models, so it does not invent singular container-level training tensors. Instead it exposes:

- `raw_train_Xs`
- `raw_train_Ys`
- `raw_train_Yvars`
- `supports_mll = True`
- `make_mll()` returning `SumMarginalLogLikelihood`

Grouped raw values are available when the corresponding child model implements the robotorchan raw-data contract. Native BoTorch children remain fully supported and yield `None` for unavailable raw values.

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import ModelListGP, SingleTaskGP

train_X1 = torch.rand(20, 3, dtype=torch.double)
train_Y1 = train_X1.sin().sum(dim=-1, keepdim=True)
train_X2 = torch.rand(16, 3, dtype=torch.double)
train_Y2 = train_X2.cos().sum(dim=-1, keepdim=True)

model = ModelListGP(
    SingleTaskGP(train_X=train_X1, train_Y=train_Y1),
    SingleTaskGP(train_X=train_X2, train_Y=train_Y2),
)
mll = model.make_mll()
fit_gpytorch_mll(mll)

assert torch.equal(model.raw_train_Xs[0], train_X1)
```

## Initial scope

Planned extension areas include:

- thin wrappers for selected existing BoTorch models when they benefit from the common robotorchan API;
- custom acquisition functions for Bayesian optimization and active learning;
- multi-objective, constrained, robust, and risk-aware extensions;
- classification / boundary-search acquisition functions;
- ordinal and preference-oriented optimization;
- lookahead and information-theoretic methods;
- utilities for mixed, discrete, and structured search spaces;
- new surrogate-model extensions that remain compatible with BoTorch APIs.

## Requirements

- Python >= 3.11
- BoTorch >= 0.18.1

## Status

Early development. The public API is not yet stable.
