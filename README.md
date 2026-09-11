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

The first model wrapper is `robotorchan.models.SingleTaskGP`. It subclasses BoTorch's `SingleTaskGP`, while adding:

- `raw_train_X`
- `raw_train_Y`
- `raw_train_Yvar`
- `make_mll()`

The intention is to use the same names across compatible model wrappers so downstream utilities do not need model-specific branching for common operations.

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = train_X.sin().sum(dim=-1, keepdim=True)

model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
mll = model.make_mll()
fit_gpytorch_mll(mll)

assert torch.equal(model.raw_train_X, train_X)
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
