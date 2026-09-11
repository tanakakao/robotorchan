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
- `robotorchan.models.SingleTaskVariationalGP`
- `robotorchan.models.PairwiseGP`

Supervised single-model wrappers add the common robotorchan model surface where applicable:

- `raw_train_X`
- `raw_train_Y`
- `raw_train_Yvar`
- `raw_data`
- `supports_mll`
- `make_mll()`

The wrappers preserve the upstream constructor surface and delegate predictive behavior, kernels, transforms, conditioning, and model-specific semantics to BoTorch. For models whose upstream constructor does not expose `train_Yvar`, `raw_train_Yvar` is `None`.

`ModelListGP` is a container of independent child models, so it does not invent singular container-level training tensors. Instead it exposes:

- `raw_train_Xs`
- `raw_train_Ys`
- `raw_train_Yvars`
- `supports_mll = True`
- `make_mll()` returning `SumMarginalLogLikelihood`

Grouped raw values are available when the corresponding child model implements the robotorchan raw-data contract. Native BoTorch children remain fully supported and yield `None` for unavailable raw values.

`SingleTaskVariationalGP` preserves BoTorch's variational model semantics and uses `VariationalELBO` rather than an exact marginal log likelihood. Its `make_mll(num_data=None)` method uses the number of rows in `raw_train_X` by default. For minibatch training, pass the total training-set size explicitly as `num_data`.

```python
import torch
from robotorchan.models import SingleTaskVariationalGP

train_X = torch.rand(200, 3, dtype=torch.double)
train_Y = train_X.sin().sum(dim=-1, keepdim=True)

model = SingleTaskVariationalGP(
    train_X=train_X,
    train_Y=train_Y,
    inducing_points=40,
)
mll = model.make_mll()

assert torch.equal(model.raw_train_X, train_X)
assert mll.num_data == 200
```

`PairwiseGP` preserves preference-learning semantics rather than pretending comparisons are ordinary supervised targets. It exposes:

- `raw_datapoints`
- `raw_comparisons`
- `raw_data`
- `supports_mll = True`
- `make_mll()` returning `PairwiseLaplaceMarginalLogLikelihood`

Raw preference tensors are captured before BoTorch applies input transforms or duplicate consolidation. Both tensors may be `None`, matching BoTorch's prior-only construction mode.

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import PairwiseGP

items = torch.rand(12, 3, dtype=torch.double)
comparisons = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)

model = PairwiseGP(datapoints=items, comparisons=comparisons)
mll = model.make_mll()
fit_gpytorch_mll(mll)

assert torch.equal(model.raw_datapoints, items)
assert torch.equal(model.raw_comparisons, comparisons)
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
