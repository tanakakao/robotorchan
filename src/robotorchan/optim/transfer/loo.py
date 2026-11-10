"""Leave-one-out target predictions for RGPE ranking weights."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


def target_loo_predictions(
    train_x: Tensor, train_y: Tensor, *, cat_dims: list[int] | None = None
) -> Tensor:
    """Fit n independent scalar SingleTaskGP models, omitting each point once.

    This deliberately uses refitting rather than in-sample predictions.
    It is intended for the small target datasets used in transfer BO.
    Input targets may have shape (n,) or (n, 1); output has shape (n,).
    """
    if train_x.ndim != 2 or train_x.shape[0] < 3:
        raise ValueError("train_x must have shape (n, d) with n >= 3")
    n = train_x.shape[0]
    if train_y.shape not in ((n,), (n, 1)):
        raise ValueError("train_y must have shape (n,) or (n, 1)")
    if not bool(torch.isfinite(train_x).all() and torch.isfinite(train_y).all()):
        raise ValueError("training observations must be finite")
    targets = train_y.reshape(n, 1)
    predictions = []
    for index in range(n):
        mask = torch.arange(n, device=train_x.device) != index
        model = (
            MixedSingleTaskGP(train_x[mask], targets[mask], cat_dims=cat_dims)
            if cat_dims is not None
            else SingleTaskGP(train_x[mask], targets[mask])
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        with torch.no_grad():
            mean = model.posterior(train_x[index : index + 1]).mean.reshape(())
        predictions.append(mean)
    return torch.stack(predictions)
