"""Native MLL fit contracts for all nonstationary GP variants."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedNonstationaryMultiTaskGP,
    MixedNonstationarySingleTaskGP,
    NonstationaryKroneckerMultiTaskGP,
    NonstationaryMultiTaskGP,
    NonstationarySingleTaskGP,
)


@pytest.mark.parametrize(
    "model_class",
    [
        NonstationarySingleTaskGP,
        MixedNonstationarySingleTaskGP,
        NonstationaryMultiTaskGP,
        MixedNonstationaryMultiTaskGP,
        NonstationaryKroneckerMultiTaskGP,
    ],
)
def test_nonstationary_fit_uses_native_mll(model_class):
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x)
    mixed = model_class in (MixedNonstationarySingleTaskGP, MixedNonstationaryMultiTaskGP)
    multitask = model_class in (NonstationaryMultiTaskGP, MixedNonstationaryMultiTaskGP)
    kronecker = model_class is NonstationaryKroneckerMultiTaskGP

    X = x
    kwargs = {}
    if mixed:
        category = (torch.arange(8) % 2).to(x).unsqueeze(-1)
        X = torch.cat([X, category], dim=-1)
        y = y + 0.1 * category
        kwargs["cat_dims"] = [1]
    if multitask:
        task0 = torch.cat([X, torch.zeros_like(x)], dim=-1)
        task1 = torch.cat([X, torch.ones_like(x)], dim=-1)
        X = torch.cat([task0, task1], dim=0)
        y = torch.cat([y, y + 0.2], dim=0)
        kwargs["task_feature"] = X.shape[-1] - 1
    if kronecker:
        y = torch.cat([y, y + 0.2], dim=-1)

    model = model_class(X, y, **kwargs)
    assert model.supports_fit is True
    assert model.supports_mll is True
    optimizer_kwargs = {"options": {"maxiter": 2}}
    with patch("robotorchan.models.base.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs=optimizer_kwargs) is None
    fit.assert_called_once()
    args, call_kwargs = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert call_kwargs["optimizer_kwargs"] == optimizer_kwargs
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, y)
