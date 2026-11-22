"""Focused fit delegation checks for structured-output GP wrappers."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    HigherOrderGP,
    LatentKroneckerGP,
    MixedHigherOrderGP,
    MixedLatentKroneckerGP,
)


@pytest.mark.parametrize("mixed", [False, True])
def test_higher_order_fit_uses_torch_optimizer(mixed):
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    if mixed:
        X = torch.cat([X, (torch.arange(8) % 2).double().unsqueeze(-1)], dim=-1)
    Y = torch.sin(X[:, :1] * 3).repeat(1, 3)
    cls = MixedHigherOrderGP if mixed else HigherOrderGP
    model = cls(train_X=X, train_Y=Y, **({"cat_dims": [1]} if mixed else {}))
    assert model.supports_fit is True
    with patch("robotorchan.models.structured.higher_order.fit_gpytorch_mll_torch") as fit:
        assert model.fit(step_limit=2) is None
    fit.assert_called_once()
    args, kwargs = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert kwargs == {"step_limit": 2}
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)


@pytest.mark.parametrize("mixed", [False, True])
def test_latent_kronecker_fit_uses_iterative_context(mixed):
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    if mixed:
        X = torch.cat([X, (torch.arange(8) % 2).double().unsqueeze(-1)], dim=-1)
    T = torch.linspace(0.0, 1.0, 4, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(X[:, :1] + T.squeeze(-1))
    cls = MixedLatentKroneckerGP if mixed else LatentKroneckerGP
    model = cls(train_X=X, train_T=T, train_Y=Y, **({"cat_dims": [1]} if mixed else {}))
    assert model.supports_fit is True
    with patch("robotorchan.models.structured.latent_kronecker.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs={"options": {"maxiter": 2}}) is None
    fit.assert_called_once()
    args, kwargs = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert kwargs["optimizer_kwargs"] == {"options": {"maxiter": 2}}
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_T, T)
    torch.testing.assert_close(model.raw_train_Y, Y)
