"""Fit API contract for the independent exact-GP model list."""

from unittest.mock import patch

import torch
from gpytorch.mlls import SumMarginalLogLikelihood

from robotorchan.models import ModelListGP, SingleTaskGP


def test_model_list_fit_uses_summed_mll() -> None:
    train_X = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    model = ModelListGP(
        SingleTaskGP(train_X, torch.sin(train_X)),
        SingleTaskGP(train_X, torch.cos(train_X)),
    )

    assert model.supports_fit is True
    assert isinstance(model.make_mll(), SumMarginalLogLikelihood)

    with patch("robotorchan.models.standard.model_list.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs={"options": {"maxiter": 5}}) is None

    fit.assert_called_once()
    args, kwargs = fit.call_args
    assert isinstance(args[0], SumMarginalLogLikelihood)
    assert kwargs["optimizer_kwargs"] == {"options": {"maxiter": 5}}
