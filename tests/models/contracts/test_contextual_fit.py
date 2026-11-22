"""Contextual GP models use the shared native Exact GP fit() contract."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import LCEAGP, LCEMGP, SACGP, MixedLCEMGP


@pytest.mark.parametrize("model_class", [SACGP, LCEAGP, LCEMGP, MixedLCEMGP])
def test_contextual_fit_delegates_to_native_mll(model_class):
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    if model_class in (SACGP, LCEAGP):
        X = torch.cat([x, 1.0 - x], dim=-1)
        Y = torch.sin(x * 3) + 0.3 * (1.0 - x)
        kwargs = {
            "train_Yvar": None,
            "decomposition": {"context_0": [0], "context_1": [1]},
        }
    else:
        task = (torch.arange(8) % 2).to(torch.double).unsqueeze(-1)
        if model_class is MixedLCEMGP:
            category = (torch.arange(8) % 2).to(torch.double).unsqueeze(-1)
            X = torch.cat([category, x, task], dim=-1)
            kwargs = {
                "task_feature": 2,
                "cat_dims": [0],
                "context_cat_feature": torch.tensor([[0.0], [1.0]], dtype=torch.double),
            }
        else:
            X = torch.cat([x, task], dim=-1)
            kwargs = {"task_feature": 1}
        Y = torch.sin(x * 3) + 0.2 * task

    model = model_class(train_X=X, train_Y=Y, **kwargs)
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
    torch.testing.assert_close(model.raw_train_Y, Y)
