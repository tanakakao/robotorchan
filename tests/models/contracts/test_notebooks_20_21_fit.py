"""Native fit contracts exercised by Notebooks 20 and 21."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import PCAMultiTaskGP, MixedPCAGP, MixedPLSGP


@pytest.mark.parametrize("model_name", ["pca_multitask", "mixed_pca", "mixed_pls"])
def test_notebook_20_21_model_fit_delegates_to_botorch(model_name):
    torch.manual_seed(0)
    X = torch.rand(12, 5, dtype=torch.double)
    Y = torch.sin(4 * X[:, :1])
    if model_name == "pca_multitask":
        X = torch.cat(
            [
                torch.cat([X, torch.zeros(len(X), 1, dtype=X.dtype)], dim=-1),
                torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=-1),
            ],
            dim=0,
        )
        Y = torch.cat([Y, Y + 0.2], dim=0)
        model = PCAMultiTaskGP(X, Y, task_feature=-1, n_components=2)
    else:
        category = (torch.arange(len(X)) % 3).to(X).unsqueeze(-1)
        X = torch.cat([X, category], dim=-1)
        Y = Y + 0.2 * category
        model_cls = MixedPCAGP if model_name == "mixed_pca" else MixedPLSGP
        model = model_cls(X, Y, n_components=2, cat_dims=[5])

    assert model.supports_fit is True
    assert model.supports_mll is True
    optimizer_kwargs = {"options": {"maxiter": 2}}
    with patch("robotorchan.models.base.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs=optimizer_kwargs) is None
    fit.assert_called_once()
    args, kwargs = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert kwargs["optimizer_kwargs"] == optimizer_kwargs
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
