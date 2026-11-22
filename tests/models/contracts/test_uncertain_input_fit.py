"""Native exact-GP fit contracts for uncertain-input and categorical-input models."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedUncertainInputSingleTaskGP,
    UncertainCategoricalSingleTaskGP,
    UncertainInputSingleTaskGP,
)


@pytest.mark.parametrize("family", ["continuous", "mixed", "categorical"])
def test_uncertain_input_fit_reuses_native_mll(family):
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x)
    kwargs = {"optimizer_kwargs": {"options": {"maxiter": 2}}}

    if family == "continuous":
        uncertainty = torch.full_like(x, 0.03)
        model = UncertainInputSingleTaskGP(x, y, train_X_std=uncertainty)
        torch.testing.assert_close(model.raw_train_X, x)
        torch.testing.assert_close(model.raw_train_X_std, uncertainty)
    elif family == "mixed":
        categories = (torch.arange(8) % 2).to(x).unsqueeze(-1)
        X = torch.cat([x, categories], dim=-1)
        model = MixedUncertainInputSingleTaskGP(
            X, y + 0.1 * categories, cat_dims=[1], train_X_std=torch.full_like(x, 0.03)
        )
        torch.testing.assert_close(model.raw_train_X, X)
        torch.testing.assert_close(model.raw_train_X_std, torch.full_like(x, 0.03))
    else:
        probs = torch.stack([1.0 - x[:, 0], x[:, 0]], dim=-1)
        model = UncertainCategoricalSingleTaskGP(x, probs, y)
        torch.testing.assert_close(model.raw_train_X_cont, x)
        torch.testing.assert_close(model.raw_train_category_probabilities, probs)

    assert model.supports_fit is True
    assert model.supports_mll is True
    with patch("robotorchan.models.base.fit_gpytorch_mll") as fit:
        assert model.fit(**kwargs) is None
    fit.assert_called_once()
    args, call_kwargs = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert call_kwargs["optimizer_kwargs"] == kwargs["optimizer_kwargs"]
