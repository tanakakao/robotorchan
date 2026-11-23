"""Native fit() contracts for reduced-input and reduced-output exact GP models."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedPCAGP,
    MixedReducedKroneckerMultiTaskGP,
    MixedReducedMultiTaskGP,
    OutputPCAGP,
    PCAGP,
    PCAKroneckerMultiTaskGP,
    PCAMultiTaskGP,
    PLSGP,
    RandomProjectionGP,
    ReducedGP,
)
from robotorchan.reduction import PCAInputReducer


@pytest.mark.parametrize(
    "family",
    [
        "pca",
        "pls",
        "random",
        "generic",
        "mixed",
        "multitask",
        "mixed_multitask",
        "kronecker",
        "mixed_kronecker",
        "output",
    ],
)
def test_notebook18_reduced_fit_delegates_to_native_mll(family):
    torch.manual_seed(0)
    X = torch.rand(12, 5, dtype=torch.double)
    Y = torch.sin(2 * torch.pi * X[:, :1])
    if family == "pca":
        model = PCAGP(X, Y, n_components=2)
    elif family == "pls":
        model = PLSGP(X, Y, n_components=2)
    elif family == "random":
        model = RandomProjectionGP(X, Y, n_components=2, random_state=0)
    elif family == "generic":
        model = ReducedGP(X, Y, input_reducer=PCAInputReducer(n_components=2))
    elif family == "mixed":
        categories = (torch.arange(12) % 2).to(X).unsqueeze(-1)
        X = torch.cat([X, categories], dim=-1)
        Y = Y + 0.1 * categories
        model = MixedPCAGP(X, Y, n_components=2, cat_dims=[5])
    elif family in ("multitask", "mixed_multitask"):
        if family == "mixed_multitask":
            categories = (torch.arange(12) % 2).to(X).unsqueeze(-1)
            X = torch.cat([X, categories], dim=-1)
            Y = Y + 0.1 * categories
        X = torch.cat(
            [
                torch.cat([X, torch.zeros(12, 1, dtype=X.dtype)], dim=-1),
                torch.cat([X, torch.ones(12, 1, dtype=X.dtype)], dim=-1),
            ],
            dim=0,
        )
        Y = torch.cat([Y, Y + 0.2], dim=0)
        if family == "multitask":
            model = PCAMultiTaskGP(X, Y, task_feature=5, n_components=2)
        else:
            model = MixedReducedMultiTaskGP(
                X,
                Y,
                task_feature=6,
                cat_dims=[5],
                input_reducer=PCAInputReducer(n_components=2),
            )
    elif family in ("kronecker", "mixed_kronecker"):
        if family == "mixed_kronecker":
            categories = (torch.arange(12) % 2).to(X).unsqueeze(-1)
            X = torch.cat([X, categories], dim=-1)
            Y = Y + 0.1 * categories
        Y = torch.cat([Y, Y + 0.2], dim=-1)
        if family == "kronecker":
            model = PCAKroneckerMultiTaskGP(X, Y, n_components=2)
        else:
            model = MixedReducedKroneckerMultiTaskGP(
                X, Y, cat_dims=[5], input_reducer=PCAInputReducer(n_components=2)
            )
    else:
        Y = torch.cat([Y, Y + 0.2, 0.5 * Y], dim=-1)
        model = OutputPCAGP(X, Y, n_components=2)

    assert model.supports_fit is True
    assert model.supports_mll is True
    optimizer_kwargs = {"options": {"maxiter": 2}}
    with patch("robotorchan.models.base.fit_gpytorch_mll") as native_fit:
        assert model.fit(optimizer_kwargs=optimizer_kwargs) is None
    native_fit.assert_called_once()
    args, kwargs = native_fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert kwargs["optimizer_kwargs"] == optimizer_kwargs
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
