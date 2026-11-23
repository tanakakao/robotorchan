"""Native exact-GP fitting for the expressive models used in Notebook 22."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

import robotorchan.models as models


@pytest.mark.parametrize(
    "model_name",
    [
        "InfiniteWidthBNNGP",
        "MixedInfiniteWidthBNNGP",
        "InfiniteWidthBNNMultiTaskGP",
        "MixedInfiniteWidthBNNMultiTaskGP",
        "InfiniteWidthBNNKroneckerMultiTaskGP",
        "MixedInfiniteWidthBNNKroneckerMultiTaskGP",
        "SpectralMixtureGP",
        "MixedSpectralMixtureGP",
        "SpectralMixtureMultiTaskGP",
        "MixedSpectralMixtureMultiTaskGP",
        "SpectralMixtureKroneckerMultiTaskGP",
        "MixedSpectralMixtureKroneckerMultiTaskGP",
    ],
)
def test_notebook22_exact_expressive_fit_uses_native_mll(model_name):
    torch.manual_seed(0)
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x)
    mixed = model_name.startswith("Mixed")
    multitask = "MultiTask" in model_name and "Kronecker" not in model_name
    kronecker = "Kronecker" in model_name

    if mixed:
        category = (torch.arange(len(x)) % 2).to(x).unsqueeze(-1)
        x = torch.cat([x, category], dim=-1)
    if multitask:
        task0 = torch.cat([x, torch.zeros(len(x), 1, dtype=x.dtype)], dim=-1)
        task1 = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=-1)
        x = torch.cat([task0, task1], dim=0)
        y = torch.cat([y, y + 0.2], dim=0)
    elif kronecker:
        y = torch.cat([y, y + 0.2], dim=-1)

    kwargs = {}
    if model_name.startswith(("SpectralMixture", "MixedSpectralMixture")):
        kwargs["num_mixtures"] = 2
    else:
        kwargs["depth"] = 2
    if mixed:
        kwargs["cat_dims"] = [1]
    if multitask:
        kwargs["task_feature"] = x.shape[-1] - 1

    model = getattr(models, model_name)(x, y, **kwargs)
    assert model.supports_fit is True
    assert model.supports_mll is True
    optimizer_kwargs = {"options": {"maxiter": 2}}
    with patch("robotorchan.models.base.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs=optimizer_kwargs) is None
    fit.assert_called_once()
    args, forwarded = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert forwarded["optimizer_kwargs"] == optimizer_kwargs
    torch.testing.assert_close(model.raw_train_X, x)
    torch.testing.assert_close(model.raw_train_Y, y)
