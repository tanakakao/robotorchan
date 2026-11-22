"""Hierarchical GP fit() uses the existing BoTorch exact-MLL fitting path."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
    MixedHierarchicalConditionalKernelGP,
    MixedHierarchicalConditionalKernelMultiTaskGP,
)


@pytest.mark.parametrize(
    "model_class,multitask,mixed",
    [
        (HierarchicalConditionalKernelGP, False, False),
        (MixedHierarchicalConditionalKernelGP, False, True),
        (HierarchicalConditionalKernelMultiTaskGP, True, False),
        (MixedHierarchicalConditionalKernelMultiTaskGP, True, True),
    ],
)
def test_hierarchical_fit_delegates_to_native_mll(model_class, multitask, mixed):
    dtype = torch.double
    n = 8
    common = torch.linspace(0.1, 0.9, n, dtype=dtype).unsqueeze(-1)
    parent = (torch.arange(n) % 2).to(dtype).unsqueeze(-1)
    temperature = torch.linspace(0.2, 0.8, n, dtype=dtype).unsqueeze(-1)
    pressure = torch.linspace(0.8, 0.2, n, dtype=dtype).unsqueeze(-1)
    X = torch.cat([common, parent, temperature, pressure], dim=-1)
    Y = torch.sin(common * 3) + torch.where(
        parent == 0, 0.5 * temperature, -0.5 * pressure
    )
    kwargs = {
        "hierarchical_dependencies": {1: {0: [2], 1: [3]}},
        "use_saas_prior": False,
    }
    if mixed:
        # A categorical non-parent design dimension is supported.
        X[:, 0] = (torch.arange(n) % 2).to(dtype)
        kwargs["cat_dims"] = [0]
    if multitask:
        X = torch.cat([X, (torch.arange(n) % 2).to(dtype).unsqueeze(-1)], dim=-1)
        kwargs["task_feature"] = 4
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
