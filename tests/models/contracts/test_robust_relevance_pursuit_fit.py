"""fit() must retain BoTorch's robust relevance-pursuit MLL dispatch."""

from unittest.mock import patch

import pytest
import torch
from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitMixin
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedRobustRelevancePursuitMultiTaskGP,
    MixedRobustRelevancePursuitSingleTaskGP,
    RobustRelevancePursuitMultiTaskGP,
    RobustRelevancePursuitSingleTaskGP,
)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("multitask", [False, True])
def test_robust_relevance_pursuit_fit_uses_native_dispatch(mixed, multitask):
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    X = x
    kwargs = {"convex_parameterization": True}
    if mixed:
        category = (torch.arange(8) % 2).to(x).unsqueeze(-1)
        X = torch.cat([X, category], dim=-1)
        kwargs["cat_dims"] = [1]
    if multitask:
        task = (torch.arange(8) % 2).to(x).unsqueeze(-1)
        X = torch.cat([X, task], dim=-1)
        kwargs["task_feature"] = -1
        kwargs["output_tasks"] = [0]
    Y = torch.sin(x * 3)
    if multitask:
        Y = Y + 0.15 * task
    cls = (
        MixedRobustRelevancePursuitMultiTaskGP
        if mixed and multitask
        else RobustRelevancePursuitMultiTaskGP
        if multitask
        else MixedRobustRelevancePursuitSingleTaskGP
        if mixed
        else RobustRelevancePursuitSingleTaskGP
    )
    model = cls(X, Y, **kwargs)
    assert isinstance(model, RobustRelevancePursuitMixin)
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
