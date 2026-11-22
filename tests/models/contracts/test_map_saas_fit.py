"""Regression tests for MAP-SAAS model-owned fitting entry points."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)


@pytest.mark.parametrize(
    "model_class,mixed",
    [
        (AdditiveMapSaasSingleTaskGP, False),
        (EnsembleMapSaasSingleTaskGP, False),
        (MixedAdditiveMapSaasSingleTaskGP, True),
        (MixedEnsembleMapSaasSingleTaskGP, True),
    ],
)
def test_map_saas_fit_uses_native_mll(model_class, mixed):
    train_X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    if mixed:
        categories = torch.arange(8, dtype=torch.double).remainder(2).unsqueeze(-1)
        train_X = torch.cat((train_X, categories), dim=-1)
    train_Y = torch.sin(train_X[:, :1] * 3)
    kwargs = {"cat_dims": [1]} if mixed else {}
    model = model_class(train_X, train_Y, num_taus=2, **kwargs)

    assert model.supports_fit is True
    assert model.supports_mll is True
    with patch("robotorchan.models.base.fit_gpytorch_mll") as native_fit:
        assert model.fit(optimizer_kwargs={"options": {"maxiter": 2}}) is None

    native_fit.assert_called_once()
    args, call_kwargs = native_fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert call_kwargs["optimizer_kwargs"] == {"options": {"maxiter": 2}}
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
