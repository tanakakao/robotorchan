"""Native fit() contract for heterogeneous multitask GP models."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import HeterogeneousMTGP, MixedHeterogeneousMTGP


@pytest.mark.parametrize("mixed", [False, True])
def test_heterogeneous_fit_delegates_to_native_mll(mixed):
    train_Xs = [
        torch.tensor(
            [[0.10, 0.0], [0.30, 1.0], [0.60, 0.0], [0.85, 1.0]],
            dtype=torch.double,
        ),
        torch.tensor(
            [[0.15, 0.0, 0.20], [0.35, 1.0, 0.40],
             [0.65, 0.0, 0.70], [0.90, 1.0, 0.95]],
            dtype=torch.double,
        ),
    ]
    train_Ys = [
        train_Xs[0][:, :1] + 0.2 * train_Xs[0][:, 1:2],
        train_Xs[1][:, :1] + 0.2 * train_Xs[1][:, 1:2]
        + 0.1 * train_Xs[1][:, 2:3],
    ]
    kwargs = {"cat_dims": [1]} if mixed else {}
    model_class = MixedHeterogeneousMTGP if mixed else HeterogeneousMTGP
    model = model_class(
        train_Xs=train_Xs,
        train_Ys=train_Ys,
        train_Yvars=None,
        feature_indices=[[0, 1], [0, 1, 2]],
        full_feature_dim=3,
        use_saas_prior=False,
        **kwargs,
    )

    assert model.supports_fit is True
    assert model.supports_mll is True
    optimizer_kwargs = {"options": {"maxiter": 2}}
    with patch("robotorchan.models.structured.heterogeneous.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs=optimizer_kwargs) is None
    fit.assert_called_once()
    args, call_kwargs = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    assert call_kwargs["optimizer_kwargs"] == optimizer_kwargs
    for stored, original in zip(model.raw_train_Xs, train_Xs, strict=True):
        torch.testing.assert_close(stored, original)
    for stored, original in zip(model.raw_train_Ys, train_Ys, strict=True):
        torch.testing.assert_close(stored, original)
