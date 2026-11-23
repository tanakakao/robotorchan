"""Notebook 19 training contracts for frozen and jointly optimized neural GPs."""

from unittest.mock import patch

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    VAEGP,
    AutoEncoderGP,
    AutoEncoderMultiTaskGP,
    JointEncoderMultiTaskGP,
    JointVAEGP,
    MixedAutoEncoderGP,
    SupervisedVAEGP,
)


@pytest.mark.parametrize(
    "model_class",
    [AutoEncoderGP, VAEGP, SupervisedVAEGP, MixedAutoEncoderGP, AutoEncoderMultiTaskGP],
)
def test_frozen_neural_reducer_fit_uses_native_mll(model_class):
    torch.manual_seed(2)
    X = torch.rand(12, 6, dtype=torch.double)
    Y = torch.sin(4 * X[:, :1])
    kwargs = {"latent_dim": 2, "hidden_dims": (8,), "epochs": 2}
    if model_class is MixedAutoEncoderGP:
        X = torch.cat([X, (torch.arange(len(X)) % 2).to(X).unsqueeze(-1)], dim=-1)
        kwargs["cat_dims"] = [6]
    elif model_class is AutoEncoderMultiTaskGP:
        X = torch.cat(
            [
                torch.cat([X, torch.zeros(len(X), 1, dtype=X.dtype)], dim=-1),
                torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=-1),
            ]
        )
        Y = torch.cat([Y, Y + 0.2])
        kwargs["task_feature"] = 6
    model = model_class(X, Y, **kwargs)
    assert model.supports_fit is True
    with patch("robotorchan.models.base.fit_gpytorch_mll") as fit:
        assert model.fit(optimizer_kwargs={"options": {"maxiter": 2}}) is None
    args, _ = fit.call_args
    assert isinstance(args[0], ExactMarginalLogLikelihood)
    assert args[0].model is model
    torch.testing.assert_close(model.raw_train_X, X)


@pytest.mark.parametrize("model_class", [JointVAEGP, JointEncoderMultiTaskGP])
def test_joint_neural_fit_uses_specialized_training_loss(model_class):
    torch.manual_seed(3)
    X = torch.rand(12, 6, dtype=torch.double)
    Y = torch.sin(4 * X[:, :1])
    kwargs = {"latent_dim": 2, "hidden_dims": (8,)}
    if model_class is JointEncoderMultiTaskGP:
        X = torch.cat(
            [
                torch.cat([X, torch.zeros(len(X), 1, dtype=X.dtype)], dim=-1),
                torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=-1),
            ]
        )
        Y = torch.cat([Y, Y + 0.2])
        kwargs["task_feature"] = 6
    model = model_class(X, Y, **kwargs)
    assert model.supports_fit is True
    assert model.fit(epochs=2, lr=0.01) is None
    assert not model.training
    assert not model.likelihood.training


@pytest.mark.parametrize("epochs,lr", [(0, 0.01), (1, 0.0), (1, float("inf"))])
def test_joint_multitask_fit_validates_options(epochs, lr):
    X = torch.rand(8, 6, dtype=torch.double)
    Y = torch.sin(4 * X[:, :1])
    task = (torch.arange(len(X)) % 2).to(X).unsqueeze(-1)
    task_X = torch.cat([X, task], dim=-1)
    model = JointEncoderMultiTaskGP(
        task_X, Y + 0.2 * task, task_feature=6, latent_dim=2, hidden_dims=(8,)
    )
    with pytest.raises(ValueError):
        model.fit(epochs=epochs, lr=lr)
