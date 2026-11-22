"""Fit contracts for Student-t and contaminated variational observation models."""

import pytest
import torch

from robotorchan.models import (
    ContaminatedMultiTaskGP,
    ContaminatedSingleTaskGP,
    MixedContaminatedMultiTaskGP,
    MixedContaminatedSingleTaskGP,
    MixedStudentTMultiTaskGP,
    MixedStudentTSingleTaskGP,
    StudentTMultiTaskGP,
    StudentTSingleTaskGP,
)


@pytest.mark.parametrize("family", ["student", "contaminated"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("multitask", [False, True])
def test_observation_model_fit_runs_native_training_loss(family, mixed, multitask):
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    X = x
    Y = torch.sin(2 * torch.pi * x)
    kwargs = {"num_inducing": 4}
    if mixed:
        category = (torch.arange(8) % 2).to(x).unsqueeze(-1)
        X = torch.cat([X, category], dim=-1)
        kwargs["cat_dims"] = [1]
        Y = Y + 0.1 * category
    if multitask:
        task = (torch.arange(8) % 2).to(x).unsqueeze(-1)
        X = torch.cat([X, task], dim=-1)
        kwargs["task_feature"] = X.shape[-1] - 1
        Y = Y + 0.2 * task
    model_class = {
        ("student", False, False): StudentTSingleTaskGP,
        ("student", True, False): MixedStudentTSingleTaskGP,
        ("student", False, True): StudentTMultiTaskGP,
        ("student", True, True): MixedStudentTMultiTaskGP,
        ("contaminated", False, False): ContaminatedSingleTaskGP,
        ("contaminated", True, False): MixedContaminatedSingleTaskGP,
        ("contaminated", False, True): ContaminatedMultiTaskGP,
        ("contaminated", True, True): MixedContaminatedMultiTaskGP,
    }[(family, mixed, multitask)]
    if family == "contaminated":
        kwargs["num_likelihood_samples"] = 2
    model = model_class(X, Y, **kwargs)
    assert model.supports_fit is True
    assert model.supports_mll is False
    assert model.fit(steps=2, lr=0.01) is None
    assert model.training is False
    with torch.no_grad():
        prediction = model.posterior(X[:2]).mean
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)


@pytest.mark.parametrize(
    "steps,lr", [(0, 0.03), (1, 0.0), (1, -0.1), (1, float("inf"))]
)
def test_observation_fit_rejects_invalid_optimizer_settings(steps, lr):
    X = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X)
    model = StudentTSingleTaskGP(X, Y, num_inducing=3)
    with pytest.raises(ValueError):
        model.fit(steps=steps, lr=lr)
