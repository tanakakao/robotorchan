"""Tests for mixed iterative and joint heteroskedastic surrogates."""

import pytest
import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models import (
    MixedHeteroskedasticSingleTaskGP,
    MixedJointHeteroskedasticSingleTaskGP,
)


def _data():
    x = torch.linspace(0.05, 0.95, 8, dtype=torch.double)
    category = torch.tensor([0, 1] * 4, dtype=torch.double)
    X = torch.stack([x, category], dim=-1)
    Y = (torch.sin(2 * torch.pi * x) + 0.15 * category).unsqueeze(-1)
    return X, Y


def test_mixed_iterative_heteroskedastic_contract() -> None:
    X, Y = _data()
    model = MixedHeteroskedasticSingleTaskGP(X, Y, cat_dims=[1])
    assert model.cat_dims == (1,)
    assert model.supports_mll
    posterior = model.posterior(X[:2])
    assert torch.isfinite(posterior.mean).all()


def test_mixed_joint_heteroskedastic_training_and_acquisition() -> None:
    X, Y = _data()
    model = MixedJointHeteroskedasticSingleTaskGP(
        X,
        Y,
        cat_dims=[1],
        num_inducing=4,
        num_mc_samples=4,
    )
    loss = model.training_loss()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.cat_dims == (1,)
    assert model.response_model.cat_dims == (1,)
    assert model.noise_model.cat_dims == (1,)
    assert torch.isfinite(model.predicted_noise(X[:2])).all()
    acq = qUpperConfidenceBound(model=model, beta=0.2)
    assert torch.isfinite(acq(X[:2].unsqueeze(0))).all()


def test_mixed_iterative_heteroskedastic_requires_fit_for_noise_posterior() -> None:
    X, Y = _data()
    model = MixedHeteroskedasticSingleTaskGP(X, Y, cat_dims=[1])
    with pytest.raises(RuntimeError, match="fit_heteroskedastic"):
        model.noise_posterior(X)


def test_mixed_iterative_heteroskedastic_fits_mixed_noise_process() -> None:
    X, Y = _data()
    model = MixedHeteroskedasticSingleTaskGP(X, Y, cat_dims=[1])
    returned = model.fit_heteroskedastic(iterations=1)
    assert returned is model
    assert model.noise_model is not None
    assert model.noise_model.cat_dims == (1,)
    noise = model.predicted_noise(X)
    assert noise.shape == Y.shape
    assert torch.all(noise >= model.noise_floor)
    assert torch.isfinite(noise).all()


def test_mixed_heteroskedastic_kronecker_prototype_uses_mixed_response_and_noise() -> None:
    from robotorchan.models.robust.robust import MixedHeteroskedasticKroneckerMultiTaskGP

    X = torch.tensor(
        [[0.15, 0.0], [0.30, 1.0], [0.45, 0.0], [0.60, 1.0], [0.75, 0.0], [0.85, 1.0]],
        dtype=torch.double,
    )
    Y = torch.stack((torch.sin(4.0 * X[:, 0]), torch.cos(3.0 * X[:, 0])), dim=-1)
    model = MixedHeteroskedasticKroneckerMultiTaskGP(X, Y, cat_dims=[-1])
    log_noise = torch.stack(
        (torch.linspace(-4.0, -3.0, 6), torch.linspace(-2.0, -1.0, 6)), dim=-1
    ).to(dtype=torch.double)
    noise_model = model.build_noise_model(log_noise)
    model.set_noise_model(noise_model)
    model.eval()
    model.likelihood.eval()
    noise_model.eval()
    noise_model.likelihood.eval()

    assert model.cat_dims == (1,)
    assert noise_model.cat_dims == (1,)
    assert model.covar_module.data_covar_module is not None
    noise = model.predicted_noise(X[:2])
    assert noise.shape == (2, 2)
    assert torch.isfinite(noise).all()


def test_mixed_heteroskedastic_kronecker_rejects_mismatched_noise_categories() -> None:
    from robotorchan.models.robust.robust import MixedHeteroskedasticKroneckerMultiTaskGP
    from robotorchan.models.standard.multitask import MixedKroneckerMultiTaskGP

    X = torch.tensor([[0.15, 0.0], [0.30, 1.0], [0.45, 0.0], [0.60, 1.0]], dtype=torch.double)
    Y = torch.stack((torch.sin(4.0 * X[:, 0]), torch.cos(3.0 * X[:, 0])), dim=-1)
    model = MixedHeteroskedasticKroneckerMultiTaskGP(X, Y, cat_dims=[1])
    wrong_noise_model = MixedKroneckerMultiTaskGP(X, torch.log(Y.square() + 0.1), cat_dims=[0])

    with pytest.raises(ValueError, match="cat_dims"):
        model.set_noise_model(wrong_noise_model)
