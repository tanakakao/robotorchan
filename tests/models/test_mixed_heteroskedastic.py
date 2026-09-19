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
