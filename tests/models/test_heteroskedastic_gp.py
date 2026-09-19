"""Tests for the iterative heteroskedastic single-task GP."""

import pytest
import torch

from robotorchan.models import HeteroskedasticSingleTaskGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    scale = 0.02 + 0.15 * X
    Y = torch.sin(2 * torch.pi * X) + scale * torch.randn_like(X)
    return X, Y


def test_heteroskedastic_gp_retains_raw_data() -> None:
    X, Y = _data()
    model = HeteroskedasticSingleTaskGP(X, Y)
    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_Y, Y)


def test_heteroskedastic_gp_fits_noise_process() -> None:
    X, Y = _data()
    model = HeteroskedasticSingleTaskGP(X, Y)
    returned = model.fit_heteroskedastic(iterations=1)
    assert returned is model
    noise = model.predicted_noise(X)
    assert noise.shape == Y.shape
    assert torch.all(noise >= model.noise_floor)
    assert torch.isfinite(noise).all()


def test_noise_posterior_requires_fit() -> None:
    X, Y = _data()
    model = HeteroskedasticSingleTaskGP(X, Y)
    with pytest.raises(RuntimeError, match="fit_heteroskedastic"):
        model.noise_posterior(X)


def test_heteroskedastic_iterations_validate_positive() -> None:
    X, Y = _data()
    model = HeteroskedasticSingleTaskGP(X, Y)
    with pytest.raises(ValueError, match="at least 1"):
        model.fit_heteroskedastic(iterations=0)
