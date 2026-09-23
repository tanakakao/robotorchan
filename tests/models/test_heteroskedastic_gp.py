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


def test_heteroskedastic_kronecker_prototype_preserves_task_specific_noise_shape() -> None:
    from robotorchan.models.robust.robust import HeteroskedasticKroneckerMultiTaskGP

    X = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((torch.sin(4.0 * X), torch.cos(3.0 * X)), dim=-1)
    Yvar = torch.stack(
        (torch.full((6,), 0.01, dtype=torch.double), torch.full((6,), 0.04, dtype=torch.double)),
        dim=-1,
    )
    model = HeteroskedasticKroneckerMultiTaskGP(X, Y, train_Yvar=Yvar)

    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    torch.testing.assert_close(model.raw_train_Yvar, Yvar)
    assert model.raw_train_Yvar.shape == Y.shape


def test_heteroskedastic_kronecker_prototype_noise_model_keeps_output_axis() -> None:
    from robotorchan.models.robust.robust import HeteroskedasticKroneckerMultiTaskGP

    X = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((torch.sin(4.0 * X), torch.cos(3.0 * X)), dim=-1)
    model = HeteroskedasticKroneckerMultiTaskGP(X, Y)
    log_noise = torch.stack(
        (torch.linspace(-4.0, -3.0, 6), torch.linspace(-2.0, -1.0, 6)), dim=-1
    ).to(dtype=torch.double)
    noise_model = model.build_noise_model(log_noise)
    model.set_noise_model(noise_model)
    noise_model.eval()
    noise_model.likelihood.eval()

    noise = model.predicted_noise(X[:2])
    diagonal = model.observation_covariance_diagonal(X[:2])

    assert noise.shape == (2, 2)
    assert diagonal.shape == (4,)
    assert torch.isfinite(noise).all()
    assert torch.all(noise >= model.noise_floor)
    assert not torch.allclose(noise[:, 0], noise[:, 1])


def test_heteroskedastic_kronecker_prototype_rejects_broadcast_noise() -> None:
    from robotorchan.models.robust.robust import HeteroskedasticKroneckerMultiTaskGP

    X = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((torch.sin(4.0 * X), torch.cos(3.0 * X)), dim=-1)
    bad_Yvar = torch.full((6, 1), 0.01, dtype=torch.double)

    with pytest.raises(ValueError, match="same shape"):
        HeteroskedasticKroneckerMultiTaskGP(X, Y, train_Yvar=bad_Yvar)
