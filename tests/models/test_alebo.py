"""Tests for ALEBO Mahalanobis GP geometry."""

import pytest
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

from robotorchan.models import ALEBOGP
from robotorchan.models.alebo import MahalanobisRBFKernel


def test_mahalanobis_metric_is_symmetric_positive_definite() -> None:
    kernel = MahalanobisRBFKernel(ard_num_dims=3).double()

    metric = kernel.metric
    eigenvalues = torch.linalg.eigvalsh(metric)

    torch.testing.assert_close(metric, metric.transpose(-2, -1))
    assert torch.all(eigenvalues > 0)


def test_mahalanobis_kernel_has_full_metric_parameters() -> None:
    kernel = MahalanobisRBFKernel(ard_num_dims=3).double()

    with torch.no_grad():
        kernel.raw_tril.copy_(
            torch.tensor(
                [[0.2, 0.0, 0.0], [0.5, -0.3, 0.0], [-0.4, 0.7, 0.1]],
                dtype=torch.double,
            )
        )

    metric = kernel.metric
    assert metric.shape == (3, 3)
    assert not torch.equal(metric, torch.diag(torch.diagonal(metric)))


def test_kernel_matches_exp_of_mahalanobis_distance() -> None:
    kernel = MahalanobisRBFKernel(ard_num_dims=2).double()
    x1 = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    x2 = torch.tensor([[0.4, -0.2]], dtype=torch.double)

    covariance = kernel(x1, x2).to_dense()
    delta = x1[0] - x2[0]
    expected = torch.exp(-0.5 * (delta @ kernel.metric @ delta))

    torch.testing.assert_close(covariance.squeeze(), expected)


def test_alebo_gp_uses_mahalanobis_kernel_and_common_contract() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()

    model = ALEBOGP(train_X, train_Y)

    assert isinstance(model.covar_module, ScaleKernel)
    assert isinstance(model.covar_module.base_kernel, MahalanobisRBFKernel)
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
    assert model.make_mll().model is model

    posterior = model.posterior(torch.zeros(1, 2, dtype=torch.double))
    assert posterior.mean.shape == (1, 1)


def test_alebo_gp_exposes_metric_from_internal_kernel() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()

    model = ALEBOGP(train_X, train_Y)

    torch.testing.assert_close(model.metric, model.covar_module.base_kernel.metric)


def test_alebo_gp_fit_returns_same_model(monkeypatch) -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()
    model = ALEBOGP(train_X, train_Y)
    captured = {}

    def fake_fit(mll, **kwargs):
        captured["mll"] = mll
        captured["kwargs"] = kwargs
        return mll

    monkeypatch.setattr("robotorchan.models.alebo.fit_gpytorch_mll", fake_fit)

    fitted = model.fit(optimizer_kwargs={"options": {"maxiter": 3}})

    assert fitted is model
    assert captured["mll"].model is model
    assert captured["kwargs"] == {"optimizer_kwargs": {"options": {"maxiter": 3}}}


def test_alebo_gp_rejects_non_mahalanobis_covar_for_metric_access() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y, covar_module=ScaleKernel(RBFKernel()))

    with pytest.raises(TypeError, match="MahalanobisRBFKernel"):
        _ = model.metric
