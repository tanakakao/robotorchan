"""Tests for ALEBO Mahalanobis GP geometry."""

import torch
from gpytorch.kernels import ScaleKernel

from robotorchan.models import ALEBOGP, MahalanobisRBFKernel


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
    train_Y = (train_X[:, :1].square() + train_X[:, 1:].square())

    model = ALEBOGP(train_X, train_Y)

    assert isinstance(model.covar_module, ScaleKernel)
    assert isinstance(model.covar_module.base_kernel, MahalanobisRBFKernel)
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
    assert model.make_mll().model is model

    posterior = model.posterior(torch.zeros(1, 2, dtype=torch.double))
    assert posterior.mean.shape == (1, 1)
