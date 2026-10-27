"""Independent numerical reference for the nonstationary Gibbs covariance."""

from __future__ import annotations

import math

import torch

from robotorchan.models.robust.nonstationary import GibbsKernel


def test_gibbs_kernel_matches_manual_one_dimensional_formula() -> None:
    kernel = GibbsKernel(input_dim=1).double()
    with torch.no_grad():
        kernel.lengthscale_intercept.fill_(0.3)
        kernel.lengthscale_slope.fill_(0.8)
    X = torch.tensor([[0.0], [0.4], [0.9]], dtype=torch.double)
    lengthscales = torch.nn.functional.softplus(0.3 + 0.8 * X[:, 0]) + 1e-3

    expected = torch.empty((3, 3), dtype=torch.double)
    for i in range(3):
        for j in range(3):
            a = lengthscales[i].item()
            b = lengthscales[j].item()
            denominator = a * a + b * b
            distance = (X[i, 0] - X[j, 0]).item()
            expected[i, j] = math.sqrt(2 * a * b / denominator) * math.exp(
                -(distance * distance) / denominator
            )

    actual = kernel(X, X).to_dense()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, actual.T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        torch.diagonal(actual),
        torch.ones(3, dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.linalg.eigvalsh(actual).min() >= -1e-10
    torch.testing.assert_close(kernel(X, X, diag=True), torch.diagonal(actual))


def test_gibbs_kernel_constant_lengthscale_matches_rbf_limit() -> None:
    kernel = GibbsKernel(input_dim=1).double()
    with torch.no_grad():
        kernel.lengthscale_intercept.fill_(0.4)
        kernel.lengthscale_slope.zero_()
    X = torch.tensor([[0.0], [0.3], [0.8]], dtype=torch.double)
    ell = kernel.local_lengthscale(X)[0, 0]
    squared_distances = (X[:, None, 0] - X[None, :, 0]).square()
    expected = torch.exp(-squared_distances / (2 * ell.square()))
    torch.testing.assert_close(kernel(X, X).to_dense(), expected, atol=1e-10, rtol=1e-10)
