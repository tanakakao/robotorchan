"""Independent Gaussian-input expected-RBF kernel numerical reference."""

from __future__ import annotations

import math

import torch

from robotorchan.models.uncertain.uncertain_input import GaussianUncertainInputKernel


def test_gaussian_uncertain_kernel_matches_one_dimensional_formula() -> None:
    kernel = GaussianUncertainInputKernel(input_dim=1).double()
    with torch.no_grad():
        kernel.lengthscale = 0.7
    means = torch.tensor([0.0, 0.4, 0.9], dtype=torch.double)
    variances = torch.tensor([0.01, 0.04, 0.02], dtype=torch.double)
    X = torch.stack([means, variances], dim=-1)
    lengthscale_sq = kernel.lengthscale.item() ** 2

    expected = torch.empty((3, 3), dtype=torch.double)
    for i in range(3):
        for j in range(3):
            total_variance = variances[i].item() + variances[j].item()
            denominator = lengthscale_sq + total_variance
            delta = means[i].item() - means[j].item()
            expected[i, j] = math.sqrt(lengthscale_sq / denominator) * math.exp(
                -delta * delta / (2 * denominator)
            )

    actual = kernel(X, X).to_dense()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, actual.T, atol=1e-10, rtol=1e-10)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-10
    torch.testing.assert_close(kernel(X, X, diag=True), actual.diagonal())


def test_zero_input_uncertainty_reduces_to_rbf() -> None:
    kernel = GaussianUncertainInputKernel(input_dim=1).double()
    with torch.no_grad():
        kernel.lengthscale = 0.8
    means = torch.tensor([0.0, 0.3, 0.8], dtype=torch.double)
    X = torch.stack([means, torch.zeros_like(means)], dim=-1)
    ell = kernel.lengthscale.item()
    squared_distances = (means[:, None] - means[None, :]).square()
    expected = torch.exp(-squared_distances / (2 * ell * ell))
    torch.testing.assert_close(kernel(X, X).to_dense(), expected, atol=1e-10, rtol=1e-10)
