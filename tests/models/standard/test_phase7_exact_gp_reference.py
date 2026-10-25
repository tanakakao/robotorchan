"""Independent exact-GP posterior conditioning reference tests."""

import torch
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean

from robotorchan.models import SingleTaskGP


def test_exact_gp_posterior_matches_independent_cholesky_conditioning() -> None:
    train_x = torch.tensor([[0.05], [0.3], [0.7], [0.95]], dtype=torch.double)
    train_y = torch.tensor([[0.2], [-0.4], [0.8], [0.1]], dtype=torch.double)
    test_x = torch.tensor([[0.15], [0.55], [0.85]], dtype=torch.double)
    noise = 0.025
    lengthscale = 0.4

    kernel = RBFKernel()
    kernel.lengthscale = lengthscale
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=torch.full_like(train_y, noise),
        covar_module=kernel,
        mean_module=ZeroMean(),
        outcome_transform=None,
    )
    model.eval()
    model.likelihood.eval()

    # Independent analytic RBF matrix: no model.posterior or kernel.forward.
    def covariance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(left, right).square()
        return torch.exp(-distances / (2.0 * lengthscale**2))

    k_xx = covariance(train_x, train_x)
    k_xs = covariance(train_x, test_x)
    k_ss = covariance(test_x, test_x)
    system = k_xx + noise * torch.eye(train_x.shape[0], dtype=torch.double)
    cholesky = torch.linalg.cholesky(system)
    alpha = torch.cholesky_solve(train_y, cholesky)
    reference_mean = k_xs.T @ alpha
    reference_covariance = k_ss - k_xs.T @ torch.cholesky_solve(k_xs, cholesky)

    posterior = model.posterior(test_x)
    actual_covariance = posterior.distribution.covariance_matrix

    torch.testing.assert_close(posterior.mean, reference_mean, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_covariance, reference_covariance, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        posterior.variance.squeeze(-1),
        reference_covariance.diagonal(),
        atol=2e-5,
        rtol=2e-5,
    )
