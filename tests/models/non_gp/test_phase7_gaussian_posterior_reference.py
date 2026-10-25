"""Analytical reparameterization reference for non-GP Gaussian posteriors."""

import torch

from robotorchan.models.non_gp.distribution_posterior import GaussianDistributionPosterior


def test_gaussian_distribution_posterior_matches_affine_normal_reference() -> None:
    """Verify base-sample transformation, batch shape, and input gradients."""
    mean = torch.tensor(
        [[[1.0], [-2.0]], [[0.5], [3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    variance = torch.tensor(
        [[[0.25], [4.0]], [[1.0], [0.0]]],
        dtype=torch.float64,
    )
    posterior = GaussianDistributionPosterior(mean=mean, variance=variance)
    base_samples = torch.tensor(
        [
            [[[0.0], [1.0]], [[-1.0], [2.0]]],
            [[[2.0], [-0.5]], [[0.0], [-3.0]]],
        ],
        dtype=torch.float64,
    )

    actual = posterior.rsample_from_base_samples(
        sample_shape=torch.Size([2]),
        base_samples=base_samples,
    )
    expected = mean.unsqueeze(0) + torch.sqrt(variance).unsqueeze(0) * base_samples

    assert actual.shape == (2, 2, 2, 1)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(posterior.mean, mean)
    torch.testing.assert_close(posterior.variance, variance)
    torch.testing.assert_close(actual[..., 1, 0][..., 1], mean[1, 1].expand(2))

    actual.sum().backward()
    assert mean.grad is not None
    torch.testing.assert_close(mean.grad, torch.full_like(mean, 2.0))
