"""Regression tests for DeepGP posterior trajectory sampling."""

import torch

from robotorchan.models.deep_gp_posterior import DeepGPPosterior


def test_base_samples_preserve_t_batch_dimensions() -> None:
    stored = torch.arange(5 * 3 * 2, dtype=torch.double).reshape(5, 3, 2, 1)
    posterior = DeepGPPosterior(stored)
    base_samples = torch.randn(7, 3, 2, 1, dtype=torch.double)

    samples = posterior.rsample_from_base_samples(torch.Size([7]), base_samples)

    assert samples.shape == (7, 3, 2, 1)


def test_normal_base_samples_select_empirical_trajectories_without_shape_leak() -> None:
    stored = torch.randn(11, 4, 1, 1, dtype=torch.double)
    posterior = DeepGPPosterior(stored)
    base_samples = torch.linspace(-3.0, 3.0, 9, dtype=torch.double).reshape(9, 1, 1, 1)
    base_samples = base_samples.expand(9, 4, 1, 1)

    samples = posterior.rsample_from_base_samples(torch.Size([9]), base_samples)

    assert samples.shape == (9, 4, 1, 1)
    assert torch.isfinite(samples).all()
