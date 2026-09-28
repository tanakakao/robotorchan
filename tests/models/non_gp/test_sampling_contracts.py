"""Sampling contracts for external probabilistic non-GP posteriors."""

import torch
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models.non_gp.distribution_posterior import GaussianDistributionPosterior
from robotorchan.models.non_gp.posterior import make_ensemble_posterior


def test_external_gaussian_posterior_supports_normal_samplers() -> None:
    mean = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    variance = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double)
    posterior = GaussianDistributionPosterior(mean=mean, variance=variance)

    for sampler_cls in (SobolQMCNormalSampler, IIDNormalSampler):
        samples = sampler_cls(torch.Size([32]), seed=123)(posterior)
        assert samples.shape == torch.Size([32, 3, 1])
        assert samples.dtype == mean.dtype
        assert samples.device == mean.device
        assert torch.isfinite(samples).all()


def test_external_gaussian_posterior_reuses_base_samples() -> None:
    mean = torch.zeros(3, 1, dtype=torch.double)
    variance = torch.full_like(mean, 0.25)
    posterior = GaussianDistributionPosterior(mean=mean, variance=variance)
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=456)

    first = sampler(posterior)
    second = sampler(posterior)

    torch.testing.assert_close(first, second)


def test_ensemble_posterior_uses_index_sampler() -> None:
    values = torch.arange(5 * 3, dtype=torch.double).reshape(5, 3, 1)
    posterior = make_ensemble_posterior(values)
    sampler = IndexSampler(torch.Size([32]), seed=789)

    samples = sampler(posterior)

    assert samples.shape == torch.Size([32, 3, 1])
    assert samples.dtype == values.dtype
    assert samples.device == values.device
    assert torch.isfinite(samples).all()
    for sample in samples:
        assert any(torch.equal(sample, member) for member in values)


def test_index_sampler_reuses_ensemble_indices() -> None:
    values = torch.arange(7 * 2, dtype=torch.double).reshape(7, 2, 1)
    posterior = make_ensemble_posterior(values)
    sampler = IndexSampler(torch.Size([16]), seed=987)

    first = sampler(posterior)
    second = sampler(posterior)

    torch.testing.assert_close(first, second)
