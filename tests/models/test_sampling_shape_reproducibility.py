"""Cross-model shape, base-sample, and reproducibility sampling contracts."""

import torch
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.models.non_gp.posterior import make_ensemble_posterior


def _gaussian_posterior():
    train_x = torch.rand(12, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = SingleTaskGP(train_x, train_y)
    candidate = torch.rand(2, 3, 2, dtype=torch.double)
    return model.posterior(candidate)


def test_normal_samplers_support_multidimensional_sample_shape() -> None:
    posterior = _gaussian_posterior()

    for sampler_cls in (SobolQMCNormalSampler, IIDNormalSampler):
        sampler = sampler_cls(torch.Size([4, 5]), seed=123)
        samples = sampler(posterior)

        assert samples.shape == torch.Size([4, 5, 2, 3, 1])
        assert torch.isfinite(samples).all()


def test_normal_sampler_reuses_base_samples_for_common_random_numbers() -> None:
    posterior = _gaussian_posterior()
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=456)

    first = sampler(posterior)
    first_base_samples = sampler.base_samples.clone()
    second = sampler(posterior)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_base_samples, sampler.base_samples)


def test_normal_sampler_seed_controls_base_samples() -> None:
    posterior = _gaussian_posterior()
    first_sampler = IIDNormalSampler(torch.Size([16]), seed=111)
    second_sampler = IIDNormalSampler(torch.Size([16]), seed=222)

    first_sampler(posterior)
    second_sampler(posterior)

    assert not torch.equal(first_sampler.base_samples, second_sampler.base_samples)


def test_index_sampler_supports_multidimensional_sample_shape() -> None:
    values = torch.arange(7 * 2 * 3, dtype=torch.double).reshape(7, 2, 3, 1)
    posterior = make_ensemble_posterior(values)
    sampler = IndexSampler(torch.Size([4, 5]), seed=789)

    samples = sampler(posterior)

    assert samples.shape == torch.Size([4, 5, 2, 3, 1])
    assert torch.isfinite(samples).all()


def test_index_sampler_reuses_base_samples_for_common_random_numbers() -> None:
    values = torch.arange(9 * 3, dtype=torch.double).reshape(9, 3, 1)
    posterior = make_ensemble_posterior(values)
    sampler = IndexSampler(torch.Size([16]), seed=987)

    first = sampler(posterior)
    first_base_samples = sampler.base_samples.clone()
    second = sampler(posterior)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_base_samples, sampler.base_samples)
