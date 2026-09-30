"""Sampling contracts for robust and special posterior models."""

import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import RobustRelevancePursuitSingleTaskGP
from robotorchan.models.expressive.deep_gp_posterior import DeepGPPosterior


def _make_robust_model() -> RobustRelevancePursuitSingleTaskGP:
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)
    return RobustRelevancePursuitSingleTaskGP(train_X=train_x, train_Y=train_y)


def test_robust_gp_uses_standard_gaussian_sampling() -> None:
    model = _make_robust_model()
    model.eval()
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(candidate)

    for sampler_cls in (SobolQMCNormalSampler, IIDNormalSampler):
        samples = sampler_cls(torch.Size([16]), seed=123)(posterior)
        assert samples.shape == torch.Size([16, 2, 1])
        assert torch.isfinite(samples).all()


def test_robust_gp_sampling_preserves_candidate_gradient() -> None:
    model = _make_robust_model()
    model.eval()
    candidate = torch.tensor(
        [[0.25], [0.75]],
        dtype=torch.double,
        requires_grad=True,
    )
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=456)(posterior)
    gradient = torch.autograd.grad(samples.mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def test_deep_gp_posterior_reuses_identical_base_samples() -> None:
    stored = torch.arange(13 * 3, dtype=torch.double).reshape(13, 3, 1)
    posterior = DeepGPPosterior(stored)
    base_samples = torch.randn(16, 3, 1, dtype=torch.double)

    first = posterior.rsample_from_base_samples(torch.Size([16]), base_samples)
    second = posterior.rsample_from_base_samples(torch.Size([16]), base_samples.clone())

    torch.testing.assert_close(first, second)
    assert first.shape == torch.Size([16, 3, 1])


def test_deep_gp_posterior_base_samples_select_whole_q_trajectory() -> None:
    stored = torch.arange(17 * 3, dtype=torch.double).reshape(17, 3, 1)
    posterior = DeepGPPosterior(stored)
    base_samples = torch.randn(32, 3, 1, dtype=torch.double)

    samples = posterior.rsample_from_base_samples(torch.Size([32]), base_samples)

    for sample in samples:
        assert any(torch.equal(sample, trajectory) for trajectory in stored)
