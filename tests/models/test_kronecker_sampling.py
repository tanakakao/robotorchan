"""Sampling-contract tests for Kronecker multi-task models."""

import pytest
import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models.standard.multitask import KroneckerMultiTaskGP
from robotorchan.models.structured.latent_kronecker import LatentKroneckerGP


def _make_kronecker_model() -> KroneckerMultiTaskGP:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            0.7 * torch.sin(train_x * 3.0) + 0.3 * train_x,
        ],
        dim=-1,
    )
    return KroneckerMultiTaskGP(train_X=train_x, train_Y=train_y, rank=1)


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_kronecker_normal_sampler_preserves_joint_output_shape(sampler_cls) -> None:
    model = _make_kronecker_model()
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([32]), seed=123)(posterior)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([32, 2, 2])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


def test_kronecker_iid_samples_preserve_joint_covariance() -> None:
    torch.manual_seed(17)
    model = _make_kronecker_model()
    candidate = torch.tensor([[0.5]], dtype=torch.double)
    posterior = model.posterior(candidate)
    covariance = posterior.distribution.covariance_matrix
    samples = IIDNormalSampler(torch.Size([8192]), seed=789)(posterior).squeeze(-2)
    empirical_covariance = torch.cov(samples.transpose(0, 1))

    torch.testing.assert_close(empirical_covariance, covariance, rtol=0.12, atol=0.01)


def test_kronecker_normal_sampler_preserves_candidate_gradient() -> None:
    model = _make_kronecker_model()
    candidate = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=456)(posterior)
    gradient = torch.autograd.grad(samples.mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def _make_latent_model() -> LatentKroneckerGP:
    train_x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    train_t = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.sin(train_x * 2.0) + train_t.squeeze(-1)
    return LatentKroneckerGP(train_X=train_x, train_T=train_t, train_Y=train_y)


def test_latent_kronecker_native_posterior_rsample_shape_and_finite() -> None:
    model = _make_latent_model()
    test_x = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    test_t = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    with model.use_iterative_methods():
        posterior = model.posterior(test_x, test_t)
        samples = posterior.rsample(torch.Size([16]))

    assert samples.shape == torch.Size([16, 2, 2])
    assert samples.dtype == test_x.dtype
    assert samples.device == test_x.device
    assert torch.isfinite(samples).all()


def test_latent_kronecker_native_sampling_preserves_candidate_gradient() -> None:
    model = _make_latent_model()
    test_x = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    test_t = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    with model.use_iterative_methods():
        posterior = model.posterior(test_x, test_t)
        samples = posterior.rsample(torch.Size([8]))
        gradient = torch.autograd.grad(samples.mean(), test_x)[0]

    assert gradient.shape == test_x.shape
    assert torch.isfinite(gradient).all()
