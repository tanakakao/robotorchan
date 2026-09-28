"""Sampling-contract tests for the standard single-task GP."""

import pytest
import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
@pytest.mark.parametrize("q", [1, 3])
def test_single_task_gp_normal_sampler_shape_dtype_and_finite(sampler_cls, q: int) -> None:
    train_x = torch.rand(10, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = SingleTaskGP(train_x, train_y)

    candidate = torch.rand(2, q, 2, dtype=torch.double)
    posterior = model.posterior(candidate)
    sampler = sampler_cls(sample_shape=torch.Size([16]), seed=123)
    samples = sampler(posterior)

    assert samples.shape == torch.Size([16, 2, q, 1])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_single_task_gp_normal_sampler_is_reproducible(sampler_cls) -> None:
    train_x = torch.rand(10, 2, dtype=torch.double)
    train_y = torch.cos(train_x[:, :1] * 2.0)
    model = SingleTaskGP(train_x, train_y)
    posterior = model.posterior(torch.rand(3, 2, dtype=torch.double))

    first = sampler_cls(sample_shape=torch.Size([8]), seed=321)(posterior)
    second = sampler_cls(sample_shape=torch.Size([8]), seed=321)(posterior)

    assert torch.equal(first, second)


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_single_task_gp_normal_sampler_preserves_candidate_gradient(sampler_cls) -> None:
    train_x = torch.rand(10, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 4.0)
    model = SingleTaskGP(train_x, train_y)

    candidate = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = sampler_cls(sample_shape=torch.Size([8]), seed=456)
    loss = sampler(posterior).mean()
    gradient = torch.autograd.grad(loss, candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()
