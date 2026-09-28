"""Sampling-contract tests for representative mixed GP models."""

import pytest
import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import MixedMultiTaskGP, MixedSingleTaskGP


def _make_single_task_model() -> MixedSingleTaskGP:
    continuous = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    categorical = torch.arange(12, dtype=torch.double).remainder(3).unsqueeze(-1)
    train_x = torch.cat([continuous, categorical], dim=-1)
    train_y = torch.sin(continuous * 3.0) + 0.2 * categorical
    return MixedSingleTaskGP(train_X=train_x, train_Y=train_y, cat_dims=[-1])


def _make_multi_task_model() -> MixedMultiTaskGP:
    continuous = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    categorical = torch.arange(8, dtype=torch.double).remainder(3).unsqueeze(-1)
    task_0 = torch.zeros_like(continuous)
    task_1 = torch.ones_like(continuous)
    train_x = torch.cat(
        [
            torch.cat([continuous, categorical, task_0], dim=-1),
            torch.cat([continuous, categorical, task_1], dim=-1),
        ],
        dim=0,
    )
    train_y = torch.cat(
        [
            torch.sin(continuous * 3.0) + 0.2 * categorical,
            torch.sin(continuous * 3.0) + 0.2 * categorical + 0.4,
        ],
        dim=0,
    )
    return MixedMultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        cat_dims=[1],
    )


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_mixed_single_task_normal_sampling_contract(sampler_cls) -> None:
    model = _make_single_task_model()
    model.eval()
    candidate = torch.tensor([[0.25, 0.0], [0.75, 2.0]], dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([32]), seed=123)(posterior)

    assert samples.shape == torch.Size([32, 2, 1])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_mixed_multi_task_normal_sampling_contract(sampler_cls) -> None:
    model = _make_multi_task_model()
    model.eval()
    candidate = torch.tensor([[0.25, 0.0], [0.75, 2.0]], dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([32]), seed=456)(posterior)

    assert samples.shape == torch.Size([32, 2, 2])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


def test_mixed_single_task_sampling_preserves_continuous_gradient() -> None:
    model = _make_single_task_model()
    model.eval()
    continuous = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    categorical = torch.tensor([[0.0], [2.0]], dtype=torch.double)
    candidate = torch.cat([continuous, categorical], dim=-1)
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=789)(posterior)
    gradient = torch.autograd.grad(samples.mean(), continuous)[0]

    assert gradient.shape == continuous.shape
    assert torch.isfinite(gradient).all()


def test_mixed_multi_task_sampling_preserves_continuous_gradient() -> None:
    model = _make_multi_task_model()
    model.eval()
    continuous = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    categorical = torch.tensor([[0.0], [2.0]], dtype=torch.double)
    candidate = torch.cat([continuous, categorical], dim=-1)
    posterior = model.posterior(candidate)
    samples = IIDNormalSampler(torch.Size([16]), seed=987)(posterior)
    gradient = torch.autograd.grad(samples.mean(), continuous)[0]

    assert gradient.shape == continuous.shape
    assert torch.isfinite(gradient).all()
