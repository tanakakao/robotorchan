"""Sampling-contract tests for the standard long-format multi-task GP."""

import pytest
import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models.standard.multitask import MultiTaskGP


def _make_model() -> MultiTaskGP:
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    task_0 = torch.zeros_like(x)
    task_1 = torch.ones_like(x)
    train_x = torch.cat(
        [
            torch.cat([x, task_0], dim=-1),
            torch.cat([x, task_1], dim=-1),
        ],
        dim=0,
    )
    train_y = torch.cat(
        [
            torch.sin(x * 3.0),
            torch.sin(x * 3.0) + 0.5,
        ],
        dim=0,
    )
    return MultiTaskGP(train_X=train_x, train_Y=train_y, task_feature=-1, rank=1)


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_multi_task_gp_normal_sampler_preserves_joint_output_shape(sampler_cls) -> None:
    model = _make_model()
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(candidate)
    sampler = sampler_cls(torch.Size([32]), seed=123)

    samples = sampler(posterior)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([32, 2, 2])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_multi_task_gp_normal_sampler_preserves_candidate_gradient(sampler_cls) -> None:
    model = _make_model()
    candidate = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = sampler_cls(torch.Size([16]), seed=456)

    loss = sampler(posterior).mean()
    gradient = torch.autograd.grad(loss, candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def test_multi_task_gp_iid_samples_preserve_posterior_task_covariance() -> None:
    torch.manual_seed(17)
    model = _make_model()
    candidate = torch.tensor([[0.5]], dtype=torch.double)
    posterior = model.posterior(candidate)
    covariance = posterior.distribution.covariance_matrix
    sampler = IIDNormalSampler(torch.Size([8192]), seed=789)

    samples = sampler(posterior).squeeze(-2)
    empirical_covariance = torch.cov(samples.transpose(0, 1))

    torch.testing.assert_close(
        empirical_covariance,
        covariance,
        rtol=0.12,
        atol=0.01,
    )
