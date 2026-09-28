"""Sampling-contract tests for standard multi-fidelity GP models."""

import pytest
import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import SingleTaskMultiFidelityGP


def _make_model() -> SingleTaskMultiFidelityGP:
    design = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    fidelity = torch.linspace(0.2, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat([design, fidelity], dim=-1)
    train_y = torch.sin(design * 3.0) + 0.3 * (1.0 - fidelity)
    return SingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_y,
        data_fidelities=[-1],
    )


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_multi_fidelity_normal_sampling_contract(sampler_cls) -> None:
    model = _make_model()
    model.eval()
    candidate = torch.tensor(
        [[0.25, 1.0], [0.50, 0.6], [0.75, 0.3]],
        dtype=torch.double,
    )
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([32]), seed=123)(posterior)

    assert samples.shape == torch.Size([32, 3, 1])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_multi_fidelity_sampler_preserves_t_batch_shape(sampler_cls) -> None:
    model = _make_model()
    model.eval()
    candidate = torch.tensor(
        [
            [[0.20, 1.0], [0.40, 0.8]],
            [[0.60, 0.6], [0.80, 0.4]],
        ],
        dtype=torch.double,
    )
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([16]), seed=456)(posterior)

    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert samples.shape == torch.Size([16, 2, 2, 1])
    assert torch.isfinite(samples).all()


def test_multi_fidelity_sampling_preserves_design_gradient() -> None:
    model = _make_model()
    model.eval()
    design = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    fidelity = torch.ones(2, 1, dtype=torch.double)
    candidate = torch.cat([design, fidelity], dim=-1)
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=789)(posterior)
    gradient = torch.autograd.grad(samples.mean(), design)[0]

    assert gradient.shape == design.shape
    assert torch.isfinite(gradient).all()


def test_multi_fidelity_sampling_preserves_fidelity_gradient() -> None:
    model = _make_model()
    model.eval()
    design = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    fidelity = torch.tensor([[0.4], [0.8]], dtype=torch.double, requires_grad=True)
    candidate = torch.cat([design, fidelity], dim=-1)
    posterior = model.posterior(candidate)
    samples = IIDNormalSampler(torch.Size([16]), seed=987)(posterior)
    gradient = torch.autograd.grad(samples.mean(), fidelity)[0]

    assert gradient.shape == fidelity.shape
    assert torch.isfinite(gradient).all()
