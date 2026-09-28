"""Sampling-contract tests for representative high-dimensional GP models."""

import pytest
import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import ALEBOGP, PCAGP, PLSGP, RandomProjectionGP


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(91)
    train_x = torch.rand(16, 6, dtype=torch.double)
    signal = torch.sin(train_x[:, :1] * 3.0)
    linear = 0.4 * train_x[:, 1:2] - 0.2 * train_x[:, 2:3]
    train_y = signal + linear
    return train_x, train_y


@pytest.mark.parametrize("model_cls", [PCAGP, PLSGP, RandomProjectionGP])
@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_reduced_gp_normal_sampling_contract(model_cls, sampler_cls) -> None:
    train_x, train_y = _make_data()
    model = model_cls(train_X=train_x, train_Y=train_y, n_components=3)
    model.eval()
    candidate = torch.rand(3, 6, dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([16]), seed=123)(posterior)

    assert posterior.mean.shape == torch.Size([3, 1])
    assert samples.shape == torch.Size([16, 3, 1])
    assert samples.dtype == candidate.dtype
    assert samples.device == candidate.device
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("model_cls", [PCAGP, PLSGP, RandomProjectionGP])
def test_reduced_gp_sampling_preserves_raw_input_gradient(model_cls) -> None:
    train_x, train_y = _make_data()
    model = model_cls(train_X=train_x, train_Y=train_y, n_components=3)
    model.eval()
    candidate = torch.rand(2, 6, dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=456)(posterior)
    gradient = torch.autograd.grad(samples.mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("sampler_cls", [SobolQMCNormalSampler, IIDNormalSampler])
def test_alebo_normal_sampling_contract(sampler_cls) -> None:
    train_x, train_y = _make_data()
    train_yvar = torch.full_like(train_y, 1e-4)
    model = ALEBOGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
    model.eval()
    candidate = torch.rand(3, 6, dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = sampler_cls(torch.Size([16]), seed=789)(posterior)

    assert samples.shape == torch.Size([16, 3, 1])
    assert torch.isfinite(samples).all()


def test_alebo_sampling_preserves_candidate_gradient() -> None:
    train_x, train_y = _make_data()
    train_yvar = torch.full_like(train_y, 1e-4)
    model = ALEBOGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
    model.eval()
    candidate = torch.rand(2, 6, dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    samples = IIDNormalSampler(torch.Size([16]), seed=987)(posterior)
    gradient = torch.autograd.grad(samples.mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()
