"""Gradient, dtype, and device contracts for Objective and robust pipelines."""

import pytest
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.risk_measures import Expectation as BoTorchExpectation
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.objectives import Expectation
from robotorchan.uncertainty import GaussianPerturbation


def _model(dtype: torch.dtype, device: torch.device) -> SingleTaskGP:
    train_x = torch.rand(14, 2, dtype=dtype, device=device)
    train_y = torch.sin(train_x[:, :1] * 3.0) - 0.2 * train_x[:, 1:2]
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mc_objective_preserves_dtype_and_candidate_gradient(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    model = _model(dtype, device)
    candidate = torch.rand(2, 2, dtype=dtype, device=device, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = SobolQMCNormalSampler(torch.Size([8]), seed=17)
    samples = sampler(posterior)
    objective = GenericMCObjective(lambda Y, X=None: Y.squeeze(-1).square())

    values = objective(samples, X=candidate)
    gradient = torch.autograd.grad(values.mean(), candidate)[0]

    assert values.dtype == dtype
    assert values.device == device
    assert gradient.dtype == dtype
    assert gradient.device == device
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_robust_path_preserves_dtype_device_and_gradient(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    model = _model(dtype, device)
    candidate = torch.rand(2, 2, dtype=dtype, device=device, requires_grad=True)
    scenarios = GaussianPerturbation(std=0.03).sample(candidate, n_w=4)
    posterior = model.posterior(scenarios)
    sampler = SobolQMCNormalSampler(torch.Size([8]), seed=19)
    samples = sampler(posterior).squeeze(-1)

    values = Expectation()(samples)
    gradient = torch.autograd.grad(values.mean(), candidate)[0]

    assert scenarios.dtype == dtype
    assert scenarios.device == device
    assert values.dtype == dtype
    assert values.device == device
    assert gradient.dtype == dtype
    assert gradient.device == device
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_robust_path_preserves_dtype_device_and_gradient(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    train_x = torch.rand(14, 2, dtype=dtype, device=device)
    train_y = torch.sin(train_x[:, :1] * 3.0) - 0.2 * train_x[:, 1:2]
    perturbations = torch.tensor(
        [[0.0, 0.0], [0.01, -0.02], [-0.01, 0.02]],
        dtype=dtype,
        device=device,
    )
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=InputPerturbation(perturbation_set=perturbations),
    )
    model.eval()
    candidate = torch.rand(2, 2, dtype=dtype, device=device, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = SobolQMCNormalSampler(torch.Size([8]), seed=23)
    samples = sampler(posterior)

    values = BoTorchExpectation(n_w=perturbations.shape[0])(samples)
    gradient = torch.autograd.grad(values.mean(), candidate)[0]

    assert values.dtype == dtype
    assert values.device == device
    assert gradient.dtype == dtype
    assert gradient.device == device
    assert torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_objective_and_explicit_robust_path_preserve_cuda_device() -> None:
    device = torch.device("cuda")
    dtype = torch.float64
    model = _model(dtype, device)
    candidate = torch.rand(2, 2, dtype=dtype, device=device, requires_grad=True)
    scenarios = GaussianPerturbation(std=0.03).sample(candidate, n_w=4)
    posterior = model.posterior(scenarios)
    sampler = SobolQMCNormalSampler(torch.Size([8]), seed=29)
    samples = sampler(posterior).squeeze(-1)

    values = Expectation()(samples)
    gradient = torch.autograd.grad(values.mean(), candidate)[0]

    assert scenarios.device.type == "cuda"
    assert values.device.type == "cuda"
    assert gradient.device.type == "cuda"
