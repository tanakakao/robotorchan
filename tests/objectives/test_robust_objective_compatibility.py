"""E2E contracts for native and explicit-axis robust objective pipelines."""

import torch
from botorch.acquisition.risk_measures import Expectation as BoTorchExpectation
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.objectives import Expectation
from robotorchan.uncertainty import GaussianPerturbation


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.rand(16, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0) - 0.2 * train_x[:, 1:2]
    return train_x, train_y


def test_botorch_input_perturbation_flattens_q_and_scenario_axes() -> None:
    perturbations = torch.tensor(
        [[0.0, 0.0], [0.01, -0.02], [-0.01, 0.02]],
        dtype=torch.double,
    )
    transform = InputPerturbation(perturbation_set=perturbations)
    transform.eval()
    X = torch.rand(2, 4, 2, dtype=torch.double)

    transformed = transform(X)

    assert transformed.shape == torch.Size([2, 12, 2])


def test_robotorchan_scenario_generator_keeps_scenario_axis_explicit() -> None:
    X = torch.rand(2, 4, 2, dtype=torch.double)
    scenarios = GaussianPerturbation(std=0.03).sample(X, n_w=3)

    assert scenarios.shape == torch.Size([2, 4, 3, 2])


def test_native_botorch_robust_pipeline_preserves_candidate_axis_and_gradient() -> None:
    train_x, train_y = _training_data()
    perturbations = torch.tensor(
        [[0.0, 0.0], [0.01, -0.02], [-0.01, 0.02]],
        dtype=torch.double,
    )
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=InputPerturbation(perturbation_set=perturbations),
    )
    model.eval()
    candidate = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=91)
    samples = sampler(posterior)
    objective = BoTorchExpectation(n_w=perturbations.shape[0])

    robust_samples = objective(samples)
    value = robust_samples.mean()
    gradient = torch.autograd.grad(value, candidate)[0]

    assert posterior.mean.shape == torch.Size([6, 1])
    assert robust_samples.shape == torch.Size([16, 2])
    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def test_explicit_axis_pipeline_preserves_candidate_axis_and_gradient() -> None:
    train_x, train_y = _training_data()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    candidate = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
    scenarios = GaussianPerturbation(std=0.03).sample(candidate, n_w=3)
    posterior = model.posterior(scenarios)
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=91)
    samples = sampler(posterior).squeeze(-1)

    robust_samples = Expectation()(samples)
    value = robust_samples.mean()
    gradient = torch.autograd.grad(value, candidate)[0]

    assert posterior.mean.shape == torch.Size([2, 3, 1])
    assert robust_samples.shape == torch.Size([16, 2])
    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()
