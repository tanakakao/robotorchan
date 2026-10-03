"""E2E contracts for native and explicit-axis robust objective pipelines."""

import torch
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.risk_measures import Expectation as BoTorchExpectation
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.objectives import Expectation
from robotorchan.optim import CandidateConstraints
from robotorchan.optim.backends import optimize_acqf_botorch
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


def test_native_robust_acquisition_with_nonlinear_nominal_candidate_constraint() -> None:
    train_x, train_y = _training_data()
    perturbations = torch.tensor(
        [[0.0, 0.0], [0.03, 0.0], [-0.03, 0.0]],
        dtype=torch.double,
    )
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=InputPerturbation(perturbation_set=perturbations),
    )
    model.eval()
    acquisition = qSimpleRegret(
        model=model,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=151),
        objective=BoTorchExpectation(n_w=perturbations.shape[0]),
    )
    seen_shapes: list[torch.Size] = []

    def nominal_constraint(x: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(x.shape)
        return x.new_tensor(0.7) - x[0]

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((nominal_constraint, True),),
    )
    initial_conditions = torch.tensor(
        [
            [[0.2, 0.3]],
            [[0.4, 0.5]],
            [[0.6, 0.7]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf_botorch(
        acquisition,
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert candidate.shape == torch.Size([1, 2])
    assert torch.isfinite(value).all()
    assert nominal_constraint(candidate[0]) >= -1e-6
    assert seen_shapes
    assert set(seen_shapes) == {torch.Size([2])}


def test_nominal_constraint_does_not_imply_perturbed_scenario_feasibility() -> None:
    perturbations = torch.tensor(
        [[0.0], [0.05], [-0.05]],
        dtype=torch.double,
    )
    transform = InputPerturbation(perturbation_set=perturbations)
    transform.eval()
    nominal = torch.tensor([[0.68]], dtype=torch.double)

    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.7) - x[0]

    scenarios = transform(nominal)

    assert constraint(nominal[0]) >= 0
    assert scenarios.shape == torch.Size([3, 1])
    assert torch.any(torch.stack([constraint(x) for x in scenarios]) < 0)
