"""Objective-to-Acquisition integration contracts for native BoTorch acquisitions."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.risk_measures import Expectation
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import ModelListGP, SingleTaskGP


def _two_output_model() -> tuple[ModelListGP, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(
        SingleTaskGP(train_x, torch.sin(3.0 * train_x)),
        SingleTaskGP(train_x, torch.cos(3.0 * train_x)),
    )
    return model, train_x


def _scalar_objective() -> GenericMCObjective:
    return GenericMCObjective(lambda Y, X=None: 0.7 * Y[..., 0] + 0.3 * Y[..., 1])


def test_scalar_objective_connects_to_improvement_acquisition() -> None:
    model, train_x = _two_output_model()
    objective = _scalar_objective()
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=131)
    with torch.no_grad():
        baseline = objective(sampler(model.posterior(train_x))).mean(dim=0)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=baseline.max(),
        sampler=sampler,
        objective=objective,
    )
    X = torch.tensor([[[0.35], [0.65]]], dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    gradient = torch.autograd.grad(value.sum(), X)[0]

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()


def test_scalar_objective_connects_to_noisy_improvement_acquisition() -> None:
    model, train_x = _two_output_model()
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=137),
        objective=_scalar_objective(),
        prune_baseline=False,
    )
    X = torch.tensor([[[0.45]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_scalar_objective_connects_to_ucb_acquisition() -> None:
    model, _ = _two_output_model()
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=139),
        objective=_scalar_objective(),
    )
    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_objective_and_outcome_constraints_connect_to_acquisition_together() -> None:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(
        SingleTaskGP(train_x, -(train_x - 0.7).square() + 1.0),
        SingleTaskGP(train_x, train_x - 0.8),
    )
    objective = GenericMCObjective(lambda Y, X=None: Y[..., 0])
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=149)
    with torch.no_grad():
        best_f = objective(sampler(model.posterior(train_x))).mean(dim=0).max()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=sampler,
        objective=objective,
        constraints=[lambda Y: Y[..., 1]],
    )
    X = torch.tensor([[[0.5], [0.7]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_native_risk_objective_connects_to_noisy_improvement_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(3.0 * train_x)
    perturbations = torch.tensor([[0.0], [0.01], [-0.01]], dtype=torch.double)
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=InputPerturbation(perturbation_set=perturbations),
    )
    model.eval()
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=151),
        objective=Expectation(n_w=perturbations.shape[0]),
        prune_baseline=False,
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    gradient = torch.autograd.grad(value.sum(), X)[0]

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
