"""BoTorch Objective and PosteriorTransform pass-through contracts."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import (
    GenericMCObjective,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


def _two_output_model() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat((torch.sin(3.0 * train_x), torch.cos(3.0 * train_x)), dim=-1)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=2))
    return model, train_x, train_y


def test_botorch_posterior_transform_instance_passes_through_unchanged() -> None:
    model, _, _ = _two_output_model()
    transform = ScalarizedPosteriorTransform(
        weights=torch.tensor([0.7, 0.3], dtype=torch.double),
    )
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    transformed = model.posterior(X, posterior_transform=transform)
    raw = model.posterior(X)
    expected = transform(raw)

    torch.testing.assert_close(transformed.mean, expected.mean)
    torch.testing.assert_close(transformed.variance, expected.variance)
    assert transformed.mean.shape == torch.Size([2, 1])


def test_botorch_objective_instance_is_retained_by_acquisition() -> None:
    model, _, train_y = _two_output_model()
    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = LinearMCObjective(weights=weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=181),
        objective=objective,
    )

    assert acquisition.objective is objective
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_custom_generic_mc_objective_requires_no_robotorchan_registration() -> None:
    model, _, _ = _two_output_model()

    def custom_objective(Y: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        del X
        return Y[..., 0].square() - 0.25 * Y[..., 1].square()

    objective = GenericMCObjective(custom_objective)
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=191)
    with torch.no_grad():
        baseline_samples = sampler(
            model.posterior(
                torch.linspace(
                    0.0,
                    1.0,
                    10,
                    dtype=torch.double,
                ).unsqueeze(-1)
            )
        )
        best_f = objective(baseline_samples).mean(dim=0).max()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=sampler,
        objective=objective,
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    gradient = torch.autograd.grad(value.sum(), X)[0]

    assert acquisition.objective is objective
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()


def test_botorch_utility_accepts_robotorchan_model_without_conversion() -> None:
    model, train_x, _ = _two_output_model()

    cost = get_infeasible_cost(X=train_x, model=model)

    assert cost.shape == torch.Size([2])
    assert torch.isfinite(cost).all()
