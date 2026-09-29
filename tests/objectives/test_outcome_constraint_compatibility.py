"""Outcome-constraint contracts for native BoTorch MC acquisitions."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import ModelListGP, SingleTaskGP


def _model() -> tuple[ModelListGP, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    objective_Y = -(train_X - 0.7).square() + 1.0
    upper_constraint_Y = train_X - 0.8
    lower_constraint_Y = 0.2 - train_X
    model = ModelListGP(
        SingleTaskGP(train_X, objective_Y),
        SingleTaskGP(train_X, upper_constraint_Y),
        SingleTaskGP(train_X, lower_constraint_Y),
    )
    return model, train_X


def test_outcome_constraint_nonpositive_values_are_feasible() -> None:
    samples = torch.tensor(
        [
            [[[1.0, -0.1, -0.2], [0.8, 0.2, -0.3]]],
            [[[0.9, -0.4, 0.1], [0.7, -0.2, -0.1]]],
        ],
        dtype=torch.double,
    )
    constraints = [
        lambda Y: Y[..., 1],
        lambda Y: Y[..., 2],
    ]

    feasibility = torch.stack([constraint(samples) <= 0 for constraint in constraints]).all(dim=0)

    expected = torch.tensor(
        [
            [[True, False]],
            [[False, True]],
        ]
    )
    torch.testing.assert_close(feasibility, expected)


def test_constraints_preserve_sample_batch_and_q_axes() -> None:
    samples = torch.randn(5, 7, 2, 3, 3, dtype=torch.double)

    def upper(Y: torch.Tensor) -> torch.Tensor:
        return Y[..., 1]

    def lower(Y: torch.Tensor) -> torch.Tensor:
        return Y[..., 2]

    assert upper(samples).shape == torch.Size([5, 7, 2, 3])
    assert lower(samples).shape == torch.Size([5, 7, 2, 3])


def test_scalar_objective_and_multiple_outcome_constraints_compose_natively() -> None:
    model, train_X = _model()
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=73)
    objective = GenericMCObjective(lambda Y, X=None: Y[..., 0])
    with torch.no_grad():
        baseline_samples = sampler(model.posterior(train_X))
        best_f = objective(baseline_samples).mean(dim=0).max()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=sampler,
        objective=objective,
        constraints=[
            lambda Y: Y[..., 1],
            lambda Y: Y[..., 2],
        ],
    )
    X = torch.tensor([[[0.5], [0.7]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_outcome_constraints_receive_raw_model_samples_not_scalar_objective_values() -> None:
    samples = torch.randn(16, 2, 3, dtype=torch.double)
    objective = GenericMCObjective(lambda Y, X=None: Y[..., 0].square())

    def constraint(Y: torch.Tensor) -> torch.Tensor:
        return Y[..., 2]

    objective_values = objective(samples)
    constraint_values = constraint(samples)

    assert objective_values.shape == torch.Size([16, 2])
    assert constraint_values.shape == torch.Size([16, 2])
    torch.testing.assert_close(constraint_values, samples[..., 2])
