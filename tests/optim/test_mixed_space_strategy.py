"""Tests for mixed-variable acquisition optimization."""

import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models.standard.single_task import MixedSingleTaskGP
from robotorchan.optim import CandidateConstraints, MixedSpaceStrategy


def test_mixed_space_strategy_respects_category_and_linear_constraint() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.0], [0.7, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2 * train_X[:, 1:2]).sin()
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = PosteriorMean(model)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([-1.0], dtype=torch.double),
                -0.6,
            ),
        ),
    )
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=16,
        constraints=constraints,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert result.candidates[0, 0] <= 0.6 + 1e-6
    assert result.candidates[0, 1].item() in {0.0, 1.0}


def test_mixed_space_strategy_respects_intrapoint_nonlinear_constraint() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.0], [0.7, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2 * train_X[:, 1:2]).sin()
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = PosteriorMean(model)

    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.36) - x[0].square()

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((constraint, True),),
    )
    initial_conditions = torch.tensor(
        [[[0.1, 0.0]], [[0.2, 0.0]], [[0.3, 0.0]], [[0.4, 0.0]]],
        dtype=torch.double,
    )
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=4,
        raw_samples=32,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert result.candidates[0, 1].item() in {0.0, 1.0}
    assert constraint(result.candidates[0]) >= -1e-6


def test_mixed_space_strategy_rejects_interpoint_nonlinear_constraint() -> None:
    def constraint(X: torch.Tensor) -> torch.Tensor:
        return (X[0] - X[-1]).square().sum() - X.new_tensor(0.25)

    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        constraints=CandidateConstraints(
            nonlinear_inequality_constraints=((constraint, False),),
        ),
        batch_initial_conditions=torch.tensor([[[0.1, 0.0]]], dtype=torch.double),
    )

    try:
        strategy.optimize(None, q=2)  # type: ignore[arg-type]
    except ValueError as error:
        assert str(error) == (
            "BoTorch mixed acquisition optimization does not support inter-point nonlinear constraints."
        )
    else:
        raise AssertionError("Expected inter-point nonlinear constraints to be rejected.")


def test_mixed_space_strategy_requires_initial_conditions_for_nonlinear_constraint() -> None:
    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.36) - x[0].square()

    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        constraints=CandidateConstraints(
            nonlinear_inequality_constraints=((constraint, True),),
        ),
    )

    try:
        strategy.optimize(None)  # type: ignore[arg-type]
    except ValueError as error:
        assert str(error) == (
            "Nonlinear candidate constraints require feasible batch_initial_conditions."
        )
    else:
        raise AssertionError("Expected nonlinear constraints without initial conditions to fail.")


def test_mixed_nonlinear_constraint_forwards_batch_limit(monkeypatch) -> None:
    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x[0]

    initial_conditions = torch.tensor([[[0.5, 0.0]]], dtype=torch.double)
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        constraints=CandidateConstraints(
            nonlinear_inequality_constraints=((constraint, True),),
        ),
        batch_initial_conditions=initial_conditions,
    )
    captured = {}

    def fake_optimize_acqf_mixed(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 2, dtype=torch.double), torch.tensor(0.0, dtype=torch.double)

    monkeypatch.setattr(\n        "robotorchan.optim.backends.botorch.botorch_optimize_acqf_mixed",\n        fake_optimize_acqf_mixed,\n    )
    strategy.optimize(None)  # type: ignore[arg-type]

    assert captured["options"]["batch_limit"] == 1
    assert captured["batch_initial_conditions"] is initial_conditions
