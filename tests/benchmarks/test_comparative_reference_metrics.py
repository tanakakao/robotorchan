"""Independent reference cases for comparative benchmark metric validation."""

import pytest
import torch

from robotorchan.benchmarks.metrics import (
    cumulative_feasibility_rate,
    hypervolume_2d,
    hypervolume_curve,
    simple_regret_curve,
)
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import BenchmarkTrajectory


def _trajectory(problem: BenchmarkProblem, X: torch.Tensor) -> BenchmarkTrajectory:
    return BenchmarkTrajectory(
        seed=0,
        X=X,
        Y_observed=problem.evaluate_truth(X),
        Y_truth=problem.evaluate_truth(X),
        constraints=problem.evaluate_constraints(X),
        costs=problem.evaluate_cost(X),
        initial_points=2,
    )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("minimize", [4.0, 1.0, 0.0]),
        ("maximize", [3.0, 3.0, 0.0]),
    ],
)
def test_regret_respects_direction(direction: str, expected: list[float]) -> None:
    def objective(X: torch.Tensor) -> torch.Tensor:
        return X[..., :1].square()

    optimum = 0.0 if direction == "minimize" else 4.0
    problem = BenchmarkProblem(
        name="reference_regret",
        bounds=torch.tensor([[0.0], [2.0]], dtype=torch.double),
        objective=objective,
        directions=(direction,),
        variable_types=("continuous",),
        optimal_value=torch.tensor([optimum], dtype=torch.double),
    )
    X = torch.tensor(
        [[2.0], [1.0], [0.0]] if direction == "minimize" else [[1.0], [0.0], [2.0]],
        dtype=torch.double,
    )
    actual = simple_regret_curve(problem, _trajectory(problem, X))
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.double))


def test_regret_uses_truth_not_observed_values() -> None:
    def objective(X: torch.Tensor) -> torch.Tensor:
        return X[..., :1]

    problem = BenchmarkProblem(
        name="truth_regret",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=objective,
        directions=("maximize",),
        variable_types=("continuous",),
        optimal_value=torch.tensor([1.0], dtype=torch.double),
    )
    X = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    trajectory = _trajectory(problem, X)
    observed = torch.full_like(trajectory.Y_observed, 100.0)
    altered = BenchmarkTrajectory(
        seed=trajectory.seed,
        X=trajectory.X,
        Y_observed=observed,
        Y_truth=trajectory.Y_truth,
        constraints=trajectory.constraints,
        costs=trajectory.costs,
        initial_points=trajectory.initial_points,
    )
    torch.testing.assert_close(
        simple_regret_curve(problem, altered),
        torch.tensor([0.8, 0.5, 0.2], dtype=torch.double),
    )


def test_regret_is_infinite_until_first_feasible_point() -> None:
    def objective(X: torch.Tensor) -> torch.Tensor:
        return X[..., :1]

    def constraints(X: torch.Tensor) -> torch.Tensor:
        return X[..., :1] - 0.5

    problem = BenchmarkProblem(
        name="feasible_regret",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=objective,
        directions=("maximize",),
        variable_types=("continuous",),
        constraints=constraints,
        n_constraints=1,
        optimal_value=torch.tensor([1.0], dtype=torch.double),
    )
    X = torch.tensor([[0.1], [0.3], [0.5], [0.9]], dtype=torch.double)
    trajectory = _trajectory(problem, X)
    regret = simple_regret_curve(problem, trajectory)
    assert torch.isinf(regret[:2]).all()
    torch.testing.assert_close(regret[2:], torch.tensor([0.5, 0.1], dtype=torch.double))
    torch.testing.assert_close(
        cumulative_feasibility_rate(trajectory.constraints),
        torch.tensor([0.0, 0.0, 1.0 / 3.0, 0.5], dtype=torch.double),
    )


def test_hypervolume_2d_matches_independent_rectangle_union() -> None:
    points = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]], dtype=torch.double)
    reference = torch.zeros(2, dtype=torch.double)
    # Two 2x1 and 1x2 rectangles overlap on a 1x1 square.
    expected = torch.tensor(2.0 + 2.0 - 1.0, dtype=torch.double)
    torch.testing.assert_close(hypervolume_2d(points, reference), expected)


def test_hypervolume_curve_uses_truth_and_feasibility() -> None:
    def objective(X: torch.Tensor) -> torch.Tensor:
        return X

    def constraints(X: torch.Tensor) -> torch.Tensor:
        return X[..., :1] - 0.5

    problem = BenchmarkProblem(
        name="reference_hv",
        bounds=torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.double),
        objective=objective,
        directions=("maximize", "maximize"),
        variable_types=("continuous", "continuous"),
        constraints=constraints,
        n_constraints=1,
        reference_point=torch.zeros(2, dtype=torch.double),
    )
    X = torch.tensor([[0.2, 2.0], [1.0, 1.0], [2.0, 0.5]], dtype=torch.double)
    trajectory = _trajectory(problem, X)
    altered = BenchmarkTrajectory(
        seed=trajectory.seed,
        X=trajectory.X,
        Y_observed=torch.full_like(trajectory.Y_observed, 100.0),
        Y_truth=trajectory.Y_truth,
        constraints=trajectory.constraints,
        costs=trajectory.costs,
        initial_points=trajectory.initial_points,
    )
    curve = hypervolume_curve(problem, altered)
    torch.testing.assert_close(curve, torch.tensor([0.0, 1.0, 1.5], dtype=torch.double))


def test_three_objective_hypervolume_is_not_silently_accepted() -> None:
    def objective(X: torch.Tensor) -> torch.Tensor:
        return X

    problem = BenchmarkProblem(
        name="reference_hv3",
        bounds=torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double),
        objective=objective,
        directions=("maximize",) * 3,
        variable_types=("continuous",) * 3,
        reference_point=torch.zeros(3, dtype=torch.double),
    )
    X = torch.tensor([[0.5] * 3, [1.0] * 3], dtype=torch.double)
    with pytest.raises(ValueError, match="two objectives"):
        hypervolume_curve(problem, _trajectory(problem, X))
