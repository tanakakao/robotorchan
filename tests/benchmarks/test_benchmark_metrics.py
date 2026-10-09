"""Metric correctness, feasibility, and objective direction tests."""

import pytest
import torch

from robotorchan.benchmarks.heterogeneous_problems import strength_conductivity_pass
from robotorchan.benchmarks.metrics import (
    cumulative_feasibility_rate,
    feasibility_rate,
    hypervolume_2d,
    hypervolume_curve,
    simple_regret_curve,
)
from robotorchan.benchmarks.runner import BenchmarkTrajectory
from robotorchan.benchmarks.standard_problems import sphere3


def _trajectory(problem, X):
    return BenchmarkTrajectory(
        seed=0,
        X=X,
        Y_observed=problem.evaluate_truth(X),
        Y_truth=problem.evaluate_truth(X),
        constraints=problem.evaluate_constraints(X),
        costs=problem.evaluate_cost(X),
        initial_points=1,
    )


def test_feasibility_rates() -> None:
    constraints = torch.tensor([[1.0], [-1.0], [0.0], [2.0]])
    torch.testing.assert_close(feasibility_rate(constraints), torch.tensor(0.75))
    torch.testing.assert_close(
        cumulative_feasibility_rate(constraints),
        torch.tensor([1.0, 0.5, 2.0 / 3.0, 0.75]),
    )
    assert feasibility_rate(torch.empty(3, 0)) == 1


def test_simple_regret_curve_is_monotone() -> None:
    problem = sphere3()
    X = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    trajectory = _trajectory(problem, X)
    torch.testing.assert_close(
        simple_regret_curve(problem, trajectory), torch.tensor([4.0, 1.0, 0.0])
    )


def test_hypervolume_2d_exact_union() -> None:
    values = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 2.0], [0.5, 0.5]])
    torch.testing.assert_close(hypervolume_2d(values, torch.zeros(2)), torch.tensor(3.0))
    assert hypervolume_2d(values, torch.tensor([3.0, 3.0])) == 0
    assert hypervolume_2d(values[:0], torch.zeros(2)) == 0


def test_hypervolume_curve_excludes_infeasible_points() -> None:
    problem = strength_conductivity_pass()
    X = torch.tensor([[0.2, 0.7], [0.8, 0.3], [0.5, 0.8]], dtype=torch.double)
    trajectory = _trajectory(problem, X)
    curve = hypervolume_curve(problem, trajectory)
    assert curve.shape == (3,)
    assert curve[0] > 0
    torch.testing.assert_close(curve[1], curve[0])
    assert curve[2] >= curve[1]


@pytest.mark.parametrize("values", [torch.ones(2), torch.ones(2, 3)])
def test_hypervolume_rejects_invalid_shape(values) -> None:
    with pytest.raises(ValueError, match="shape"):
        hypervolume_2d(values, torch.zeros(2))
