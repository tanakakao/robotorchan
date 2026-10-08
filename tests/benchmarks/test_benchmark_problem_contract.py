"""Benchmark problem contract and truth-based regret tests."""

import pytest
import torch

from robotorchan.benchmarks.problem import BenchmarkProblem


def _problem(**kwargs: object) -> BenchmarkProblem:
    defaults = dict(
        name="quadratic",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=lambda x: -(x - 0.7).square(),
        directions=("maximize",),
        variable_types=("continuous",),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )
    defaults.update(kwargs)
    return BenchmarkProblem(**defaults)


def test_truth_observation_and_regret_are_separate() -> None:
    problem = _problem(observe=lambda x: -(x - 0.7).square() + 10)
    X = torch.tensor([[0.2], [0.7]], dtype=torch.double)
    expected = torch.tensor([[-0.25], [0.0]], dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_truth(X), expected)
    assert torch.all(problem.evaluate_observation(X) > problem.evaluate_truth(X))
    torch.testing.assert_close(problem.simple_regret(X), torch.tensor(0.0, dtype=torch.double))


def test_minimization_regret() -> None:
    problem = _problem(
        objective=lambda x: (x - 0.7).square(),
        directions=("minimize",),
    )
    X = torch.tensor([[0.2]], dtype=torch.double)
    torch.testing.assert_close(problem.simple_regret(X), torch.tensor(0.25, dtype=torch.double))


def test_constraint_and_cost_contract() -> None:
    problem = _problem(
        constraints=lambda x: x - 0.5,
        n_constraints=1,
        cost=lambda x: x + 1,
    )
    X = torch.tensor([[0.2], [0.7]], dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_constraints(X), X - 0.5)
    torch.testing.assert_close(problem.evaluate_cost(X), X + 1)
    torch.testing.assert_close(problem.simple_regret(X), torch.tensor(0.0, dtype=torch.double))
    assert torch.isinf(problem.simple_regret(X[:1]))


def test_invalid_contracts_and_inputs() -> None:
    with pytest.raises(ValueError, match="variable_types"):
        _problem(variable_types=())
    with pytest.raises(ValueError, match="constraints"):
        _problem(n_constraints=1)
    with pytest.raises(ValueError, match="within bounds"):
        _problem().evaluate_truth(torch.tensor([[1.1]]))
    with pytest.raises(ValueError, match="Discrete"):
        _problem(variable_types=("integer",)).evaluate_truth(torch.tensor([[0.5]]))
    with pytest.raises(ValueError, match="objective"):
        _problem(objective=lambda x: x.squeeze(-1)).evaluate_truth(torch.tensor([[0.5]]))


def test_batch_and_multiobjective() -> None:
    problem = _problem(
        objective=lambda x: torch.cat((x, 1 - x), dim=-1),
        directions=("maximize", "minimize"),
        optimal_value=None,
        reference_point=torch.tensor([0.0, 1.0]),
    )
    X = torch.rand(2, 3, 1, dtype=torch.double)
    assert problem.evaluate_truth(X).shape == (2, 3, 2)
    assert problem.to_maximization(problem.evaluate_truth(X)).shape == (2, 3, 2)
    with pytest.raises(ValueError, match="single-objective"):
        problem.simple_regret(X)
