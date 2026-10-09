"""Validate analytic multiobjective benchmarks and Pareto front metadata."""

import pytest
import torch

from robotorchan.benchmarks.multiobjective_problems import (
    biobjective_linear,
    branin_currin,
    dtlz2,
    register_multiobjective_problems,
    zdt1,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


@pytest.mark.parametrize("factory", [biobjective_linear, branin_currin, dtlz2, zdt1])
def test_multiobjective_problem_contract(factory) -> None:
    problem = factory()
    X = problem.bounds.mean(dim=0).unsqueeze(0)
    Y = problem.evaluate_truth(X)
    assert Y.shape == (1, problem.n_objectives)
    assert problem.reference_front.shape[-1] == problem.n_objectives
    assert torch.isfinite(problem.reference_front).all()
    assert torch.isfinite(problem.reference_point).all()
    assert problem.optimal_value is None


def test_linear_pareto_front() -> None:
    problem = biobjective_linear()
    X = torch.tensor([[0.0], [0.25], [1.0]], dtype=torch.double)
    expected = torch.tensor([[0.0, 1.0], [0.25, 0.75], [1.0, 0.0]], dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_truth(X), expected)
    assert (problem.reference_front >= problem.reference_point).all()


def test_zdt1_pareto_front() -> None:
    problem = zdt1()
    X = torch.zeros(3, 6, dtype=torch.double)
    X[:, 0] = torch.tensor([0.0, 0.25, 1.0], dtype=torch.double)
    expected = torch.tensor([[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]], dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_truth(X), expected)
    torch.testing.assert_close(problem.reference_front[25], expected[1])


def test_dtlz2_unit_sphere_pareto_front() -> None:
    problem = dtlz2()
    X = torch.full((2, 7), 0.5, dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(X).square().sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    torch.testing.assert_close(
        problem.reference_front.square().sum(dim=-1),
        torch.ones(problem.reference_front.shape[0], dtype=torch.double),
    )


def test_branin_currin_reference_and_boundaries() -> None:
    problem = branin_currin()
    X = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.double)
    Y = problem.evaluate_truth(X)
    assert Y.shape == (3, 2)
    assert torch.isfinite(Y).all()
    assert (Y > problem.reference_point).all()
    assert problem.reference_front is None
    assert Y[0, 1] > 0


def test_multiobjective_registry() -> None:
    registry = BenchmarkProblemRegistry()
    register_multiobjective_problems(registry)
    assert registry.names() == ("biobjective_linear", "branin_currin", "dtlz2", "zdt1")
    with pytest.raises(ValueError, match="already registered"):
        register_multiobjective_problems(registry)
