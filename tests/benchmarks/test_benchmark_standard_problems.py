"""Tests for analytic benchmark problem definitions."""

import pytest
import torch

from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.standard_problems import (
    branin,
    hartmann6,
    register_standard_problems,
    sphere3,
)


@pytest.mark.parametrize("factory", [branin, hartmann6, sphere3])
def test_analytic_problem_contract(factory) -> None:
    problem = factory()
    midpoint = problem.bounds.mean(dim=0).unsqueeze(0)
    values = problem.evaluate_truth(midpoint)
    assert values.shape == (1, 1)
    assert torch.isfinite(values).all()
    assert problem.evaluate_constraints(midpoint).shape == (1, 0)
    assert problem.simple_regret(midpoint).shape == ()


def test_branin_known_minimum() -> None:
    problem = branin()
    X = torch.tensor([[-torch.pi, 12.275]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(X).squeeze(),
        problem.optimal_value.squeeze(),
        atol=1e-5,
        rtol=0,
    )


def test_hartmann6_known_minimum() -> None:
    problem = hartmann6()
    X = torch.tensor(
        [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]],
        dtype=torch.double,
    )
    torch.testing.assert_close(
        problem.evaluate_truth(X).squeeze(),
        problem.optimal_value.squeeze(),
        atol=1e-4,
        rtol=0,
    )


def test_sphere_optimum_and_registry() -> None:
    problem = sphere3()
    torch.testing.assert_close(
        problem.evaluate_truth(torch.zeros(1, 3, dtype=torch.double)),
        torch.zeros(1, 1, dtype=torch.double),
    )
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    assert registry.names() == (
        "ackley2",
        "branin",
        "hartmann6",
        "rosenbrock2",
        "sphere3",
    )
    assert registry.create("branin").name == "branin"
    with pytest.raises(ValueError, match="already registered"):
        register_standard_problems(registry)
