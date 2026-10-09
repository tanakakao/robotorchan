"""Check mixed-variable domain semantics and benchmark runner integration."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.mixed_problems import (
    categorical_switch,
    mixed_category_interaction,
    mixed_process_yield,
    mixed_quadratic,
    register_mixed_problems,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark, sobol_initial_design


def test_mixed_quadratic_known_optimum() -> None:
    problem = mixed_quadratic()
    optimum = torch.tensor([[0.8, 2.0, 1.0]], dtype=torch.double)
    expected = torch.zeros(1, 1, dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_truth(optimum), expected)
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))
    assert problem.variable_types == ("continuous", "integer", "categorical")


def test_categorical_switch_known_optimum() -> None:
    problem = categorical_switch()
    optimum = torch.tensor([[0.7, 1.0]], dtype=torch.double)
    expected = torch.ones(1, 1, dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_truth(optimum), expected)
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))


@pytest.mark.parametrize("factory", [mixed_quadratic, categorical_switch])
def test_discrete_coordinates_must_be_integral(factory) -> None:
    problem = factory()
    X = problem.bounds.mean(dim=0).unsqueeze(0)
    index = next(i for i, kind in enumerate(problem.variable_types) if kind != "continuous")
    X[:, index] = 0.5
    with pytest.raises(ValueError, match="must be integral"):
        problem.evaluate_truth(X)


def test_process_yield_known_feasible_optimum() -> None:
    problem = mixed_process_yield()
    optimum = torch.tensor([[0.7, 3.0, 1.0]], dtype=torch.double)
    infeasible = torch.tensor([[0.9, 5.0, 1.0]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.ones(1, 1, dtype=torch.double)
    )
    assert (problem.evaluate_constraints(optimum) >= 0).all()
    assert (problem.evaluate_constraints(infeasible) < 0).any()
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))
    assert torch.isinf(problem.simple_regret(infeasible))


def test_category_interaction_known_optimum_and_batch() -> None:
    problem = mixed_category_interaction()
    optimum = torch.tensor([[0.85, 1.0, 1.0]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.zeros(1, 1, dtype=torch.double)
    )
    X = optimum.expand(2, 3, 3).clone()
    assert problem.evaluate_truth(X).shape == (2, 3, 1)
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))


@pytest.mark.parametrize("factory", [mixed_process_yield, mixed_category_interaction])
def test_extended_mixed_rejects_fractional_categories(factory) -> None:
    problem = factory()
    X = problem.bounds[0].unsqueeze(0).clone()
    index = next(i for i, kind in enumerate(problem.variable_types) if kind == "categorical")
    X[:, index] = 0.5
    with pytest.raises(ValueError, match="must be integral"):
        problem.evaluate_truth(X)


def test_seeded_mixed_design_and_runner() -> None:
    registry = BenchmarkProblemRegistry()
    register_mixed_problems(registry)
    assert registry.names() == (
        "categorical_switch",
        "mixed_category_interaction",
        "mixed_process_yield",
        "mixed_quadratic",
    )
    problem = registry.create("mixed_quadratic")
    initial = sobol_initial_design(problem, 12, 11, dtype=torch.double, device=torch.device("cpu"))
    problem._validate_X(initial)
    config = BenchmarkExperimentConfig(
        problem="mixed_quadratic",
        strategy="random",
        seeds=(11, 12),
        initial_points=6,
        evaluation_budget=5,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates, registry=registry)
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.X.shape == (11, 3)
        assert trajectory.evaluation_count == 5
        problem._validate_X(trajectory.X)
    with pytest.raises(ValueError, match="already registered"):
        register_mixed_problems(registry)
