"""Known-optimum and feasibility checks for constrained benchmarks."""

import pytest
import torch

from robotorchan.benchmarks.constrained_problems import (
    constrained_annulus,
    constrained_quadratic,
    register_constrained_problems,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def test_quadratic_optimum_and_infeasible_unconstrained_optimum() -> None:
    problem = constrained_quadratic()
    feasible = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    infeasible = torch.tensor([[0.8, 0.8]], dtype=torch.double)
    assert (problem.evaluate_constraints(feasible) >= 0).all()
    assert (problem.evaluate_constraints(infeasible) < 0).any()
    torch.testing.assert_close(problem.evaluate_truth(feasible).squeeze(), problem.optimal_value[0])
    torch.testing.assert_close(problem.simple_regret(feasible), torch.zeros((), dtype=torch.double))
    assert torch.isinf(problem.simple_regret(infeasible))


def test_annulus_two_constraints_and_boundary_optimum() -> None:
    problem = constrained_annulus()
    feasible = torch.tensor([[0.5, 0.0]], dtype=torch.double)
    center = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_constraints(feasible),
        torch.tensor([[0.0, 0.75]], dtype=torch.double),
    )
    assert (problem.evaluate_constraints(center) < 0).any()
    torch.testing.assert_close(problem.evaluate_truth(feasible).squeeze(), problem.optimal_value[0])
    torch.testing.assert_close(problem.simple_regret(feasible), torch.zeros((), dtype=torch.double))
    assert torch.isinf(problem.simple_regret(center))


def test_registry_and_runner() -> None:
    registry = BenchmarkProblemRegistry()
    register_constrained_problems(registry)
    assert registry.names() == ("constrained_annulus", "constrained_quadratic")
    config = BenchmarkExperimentConfig(
        problem="constrained_quadratic",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=3,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates, registry=registry)
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.constraints.shape == (7, 1)
        assert trajectory.evaluation_count == 3
    with pytest.raises(ValueError, match="already registered"):
        register_constrained_problems(registry)
