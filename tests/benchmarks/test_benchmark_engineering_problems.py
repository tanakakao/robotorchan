"""Validate engineering design objectives, feasibility and benchmark execution."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.engineering_problems import (
    cantilever_beam,
    register_engineering_problems,
    thermal_management,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


@pytest.mark.parametrize(
    ("factory", "feasible", "infeasible"),
    [
        (cantilever_beam, (1.0, 1.0), (0.2, 0.2)),
        (thermal_management, (1.0, 0.5), (0.0, 0.0)),
    ],
)
def test_engineering_feasibility(factory, feasible, infeasible) -> None:
    problem = factory()
    X = torch.tensor([feasible, infeasible], dtype=torch.double)
    truth = problem.evaluate_truth(X)
    residuals = problem.evaluate_constraints(X)
    assert truth.shape == (2, 1)
    assert residuals.shape == (2, 2)
    assert torch.isfinite(truth).all()
    assert torch.isfinite(residuals).all()
    assert (residuals[0] >= 0).all()
    assert (residuals[1] < 0).any()
    assert problem.optimal_value is None


def test_engineering_batch_contract() -> None:
    for factory in (cantilever_beam, thermal_management):
        problem = factory()
        X = problem.bounds.mean(dim=0).expand(2, 3, 2).clone()
        assert problem.evaluate_truth(X).shape == (2, 3, 1)
        assert problem.evaluate_constraints(X).shape == (2, 3, 2)


def test_engineering_registry_and_runner() -> None:
    registry = BenchmarkProblemRegistry()
    register_engineering_problems(registry)
    assert registry.names() == ("cantilever_beam", "thermal_management")
    config = BenchmarkExperimentConfig(
        problem="cantilever_beam",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=3,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates, registry=registry)
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.Y_truth.shape == (7, 1)
        assert trajectory.constraints.shape == (7, 2)
        assert trajectory.evaluation_count == 3
    with pytest.raises(ValueError, match="already registered"):
        register_engineering_problems(registry)
