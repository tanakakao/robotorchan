"""Validate high-dimensional benchmark structure and runner integration."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.high_dimensional_problems import (
    interaction_chain_30,
    register_high_dimensional_problems,
    rotated_subspace_40,
    sparse_sphere_50,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


@pytest.mark.parametrize(
    ("factory", "dimension", "center"),
    [
        (interaction_chain_30, 30, 0.4),
        (rotated_subspace_40, 40, 0.5),
        (sparse_sphere_50, 50, 0.3),
    ],
)
def test_high_dimensional_known_optimum(factory, dimension, center) -> None:
    problem = factory()
    assert problem.dimension == dimension
    optimum = torch.full((1, dimension), center, dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.zeros(1, 1, dtype=torch.double)
    )
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))
    batch = optimum.expand(2, 3, dimension).clone()
    assert problem.evaluate_truth(batch).shape == (2, 3, 1)


def test_sparse_inactive_coordinates_do_not_change_truth() -> None:
    problem = sparse_sphere_50()
    X = torch.full((2, 50), 0.3, dtype=torch.double)
    X[1, 4:] = 0.9
    torch.testing.assert_close(problem.evaluate_truth(X)[0], problem.evaluate_truth(X)[1])


def test_rotated_dense_direction_changes_truth() -> None:
    problem = rotated_subspace_40()
    X = torch.full((2, 40), 0.5, dtype=torch.double)
    X[1] += 0.1
    assert problem.evaluate_truth(X)[1, 0] > problem.evaluate_truth(X)[0, 0]


def test_interaction_chain_penalizes_neighbor_difference() -> None:
    problem = interaction_chain_30()
    X = torch.full((2, 30), 0.4, dtype=torch.double)
    X[1, 0] = 0.7
    assert problem.evaluate_truth(X)[1, 0] > problem.evaluate_truth(X)[0, 0]


def test_high_dimensional_registry_and_runner() -> None:
    registry = BenchmarkProblemRegistry()
    register_high_dimensional_problems(registry)
    assert registry.names() == (
        "interaction_chain_30",
        "rotated_subspace_40",
        "sparse_sphere_50",
    )
    config = BenchmarkExperimentConfig(
        problem="sparse_sphere_50",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=3,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates, registry=registry)
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.X.shape == (7, 50)
        assert trajectory.Y_truth.shape == (7, 1)
        assert trajectory.evaluation_count == 3
    with pytest.raises(ValueError, match="already registered"):
        register_high_dimensional_problems(registry)
