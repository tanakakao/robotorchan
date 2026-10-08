"""Validate heterogeneous regression objectives and binary feasibility."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.heterogeneous_problems import (
    register_heterogeneous_problems,
    strength_conductivity_pass,
    strength_conductivity_pass_labels,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def test_regression_objectives_and_binary_labels() -> None:
    problem = strength_conductivity_pass()
    X = torch.tensor([[0.2, 0.7], [0.8, 0.3], [0.5, 0.5]], dtype=torch.double)
    Y = problem.evaluate_truth(X)
    labels = strength_conductivity_pass_labels(X)
    assert Y.shape == (3, 2)
    assert labels.shape == (3, 1)
    assert set(labels.flatten().tolist()) == {0.0, 1.0}
    expected = (problem.evaluate_constraints(X) >= 0).to(dtype=X.dtype)
    torch.testing.assert_close(labels, expected)
    assert problem.directions == ("maximize", "maximize")
    assert problem.n_constraints == 1


def test_pass_boundary_is_feasible() -> None:
    problem = strength_conductivity_pass()
    X = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_constraints(X), torch.zeros(1, 1, dtype=torch.double))
    torch.testing.assert_close(strength_conductivity_pass_labels(X), torch.ones(1, 1, dtype=torch.double))


def test_batched_label_shape_and_dtype() -> None:
    X = torch.rand(2, 4, 2, dtype=torch.float32)
    labels = strength_conductivity_pass_labels(X)
    assert labels.shape == (2, 4, 1)
    assert labels.dtype == X.dtype
    assert ((labels == 0) | (labels == 1)).all()


def test_heterogeneous_runner_and_registry() -> None:
    registry = BenchmarkProblemRegistry()
    register_heterogeneous_problems(registry)
    assert registry.names() == ("strength_conductivity_pass",)
    config = BenchmarkExperimentConfig(
        problem="strength_conductivity_pass",
        strategy="random",
        seeds=(3, 7),
        initial_points=5,
        evaluation_budget=6,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates, registry=registry)
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.Y_truth.shape == (11, 2)
        assert trajectory.constraints.shape == (11, 1)
        labels = strength_conductivity_pass_labels(trajectory.X)
        assert torch.equal(labels.bool(), trajectory.constraints >= 0)
    with pytest.raises(ValueError, match="already registered"):
        register_heterogeneous_problems(registry)
