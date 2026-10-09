"""Validate single regression objective with a binary feasibility label."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import cumulative_feasibility_rate, simple_regret_curve
from robotorchan.benchmarks.regression_binary_problems import (
    register_regression_binary_problems,
    strength_pass,
    strength_pass_labels,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def test_strength_pass_optimum_and_binary_boundary() -> None:
    problem = strength_pass()
    optimum = torch.tensor([[0.66, 0.5800000000000001]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.tensor([[0.902]], dtype=torch.double)
    )
    torch.testing.assert_close(
        problem.evaluate_constraints(optimum),
        torch.zeros(1, 1, dtype=torch.double),
        atol=1e-14,
        rtol=0,
    )
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))
    assert strength_pass_labels(optimum).item() == 1


def test_binary_labels_match_truth_feasibility() -> None:
    problem = strength_pass()
    X = torch.tensor([[0.8, 0.3], [0.66, 0.5800000000000001], [0.2, 0.9]], dtype=torch.double)
    labels = strength_pass_labels(X)
    assert labels.shape == (3, 1)
    assert set(labels.flatten().tolist()) == {0.0, 1.0}
    torch.testing.assert_close(labels, (problem.evaluate_constraints(X) >= 0).to(X.dtype))
    batched = X.reshape(1, 3, 2).to(dtype=torch.float32)
    assert strength_pass_labels(batched).shape == (1, 3, 1)


def test_regression_binary_runner_and_metrics() -> None:
    registry = BenchmarkProblemRegistry()
    register_regression_binary_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="strength_pass",
        strategy="random",
        seeds=(3, 7),
        initial_points=5,
        evaluation_budget=6,
        q=2,
    )
    first = run_benchmark(config, random_candidates, registry=registry)
    second = run_benchmark(config, random_candidates, registry=registry)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a.X, b.X)
        assert a.Y_truth.shape == (11, 1)
        assert a.constraints.shape == (11, 1)
        torch.testing.assert_close(
            strength_pass_labels(a.X), (a.constraints >= 0).to(dtype=a.X.dtype)
        )
        assert simple_regret_curve(registry.create("strength_pass"), a).shape == (11,)
        assert cumulative_feasibility_rate(a.constraints).shape == (11,)
    with pytest.raises(ValueError, match="already registered"):
        register_regression_binary_problems(registry)
