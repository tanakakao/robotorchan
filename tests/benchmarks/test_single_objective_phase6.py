"""Phase 6 analytic single-objective benchmark regression tests."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import (
    ackley2,
    branin,
    hartmann6,
    register_standard_problems,
    rosenbrock2,
    sphere3,
)


@pytest.mark.parametrize(
    ("factory", "optimum"),
    [
        (ackley2, (0.0, 0.0)),
        (rosenbrock2, (1.0, 1.0)),
        (sphere3, (0.0, 0.0, 0.0)),
    ],
)
def test_known_minimizers(factory, optimum) -> None:
    problem = factory()
    X = torch.tensor([optimum], dtype=torch.double)
    values = problem.evaluate_truth(X)
    assert values.shape == (1, 1)
    assert torch.allclose(values, problem.optimal_value.unsqueeze(0), atol=1e-12)


@pytest.mark.parametrize("factory", [ackley2, branin, hartmann6, rosenbrock2, sphere3])
def test_batched_truth_and_regret(factory) -> None:
    problem = factory()
    X = problem.bounds.mean(dim=0).repeat(2, 3, 1)
    values = problem.evaluate_truth(X)
    regret = problem.simple_regret(X)
    assert values.shape == (2, 3, 1)
    assert regret.shape == (2,)
    assert torch.isfinite(values).all()
    assert torch.all(regret >= 0)


def test_registry_and_seeded_runner() -> None:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    assert registry.names() == (
        "ackley2",
        "branin",
        "hartmann6",
        "rosenbrock2",
        "sphere3",
    )
    config = BenchmarkExperimentConfig(
        problem="ackley2",
        strategy="random",
        seeds=(2,),
        initial_points=4,
        evaluation_budget=3,
        q=2,
    )
    first = run_benchmark(config, random_candidates, registry=registry)[0]
    second = run_benchmark(config, random_candidates, registry=registry)[0]
    assert first.X.shape == (7, 2)
    assert torch.equal(first.X, second.X)
    assert torch.equal(first.Y_observed, second.Y_observed)
