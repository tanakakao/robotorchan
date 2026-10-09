"""Verify multi-fidelity response, cost and seeded runner behavior."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.multifidelity_problems import (
    multifidelity_oscillatory,
    multifidelity_quadratic,
    register_multifidelity_problems,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


@pytest.mark.parametrize("factory", [multifidelity_quadratic, multifidelity_oscillatory])
def test_fidelity_changes_response_and_cost(factory) -> None:
    problem = factory()
    X = torch.tensor([[0.4, 0.0], [0.4, 1.0]], dtype=torch.double)
    response = problem.evaluate_truth(X)
    cost = problem.evaluate_cost(X)
    assert response.shape == cost.shape == (2, 1)
    assert response[0, 0] > response[1, 0]
    torch.testing.assert_close(cost[:, 0], torch.tensor([0.1, 1.0], dtype=torch.double))
    assert problem.evaluate_observation(X).equal(response)


def test_quadratic_high_fidelity_optimum() -> None:
    problem = multifidelity_quadratic()
    optimum = torch.tensor([[0.7, 1.0]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.zeros(1, 1, dtype=torch.double)
    )
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))


@pytest.mark.parametrize("factory", [multifidelity_quadratic, multifidelity_oscillatory])
def test_multifidelity_batch_and_bounds(factory) -> None:
    problem = factory()
    X = torch.full((2, 3, 2), 0.5, dtype=torch.double)
    assert problem.evaluate_truth(X).shape == (2, 3, 1)
    assert problem.evaluate_cost(X).shape == (2, 3, 1)
    X[..., -1] = 1.1
    with pytest.raises(ValueError, match="within bounds"):
        problem.evaluate_truth(X)


def test_multifidelity_registry_and_runner() -> None:
    registry = BenchmarkProblemRegistry()
    register_multifidelity_problems(registry)
    assert registry.names() == ("multifidelity_oscillatory", "multifidelity_quadratic")
    config = BenchmarkExperimentConfig(
        problem="multifidelity_quadratic",
        strategy="random",
        seeds=(3, 7),
        initial_points=4,
        evaluation_budget=3,
        q=2,
    )
    first = run_benchmark(config, random_candidates, registry=registry)
    second = run_benchmark(config, random_candidates, registry=registry)
    assert len(first) == 2
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a.X, b.X)
        torch.testing.assert_close(a.costs, b.costs)
        assert a.X.shape == (7, 2)
        assert a.costs.shape == (7, 1)
        assert (a.costs >= 0.1).all()
        assert (a.costs <= 1.0).all()
        assert a.evaluation_count == 3
    with pytest.raises(ValueError, match="already registered"):
        register_multifidelity_problems(registry)
