"""Test analytic probability objectives and stochastic binary labels."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import hypervolume_curve, simple_regret_curve
from robotorchan.benchmarks.probability_objective_problems import (
    pass_probability_objective,
    register_probability_objective_problems,
    sample_pass_labels,
    yield_probability_tradeoff,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def test_probability_objective_known_optimum() -> None:
    problem = pass_probability_objective()
    optimum = torch.tensor([[0.7, 0.3]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.tensor([[0.9]], dtype=torch.double)
    )
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))
    X = torch.rand(2, 3, 2, dtype=torch.double)
    probability = problem.evaluate_truth(X)
    assert probability.shape == (2, 3, 1)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_binary_labels_reproducible_and_calibrated() -> None:
    X = torch.tensor([[0.7, 0.3]], dtype=torch.double).expand(20000, -1)
    first = sample_pass_labels(X, torch.Generator().manual_seed(19))
    second = sample_pass_labels(X, torch.Generator().manual_seed(19))
    assert torch.equal(first, second)
    assert first.shape == (20000, 1)
    assert ((first == 0) | (first == 1)).all()
    assert abs(first.mean().item() - 0.9) < 0.02


@pytest.mark.parametrize("name", ["pass_probability_objective", "yield_probability_tradeoff"])
def test_probability_objective_runner(name) -> None:
    registry = BenchmarkProblemRegistry()
    register_probability_objective_problems(registry)
    config = BenchmarkExperimentConfig(
        problem=name,
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
        assert a.Y_truth.shape == (11, registry.create(name).n_objectives)
        assert a.constraints.shape == (11, 0)
        if name == "pass_probability_objective":
            assert simple_regret_curve(registry.create(name), a).shape == (11,)
        else:
            curve = hypervolume_curve(registry.create(name), a)
            assert curve.shape == (11,)
            assert (curve[1:] >= curve[:-1]).all()


def test_registry_duplicate_registration() -> None:
    registry = BenchmarkProblemRegistry()
    register_probability_objective_problems(registry)
    assert registry.names() == ("pass_probability_objective", "yield_probability_tradeoff")
    with pytest.raises(ValueError, match="already registered"):
        register_probability_objective_problems(registry)
