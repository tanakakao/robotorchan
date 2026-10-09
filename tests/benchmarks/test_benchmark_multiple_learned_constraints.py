"""Test benchmarks with two binary or mixed outcome constraints."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import cumulative_feasibility_rate, simple_regret_curve
from robotorchan.benchmarks.multiple_learned_constraints import (
    binary_constraint_labels,
    register_multiple_learned_constraint_problems,
    yield_binary_continuous_constraints,
    yield_two_binary_constraints,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def test_two_binary_constraints_and_known_optimum() -> None:
    problem = yield_two_binary_constraints()
    X = torch.tensor([[0.4, 0.36], [0.8, 0.2], [0.2, 0.8]], dtype=torch.double)
    labels = binary_constraint_labels(X)
    assert labels.shape == (3, 2)
    assert set(labels.flatten().tolist()) == {0.0, 1.0}
    torch.testing.assert_close(labels, (problem.evaluate_constraints(X) >= 0).to(X.dtype))
    optimum = torch.tensor([[0.4, 0.36]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), torch.tensor([[0.8144]], dtype=torch.double)
    )
    torch.testing.assert_close(problem.simple_regret(optimum), torch.zeros((), dtype=torch.double))


def test_mixed_binary_and_continuous_optimum() -> None:
    problem = yield_binary_continuous_constraints()
    x0 = 0.8 / 1.16
    optimum = torch.tensor([[x0, 0.2 + 0.4 * x0]], dtype=torch.double)
    torch.testing.assert_close(
        problem.evaluate_truth(optimum), problem.optimal_value.unsqueeze(0)
    )
    assert (problem.evaluate_constraints(optimum) >= -1e-12).all()
    assert problem.n_constraints == 2


@pytest.mark.parametrize(
    "name",
    ["yield_binary_continuous_constraints", "yield_two_binary_constraints"],
)
def test_multiple_constraints_runner_and_metrics(name) -> None:
    registry = BenchmarkProblemRegistry()
    register_multiple_learned_constraint_problems(registry)
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
        assert a.Y_truth.shape == (11, 1)
        assert a.constraints.shape == (11, 2)
        assert cumulative_feasibility_rate(a.constraints).shape == (11,)
        assert simple_regret_curve(registry.create(name), a).shape == (11,)


def test_registration_is_explicit() -> None:
    registry = BenchmarkProblemRegistry()
    register_multiple_learned_constraint_problems(registry)
    assert registry.names() == (
        "yield_binary_continuous_constraints",
        "yield_two_binary_constraints",
    )
    with pytest.raises(ValueError, match="already registered"):
        register_multiple_learned_constraint_problems(registry)
