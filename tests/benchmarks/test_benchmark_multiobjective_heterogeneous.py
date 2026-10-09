"""Test multiobjective regression with one or two binary feasibility outcomes."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import hypervolume_curve
from robotorchan.benchmarks.multiobjective_heterogeneous_problems import (
    multiobjective_pass_labels,
    multiobjective_two_labels,
    register_multiobjective_heterogeneous_problems,
    strength_conductivity_pass_tradeoff,
    strength_conductivity_two_pass,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


@pytest.mark.parametrize(
    ("factory", "n_constraints"),
    [(strength_conductivity_pass_tradeoff, 1), (strength_conductivity_two_pass, 2)],
)
def test_multiobjective_binary_labels(factory, n_constraints) -> None:
    problem = factory()
    X = torch.tensor([[0.8, 0.3], [0.5, 0.5], [0.2, 0.8]], dtype=torch.double)
    assert problem.evaluate_truth(X).shape == (3, 2)
    assert problem.evaluate_constraints(X).shape == (3, n_constraints)
    labels = multiobjective_pass_labels(X) if n_constraints == 1 else multiobjective_two_labels(X)
    torch.testing.assert_close(labels, (problem.evaluate_constraints(X) >= 0).to(X.dtype))
    assert ((labels == 0) | (labels == 1)).all()
    assert labels.shape == (3, n_constraints)


@pytest.mark.parametrize(
    "name",
    ["strength_conductivity_pass_tradeoff", "strength_conductivity_two_pass"],
)
def test_multiobjective_heterogeneous_runner_and_hypervolume(name) -> None:
    registry = BenchmarkProblemRegistry()
    register_multiobjective_heterogeneous_problems(registry)
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
        assert a.Y_truth.shape == (11, 2)
        assert a.constraints.shape == (11, registry.create(name).n_constraints)
        curve = hypervolume_curve(registry.create(name), a)
        assert curve.shape == (11,)
        assert (curve[1:] >= curve[:-1]).all()
        assert (curve >= 0).all()


def test_heterogeneous_problem_registration_is_explicit() -> None:
    registry = BenchmarkProblemRegistry()
    register_multiobjective_heterogeneous_problems(registry)
    assert registry.names() == (
        "strength_conductivity_pass_tradeoff",
        "strength_conductivity_two_pass",
    )
    with pytest.raises(ValueError, match="already registered"):
        register_multiobjective_heterogeneous_problems(registry)
