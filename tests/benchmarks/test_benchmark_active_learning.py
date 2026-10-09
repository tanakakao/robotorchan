"""Tests for active learning truth, labels and prediction metrics."""

import pytest
import torch

from robotorchan.benchmarks.active_learning_problems import (
    active_learning_labels,
    boundary_mae,
    brier_score,
    circle_boundary,
    circle_margin,
    classification_accuracy,
    register_active_learning_problems,
    wave_boundary,
    wave_margin,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


@pytest.mark.parametrize(
    ("factory", "margin", "boundary"),
    [(circle_boundary, circle_margin, "circle"), (wave_boundary, wave_margin, "wave")],
)
def test_active_learning_truth_and_labels(factory, margin, boundary) -> None:
    problem = factory()
    X = torch.tensor([[0.5, 0.5], [0.0, 0.0], [0.8, 0.9]], dtype=torch.double)
    torch.testing.assert_close(problem.evaluate_constraints(X), margin(X))
    labels = active_learning_labels(X, boundary=boundary)
    assert labels.shape == (3, 1)
    torch.testing.assert_close(labels, (margin(X) >= 0).to(X.dtype))
    assert problem.evaluate_truth(X).shape == (3, 1)
    assert (problem.evaluate_truth(X) == 0).all()
    with pytest.raises(ValueError, match="Unknown boundary"):
        active_learning_labels(X, boundary="unknown")


def test_active_learning_metrics() -> None:
    labels = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.double)
    probabilities = torch.tensor([[0.1], [0.9], [0.8]], dtype=torch.double)
    torch.testing.assert_close(classification_accuracy(labels, probabilities), torch.tensor(1.0))
    torch.testing.assert_close(brier_score(labels, probabilities), torch.tensor(0.02))
    torch.testing.assert_close(boundary_mae(labels, probabilities), torch.tensor(0.13333333333333333))
    with pytest.raises(ValueError, match="Probabilities"):
        brier_score(labels, probabilities + 1)
    with pytest.raises(ValueError, match="Labels"):
        classification_accuracy(labels, probabilities[:2])


@pytest.mark.parametrize("name", ["circle_boundary", "wave_boundary"])
def test_active_learning_runner_reproducibility(name) -> None:
    registry = BenchmarkProblemRegistry()
    register_active_learning_problems(registry)
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
        torch.testing.assert_close(a.constraints, b.constraints)
        assert a.constraints.shape == (11, 1)


def test_active_learning_registration() -> None:
    registry = BenchmarkProblemRegistry()
    register_active_learning_problems(registry)
    assert registry.names() == ("circle_boundary", "wave_boundary")
    with pytest.raises(ValueError, match="already registered"):
        register_active_learning_problems(registry)
