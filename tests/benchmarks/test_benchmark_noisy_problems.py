"""Validate noisy observation contracts and reproducible benchmark trajectories."""

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.noisy_problems import (
    heteroscedastic_quadratic,
    noisy_quadratic,
    register_noisy_problems,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def test_noisy_observations_do_not_modify_truth() -> None:
    for factory in (noisy_quadratic, heteroscedastic_quadratic):
        problem = factory()
        X = torch.tensor([[0.35, 0.35], [0.8, 0.4]], dtype=torch.double)
        truth = problem.evaluate_truth(X)
        observed = problem.evaluate_observation(X)
        assert truth.shape == observed.shape == (2, 1)
        assert not torch.equal(truth, observed)
        torch.testing.assert_close(problem.evaluate_truth(X), truth)
        torch.testing.assert_close(problem.simple_regret(X), torch.zeros((), dtype=torch.double))


def test_noisy_factory_reproducibility_and_batch_shapes() -> None:
    X = torch.full((2, 3, 2), 0.5, dtype=torch.double)
    for factory in (noisy_quadratic, heteroscedastic_quadratic):
        first = factory()
        second = factory()
        torch.testing.assert_close(first.evaluate_observation(X), second.evaluate_observation(X))
        assert first.evaluate_truth(X).shape == (2, 3, 1)


def test_seeded_runner_records_truth_separately() -> None:
    registry = BenchmarkProblemRegistry()
    register_noisy_problems(registry)
    assert registry.names() == ("heteroscedastic_quadratic", "noisy_quadratic")
    config = BenchmarkExperimentConfig(
        problem="noisy_quadratic",
        strategy="random",
        seeds=(3, 5),
        initial_points=4,
        evaluation_budget=3,
        q=2,
    )
    first = run_benchmark(config, random_candidates, registry=registry)
    second = run_benchmark(config, random_candidates, registry=registry)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a.X, b.X)
        torch.testing.assert_close(a.Y_truth, b.Y_truth)
        torch.testing.assert_close(a.Y_observed, b.Y_observed)
        assert a.Y_truth.shape == a.Y_observed.shape == (7, 1)
        assert not torch.equal(a.Y_truth, a.Y_observed)
