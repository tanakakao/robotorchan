"""Baseline strategy contract and reproducibility tests."""

import pytest
import torch

from robotorchan.benchmarks.baselines import (
    list_baseline_strategies,
    make_baseline_strategy,
    sobol_candidates,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


@pytest.mark.parametrize("name", ["random", "sobol"])
def test_baseline_seed_reproducibility(name: str) -> None:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy=name,
        seeds=(2, 4),
        initial_points=4,
        evaluation_budget=5,
        q=2,
    )
    strategy = make_baseline_strategy(name)
    first = run_benchmark(config, strategy, registry=registry)
    second = run_benchmark(config, strategy, registry=registry)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a.X, b.X)
        torch.testing.assert_close(a.Y_truth, b.Y_truth)
        assert a.evaluation_count == 5


def test_sobol_candidates_validate_shape() -> None:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    problem = registry.create("branin")
    X = problem.bounds.mean(dim=0).unsqueeze(0)
    Y = problem.evaluate_truth(X)
    generator = torch.Generator().manual_seed(17)
    candidates = sobol_candidates(problem, X, Y, 3, generator)
    assert candidates.shape == (3, problem.dimension)
    assert ((candidates >= problem.bounds[0]) & (candidates <= problem.bounds[1])).all()
    with pytest.raises(ValueError, match="Expected"):
        sobol_candidates(problem, X, Y, 0, generator)


def test_baseline_names_and_validation() -> None:
    assert list_baseline_strategies() == ("botorch_qei", "botorch_qnei", "random", "sobol")
    assert callable(make_baseline_strategy("botorch_qei"))
    assert callable(make_baseline_strategy("botorch_qnei"))
    with pytest.raises(ValueError, match="Unknown benchmark baseline"):
        make_baseline_strategy("unknown")
