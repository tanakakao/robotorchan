"""Paired benchmark fairness validation tests."""

from dataclasses import replace

import pytest

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.fair_comparison import FairComparisonProtocol
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _configs() -> tuple[BenchmarkExperimentConfig, BenchmarkExperimentConfig]:
    first = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(1, 2),
        initial_points=4,
        evaluation_budget=5,
        q=2,
    )
    return first, replace(first, strategy="alternative")


def test_matching_seeded_runs_pass_protocol() -> None:
    first, second = _configs()
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    a = run_benchmark(first, random_candidates, registry=registry)
    b = run_benchmark(second, random_candidates, registry=registry)
    FairComparisonProtocol(first, second).validate_trajectories(a, b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", "sphere3"),
        ("seeds", (3, 4)),
        ("initial_points", 5),
        ("evaluation_budget", 6),
        ("q", 3),
        ("dtype", "float32"),
        ("device", "cuda"),
    ],
)
def test_mismatched_settings_rejected(field: str, value: object) -> None:
    first, second = _configs()
    with pytest.raises(ValueError, match="Unmatched"):
        FairComparisonProtocol(first, replace(second, **{field: value}))


def test_identical_strategies_rejected() -> None:
    first, _ = _configs()
    with pytest.raises(ValueError, match="distinct"):
        FairComparisonProtocol(first, first)


def test_initial_design_mismatch_rejected() -> None:
    first, second = _configs()
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    a = run_benchmark(first, random_candidates, registry=registry)
    b = run_benchmark(second, random_candidates, registry=registry)
    modified = b[0].X.clone()
    modified[0, 0] += 0.01
    changed = (replace(b[0], X=modified), b[1])
    with pytest.raises(ValueError, match="Initial X"):
        FairComparisonProtocol(first, second).validate_trajectories(a, changed)


def test_seed_order_and_budget_mismatch_rejected() -> None:
    first, second = _configs()
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    a = run_benchmark(first, random_candidates, registry=registry)
    b = run_benchmark(second, random_candidates, registry=registry)
    protocol = FairComparisonProtocol(first, second)
    with pytest.raises(ValueError, match="seed order"):
        protocol.validate_trajectories(a, b[::-1])
    with pytest.raises(ValueError, match="budget"):
        protocol.validate_trajectories(a, (replace(b[0], X=b[0].X[:-1]), b[1]))
