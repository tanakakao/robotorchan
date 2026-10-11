"""Phase 24 final audit of benchmark configuration and seed contracts."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def test_config_json_contract_preserves_comparison_dimensions() -> None:
    original = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(3, 7),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    restored = BenchmarkExperimentConfig.from_dict(original.to_dict())
    assert restored == original
    for field in ("initial_points", "evaluation_budget", "q", "seeds"):
        assert field in original.to_dict()


def test_seeded_random_smoke_respects_budget_and_partial_batch() -> None:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(3, 7),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    runs = run_benchmark(config, random_candidates, registry=registry)
    assert tuple(run.seed for run in runs) == config.seeds
    for run in runs:
        assert run.evaluation_count == config.evaluation_budget
        assert run.completed_batches == (3, 5)
        assert run.X.shape[0] == 9
        assert torch.isfinite(run.Y_truth).all()


def test_invalid_comparison_seeds_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        BenchmarkExperimentConfig(
            problem="branin",
            strategy="random",
            seeds=(3, 3),
        )
