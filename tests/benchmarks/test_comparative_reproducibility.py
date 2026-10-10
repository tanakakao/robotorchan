"""Phase 23 reproducibility smoke tests for the benchmark CI suite."""

import pytest

from robotorchan.benchmarks.comparative_reproducibility import (
    verify_random_baseline_reproducibility,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


@pytest.mark.parametrize("q", [1, 3])
def test_seeded_random_trajectories_repeat_and_roundtrip(q: int) -> None:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 7),
        initial_points=4,
        evaluation_budget=5,
        q=q,
    )
    result = verify_random_baseline_reproducibility(config)
    assert result.repeat_equal
    assert result.persistence_equal
    assert result.seeds == (0, 7)


def test_smoke_rejects_other_strategies() -> None:
    config = BenchmarkExperimentConfig(problem="branin", strategy="sobol")
    with pytest.raises(ValueError, match="random strategy"):
        verify_random_baseline_reproducibility(config)


def test_budget_and_batch_size_change_experiment_contract() -> None:
    base = BenchmarkExperimentConfig(problem="branin", strategy="random")
    larger = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        evaluation_budget=base.evaluation_budget + 1,
    )
    assert base.to_dict() != larger.to_dict()
