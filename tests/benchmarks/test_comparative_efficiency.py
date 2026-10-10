"""Phase 18 benchmark timing and evaluation-cost tests."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.comparative_efficiency import summarize_efficiency
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _runs(q: int = 2):
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=q,
    )
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return run_benchmark(config, random_candidates, registry=registry)


def test_timing_and_cost_checkpoints() -> None:
    runs = _runs()
    result = summarize_efficiency(runs, q=2)
    assert result.seeds == (0, 1)
    assert result.evaluations.tolist() == [0, 2, 4, 5]
    assert result.total_seconds.shape == (2, 4)
    assert result.evaluation_cost.shape == (2, 4)
    assert (result.total_seconds[:, 1:] >= result.total_seconds[:, :-1]).all()
    assert (result.evaluation_cost[:, 1:] >= result.evaluation_cost[:, :-1]).all()
    torch.testing.assert_close(
        result.total_seconds, result.candidate_seconds + result.evaluation_seconds
    )
    for index, run in enumerate(runs):
        assert len(run.candidate_seconds) == 3
        assert len(run.evaluation_seconds) == 3
        assert run.initial_evaluation_seconds is not None
        torch.testing.assert_close(result.evaluation_cost[index, -1], run.cumulative_cost[-1])


def test_missing_timing_rejected() -> None:
    runs = _runs()
    invalid = replace(runs[0], candidate_seconds=())
    with pytest.raises(ValueError, match="missing"):
        summarize_efficiency((invalid,), q=2)


def test_invalid_timing_rejected() -> None:
    runs = _runs()
    invalid = replace(runs[0], candidate_seconds=(-1.0, *runs[0].candidate_seconds[1:]))
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_efficiency((invalid,), q=2)


def test_incompatible_budget_rejected() -> None:
    runs = _runs()
    with pytest.raises(ValueError, match="matching budgets"):
        summarize_efficiency((runs[0], replace(runs[1], initial_points=3)), q=2)


def test_reject_wrong_batch_boundaries() -> None:
    runs = _runs(q=3)
    with pytest.raises(ValueError, match="batch boundaries"):
        summarize_efficiency(runs, q=4)


def test_timing_survives_both_serializers(tmp_path) -> None:
    from robotorchan.benchmarks.persistence import load_trajectory, save_trajectory
    from robotorchan.benchmarks.storage import load_benchmark_results, save_benchmark_results

    runs = _runs()
    path = tmp_path / "trajectory.json"
    save_trajectory(runs[0], path)
    loaded = load_trajectory(path)
    assert loaded.completed_batches == runs[0].completed_batches
    assert loaded.candidate_seconds == runs[0].candidate_seconds
    summarize_efficiency((loaded,), q=2)

    config = BenchmarkExperimentConfig(
        problem="branin", strategy="random", seeds=(0, 1),
        initial_points=4, evaluation_budget=5, q=2,
    )
    path = tmp_path / "results.json"
    save_benchmark_results(path, config, runs)
    _, restored = load_benchmark_results(path)
    assert restored[0].evaluation_seconds == runs[0].evaluation_seconds
    summarize_efficiency(restored, q=2)
