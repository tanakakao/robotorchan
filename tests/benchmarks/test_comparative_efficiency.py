"""Phase 18 benchmark timing and evaluation-cost tests."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.comparative_efficiency import summarize_efficiency
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import random_candidates, run_benchmark


def _runs(q: int = 2):
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=q,
    )
    return run_benchmark(config, random_candidates)


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
