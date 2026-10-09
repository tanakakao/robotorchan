"""Test matched-seed sequential, batch and asynchronous comparisons."""

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.parallel_comparison import run_parallel_comparison


def test_parallel_comparison_matches_budgets_and_initial_designs() -> None:
    config = BenchmarkExperimentConfig(
        problem="sphere3",
        strategy="random",
        seeds=(2, 4),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    result = run_parallel_comparison(config, batch_size=3, max_concurrency=2)
    assert len(result.sequential) == len(result.batch) == len(result.asynchronous) == 2
    for sequential, batch, asynchronous in zip(
        result.sequential, result.batch, result.asynchronous, strict=True
    ):
        for trajectory in (sequential, batch, asynchronous):
            assert trajectory.X.shape == (9, 3)
            torch.testing.assert_close(trajectory.X[:4], sequential.X[:4])
        assert sequential.evaluation_count == batch.evaluation_count == 5
        assert len(asynchronous.evaluations) == 5
        assert asynchronous.max_concurrency <= 2
        assert asynchronous.simulated_makespan > 0
        assert sorted(asynchronous.completion_order) == list(range(5))


def test_parallel_comparison_is_reproducible() -> None:
    config = BenchmarkExperimentConfig(
        problem="sphere3",
        strategy="random",
        seeds=(1,),
        initial_points=3,
        evaluation_budget=4,
        q=2,
    )
    first = run_parallel_comparison(config, batch_size=2, max_concurrency=2)
    second = run_parallel_comparison(config, batch_size=2, max_concurrency=2)
    for arm in ("sequential", "batch", "asynchronous"):
        torch.testing.assert_close(getattr(first, arm)[0].X, getattr(second, arm)[0].X)


@pytest.mark.parametrize(("batch_size", "concurrency"), [(0, 2), (2, 0), (True, 2)])
def test_parallel_comparison_rejects_invalid_parallelism(batch_size, concurrency) -> None:
    config = BenchmarkExperimentConfig(problem="sphere3", strategy="random")
    with pytest.raises(ValueError, match="positive integer"):
        run_parallel_comparison(
            config, batch_size=batch_size, max_concurrency=concurrency
        )
