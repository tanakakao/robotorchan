"""Smoke-test matched sequential, batch and asynchronous baseline contracts."""

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.parallel_comparison import run_parallel_comparison


def test_parallel_comparison_has_paired_initial_design_and_equal_budget():
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0,),
        initial_points=4,
        evaluation_budget=4,
        q=3,
    )
    comparison = run_parallel_comparison(config, batch_size=3, max_concurrency=3)
    sequential = comparison.sequential[0]
    batch = comparison.batch[0]
    asynchronous = comparison.asynchronous[0]
    for run in (batch, asynchronous):
        torch.testing.assert_close(run.X[:4], sequential.X[:4], rtol=0, atol=0)
        assert run.X.shape == sequential.X.shape == (8, 2)
    assert sequential.completed_batches == (1, 2, 3, 4)
    assert batch.completed_batches == (3, 4)
    assert len(asynchronous.evaluations) == 4
    assert asynchronous.max_concurrency <= 3
    assert asynchronous.simulated_makespan > 0
