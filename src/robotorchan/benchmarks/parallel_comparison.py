"""Comparable sequential, batch and asynchronous benchmark baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from robotorchan.benchmarks.async_runner import (
    AsyncBenchmarkResult,
    run_async_benchmark,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
)


@dataclass(frozen=True)
class ParallelBenchmarkComparison:
    """Matched-seed random baselines with equal candidate evaluation budgets."""

    sequential: tuple[BenchmarkTrajectory, ...]
    batch: tuple[BenchmarkTrajectory, ...]
    asynchronous: tuple[AsyncBenchmarkResult, ...]


def _async_random_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    pending_X: Tensor,
    q: int,
    generator: torch.Generator,
) -> Tensor:
    del pending_X
    return random_candidates(problem, X, Y, q, generator)


def run_parallel_comparison(
    config: BenchmarkExperimentConfig,
    *,
    batch_size: int,
    max_concurrency: int,
    registry: BenchmarkProblemRegistry | None = None,
) -> ParallelBenchmarkComparison:
    """Compare sequential, synchronous batch and asynchronous random search.

    Each arm uses identical initial-design seeds and evaluation budgets.
    Asynchronous durations are fixed at one simulated time unit.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer.")
    sequential_config = replace(config, q=1)
    batch_config = replace(config, q=batch_size)
    async_config = replace(config, q=batch_size)
    return ParallelBenchmarkComparison(
        sequential=run_benchmark(sequential_config, random_candidates, registry=registry),
        batch=run_benchmark(batch_config, random_candidates, registry=registry),
        asynchronous=run_async_benchmark(
            async_config,
            _async_random_candidates,
            lambda candidate: 1.0,
            max_concurrency=max_concurrency,
            registry=registry,
        ),
    )
