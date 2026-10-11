"""Regression tests for isolated acquisition and fitting RNG state."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.strategy_rng import strategy_rng_scope


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def _global_random_candidates(problem, X, Y, q, generator) -> Tensor:
    del Y, generator
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    unit = torch.rand(q, problem.dimension, dtype=X.dtype, device=X.device)
    return bounds[0] + unit * (bounds[1] - bounds[0])


def test_strategy_scope_restores_global_state() -> None:
    before = torch.random.get_rng_state().clone()
    with strategy_rng_scope(5, 0, torch.device("cpu")):
        torch.rand(10)
    assert torch.equal(before, torch.random.get_rng_state())


def test_strategy_scope_is_batch_specific_and_repeatable() -> None:
    def draw(batch: int) -> Tensor:
        with strategy_rng_scope(5, batch, torch.device("cpu")):
            return torch.rand(5)

    assert torch.equal(draw(0), draw(0))
    assert not torch.equal(draw(0), draw(1))


def test_runner_isolates_global_random_acquisition() -> None:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    before = torch.random.get_rng_state().clone()
    first = run_benchmark(
        config,
        _global_random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
    torch.rand(20)
    second = run_benchmark(
        config,
        _global_random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.X, right.X)
        assert torch.equal(left.Y_observed, right.Y_observed)
    assert not torch.equal(before, torch.random.get_rng_state())
