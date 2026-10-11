"""Regression tests for deterministic and paired observation-noise streams."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.observation_noise import SeededObservationNoise
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _noise(truth: Tensor, generator: torch.Generator) -> Tensor:
    return torch.randn(
        truth.shape, generator=generator, device=truth.device, dtype=truth.dtype
    ) * 0.1


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def _run(strategy: str, policy: str) -> tuple:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy=strategy,
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    return run_benchmark(
        config,
        random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        observation_noise=SeededObservationNoise(_noise, policy=policy),
    )


def test_seeded_noise_repeats_exactly() -> None:
    first = _run("random", "paired")
    second = _run("random", "paired")
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.X, right.X)
        assert torch.equal(left.Y_observed, right.Y_observed)
        assert torch.equal(left.Y_truth, right.Y_truth)
        assert not torch.equal(left.Y_observed, left.Y_truth)


def test_paired_noise_is_strategy_invariant() -> None:
    first = _run("random", "paired")
    second = _run("sobol", "paired")
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.X, right.X)
        assert torch.equal(left.Y_observed, right.Y_observed)


def test_independent_noise_differs_across_strategies() -> None:
    first = _run("random", "independent")
    second = _run("sobol", "independent")
    assert torch.equal(first[0].X, second[0].X)
    assert not torch.equal(first[0].Y_observed, second[0].Y_observed)


def test_noise_callback_must_preserve_shape() -> None:
    bad_noise = SeededObservationNoise(lambda truth, generator: truth[..., 0])
    generator = bad_noise.generator(0, "random", torch.device("cpu"))
    try:
        bad_noise.observe(torch.zeros(3, 1), generator)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected a shape validation error.")
