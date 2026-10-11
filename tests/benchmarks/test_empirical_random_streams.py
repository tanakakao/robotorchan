"""Deterministic RNG stream contracts for empirical comparisons."""

from __future__ import annotations

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.random_streams import derive_stream_seed, make_stream_generator
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def test_named_stream_seeds_are_stable_and_distinct() -> None:
    """Streams are stable and do not share the same seed."""
    assert derive_stream_seed(7, "candidate") == derive_stream_seed(7, "candidate")
    assert derive_stream_seed(7, "candidate") != derive_stream_seed(7, "initial_design")
    assert derive_stream_seed(7, "candidate") != derive_stream_seed(8, "candidate")


def test_named_generators_do_not_consume_global_rng() -> None:
    """Constructing and drawing from a local stream leaves global state intact."""
    before = torch.random.get_rng_state().clone()
    generator = make_stream_generator(3, "candidate", torch.device("cpu"))
    torch.rand(8, generator=generator)
    assert torch.equal(before, torch.random.get_rng_state())


def test_independent_streams_repeat_for_random_baseline() -> None:
    """Repeat a restricted baseline without mixing initial and candidate RNG."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    first = run_benchmark(
        config,
        random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
    )
    second = run_benchmark(
        config,
        random_candidates,
        registry=_registry(),
        restrict_candidate_problem=True,
        independent_rng_streams=True,
    )
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.X, right.X)
        assert torch.equal(left.Y_observed, right.Y_observed)
        assert left.completed_batches == (3, 5)


def test_legacy_initial_design_is_unchanged_by_default() -> None:
    """The optional stream contract does not alter existing runner defaults."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        initial_points=4,
        evaluation_budget=1,
    )
    from robotorchan.benchmarks.runner import sobol_initial_design

    legacy = run_benchmark(config, random_candidates, registry=_registry())[0]
    expected = sobol_initial_design(
        _registry().create("branin"),
        4,
        0,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert torch.equal(legacy.X[:4], expected)
