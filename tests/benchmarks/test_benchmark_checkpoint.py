"""Checkpoint persistence and deterministic benchmark continuation."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.checkpoint import (
    load_benchmark_checkpoint,
    run_resumable_benchmark,
    save_benchmark_checkpoint,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates
from robotorchan.benchmarks.standard_problems import register_standard_problems


def _registry() -> BenchmarkProblemRegistry:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return registry


def _config() -> BenchmarkExperimentConfig:
    return BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(3,),
        initial_points=4,
        evaluation_budget=5,
        q=2,
    )


def test_resume_matches_uninterrupted_execution(tmp_path) -> None:
    config = _config()
    full, _ = run_resumable_benchmark(config, random_candidates, registry=_registry())
    partial, checkpoint = run_resumable_benchmark(
        config, random_candidates, registry=_registry(), max_batches=1
    )
    assert partial.evaluation_count == 2
    path = tmp_path / "checkpoint.json"
    save_benchmark_checkpoint(path, checkpoint)
    restored = load_benchmark_checkpoint(path)
    resumed, final_checkpoint = run_resumable_benchmark(
        config, random_candidates, registry=_registry(), checkpoint=restored
    )
    assert resumed.evaluation_count == 5
    for name in ("X", "Y_observed", "Y_truth", "constraints", "costs"):
        torch.testing.assert_close(getattr(full, name), getattr(resumed, name))
    assert final_checkpoint.trajectory.evaluation_count == 5


def test_zero_batches_preserves_initial_design() -> None:
    run, _ = run_resumable_benchmark(
        _config(), random_candidates, registry=_registry(), max_batches=0
    )
    assert run.evaluation_count == 0


def test_mismatched_configuration_rejected() -> None:
    config = _config()
    _, checkpoint = run_resumable_benchmark(
        config, random_candidates, registry=_registry(), max_batches=1
    )
    with pytest.raises(ValueError, match="configuration"):
        run_resumable_benchmark(
            replace(config, q=1),
            random_candidates,
            registry=_registry(),
            checkpoint=checkpoint,
        )


def test_multiseed_execution_rejected() -> None:
    with pytest.raises(ValueError, match="one seed"):
        run_resumable_benchmark(
            replace(_config(), seeds=(1, 2)),
            random_candidates,
            registry=_registry(),
        )
