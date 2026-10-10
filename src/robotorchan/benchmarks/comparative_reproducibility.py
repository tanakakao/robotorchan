"""Phase 23 reproducibility checks for deterministic benchmark trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.persistence import trajectory_from_record, trajectory_to_record
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
)
from robotorchan.benchmarks.standard_problems import register_standard_problems

_TENSOR_FIELDS = ("X", "Y_observed", "Y_truth", "constraints", "costs")


@dataclass(frozen=True)
class ReproducibilityResult:
    """Numerical reproducibility outcome excluding nondeterministic wall time."""

    seeds: tuple[int, ...]
    evaluation_budget: int
    repeat_equal: bool
    persistence_equal: bool


def _numerically_equal(left: BenchmarkTrajectory, right: BenchmarkTrajectory) -> bool:
    if (
        left.seed != right.seed
        or left.initial_points != right.initial_points
        or left.completed_batches != right.completed_batches
    ):
        return False
    return all(torch.equal(getattr(left, field), getattr(right, field)) for field in _TENSOR_FIELDS)


def verify_random_baseline_reproducibility(
    config: BenchmarkExperimentConfig,
) -> ReproducibilityResult:
    """Verify repeatability and JSON roundtrip for the seeded random baseline.

    Wall-clock measurements are intentionally excluded: elapsed time is not
    reproducible, even when numerical trajectories are bitwise identical.
    """
    if config.strategy != "random":
        raise ValueError("Reproducibility smoke check requires random strategy.")
    if config.device != "cpu":
        raise ValueError("Reproducibility smoke check is CPU-only.")
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    first = run_benchmark(config, random_candidates, registry=registry)
    second = run_benchmark(config, random_candidates, registry=registry)
    repeat_equal = all(
        _numerically_equal(left, right) for left, right in zip(first, second, strict=True)
    )
    persistence_equal = all(
        _numerically_equal(
            run,
            trajectory_from_record(json.loads(json.dumps(trajectory_to_record(run)))),
        )
        for run in first
    )
    return ReproducibilityResult(
        seeds=config.seeds,
        evaluation_budget=config.evaluation_budget,
        repeat_equal=repeat_equal,
        persistence_equal=persistence_equal,
    )
