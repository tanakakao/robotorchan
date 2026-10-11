"""Roundtrip and validation tests for raw benchmark trajectories."""

from __future__ import annotations

import json

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.trajectory_persistence import (
    TrajectoryArtifact,
    load_trajectory_artifact,
    save_trajectory_artifact,
)


def _artifact() -> TrajectoryArtifact:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    return TrajectoryArtifact(
        config=config,
        trajectories=run_benchmark(config, random_candidates, registry=registry),
    )


def test_trajectory_roundtrip(tmp_path) -> None:
    original = _artifact()
    path = tmp_path / "trajectories.json"
    save_trajectory_artifact(original, path)
    loaded = load_trajectory_artifact(path)
    assert loaded.config == original.config
    for first, second in zip(original.trajectories, loaded.trajectories):
        assert first.seed == second.seed
        assert first.completed_batches == second.completed_batches
        for name in ("X", "Y_observed", "Y_truth", "constraints", "costs"):
            a = getattr(first, name)
            b = getattr(second, name)
            assert a.dtype == b.dtype
            assert torch.equal(a.cpu(), b)


def test_unsupported_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "trajectories.json"
    save_trajectory_artifact(_artifact(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_trajectory_artifact(path)


def test_modified_tensor_shape_is_rejected(tmp_path) -> None:
    path = tmp_path / "trajectories.json"
    save_trajectory_artifact(_artifact(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["trajectories"][0]["X"]["shape"] = [1, 2]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor shape"):
        load_trajectory_artifact(path)
