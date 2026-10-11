"""Versioned, JSON-safe persistence for benchmark trajectories and configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import BenchmarkTrajectory

TRAJECTORY_SCHEMA_VERSION = 1
_TENSOR_FIELDS = ("X", "Y_observed", "Y_truth", "constraints", "costs")
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _encode_tensor(value: Tensor) -> dict[str, Any]:
    if value.dtype not in _DTYPES.values():
        raise ValueError("Only float32 and float64 tensors can be persisted.")
    if not torch.isfinite(value).all():
        raise ValueError("Trajectory tensors must be finite.")
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "values": value.detach().cpu().tolist(),
    }


def _decode_tensor(payload: Any) -> Tensor:
    if not isinstance(payload, dict) or set(payload) != {"dtype", "shape", "values"}:
        raise ValueError("Invalid trajectory tensor payload.")
    if payload["dtype"] not in _DTYPES:
        raise ValueError("Unsupported trajectory tensor dtype.")
    shape = payload["shape"]
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("Invalid trajectory tensor shape.")
    try:
        value = torch.tensor(payload["values"], dtype=_DTYPES[payload["dtype"]])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Invalid trajectory tensor values.") from exc
    if list(value.shape) != shape or not torch.isfinite(value).all():
        raise ValueError("Invalid trajectory tensor shape or nonfinite values.")
    return value


@dataclass(frozen=True)
class TrajectoryArtifact:
    """Experiment configuration and immutable collection of seeded trajectories."""

    config: BenchmarkExperimentConfig
    trajectories: tuple[BenchmarkTrajectory, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all observations without losing tensor precision metadata."""
        _validate_trajectories(self.config, self.trajectories)
        runs = []
        for trajectory in self.trajectories:
            run = {name: _encode_tensor(getattr(trajectory, name)) for name in _TENSOR_FIELDS}
            run.update(
                seed=trajectory.seed,
                initial_points=trajectory.initial_points,
                completed_batches=list(trajectory.completed_batches),
                candidate_seconds=list(trajectory.candidate_seconds),
                evaluation_seconds=list(trajectory.evaluation_seconds),
                initial_evaluation_seconds=trajectory.initial_evaluation_seconds,
            )
            runs.append(run)
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "trajectories": runs,
        }


def _validate_trajectories(
    config: BenchmarkExperimentConfig,
    trajectories: tuple[BenchmarkTrajectory, ...],
) -> None:
    if len(trajectories) != len(config.seeds):
        raise ValueError("Trajectory count does not match configured seeds.")
    for seed, run in zip(config.seeds, trajectories):
        n = config.initial_points + config.evaluation_budget
        if run.seed != seed or run.initial_points != config.initial_points:
            raise ValueError("Trajectory seed or initial design does not match configuration.")
        if run.X.ndim != 2 or run.X.shape[0] != n:
            raise ValueError("Trajectory candidate dimensions are inconsistent.")
        if any(
            getattr(run, name).ndim != 2 or getattr(run, name).shape[0] != n
            for name in _TENSOR_FIELDS
        ):
            raise ValueError("Trajectory observation dimensions are inconsistent.")
        if run.Y_observed.shape != run.Y_truth.shape or run.costs.shape[1] != 1:
            raise ValueError("Trajectory output or cost dimensions are inconsistent.")
        if (
            not run.completed_batches
            or run.completed_batches[-1] != config.evaluation_budget
            or len(run.completed_batches) != len(run.candidate_seconds)
            or len(run.completed_batches) != len(run.evaluation_seconds)
        ):
            raise ValueError("Trajectory batch history is inconsistent.")
        if any(
            current <= previous or current - previous > config.q
            for previous, current in zip((0, *run.completed_batches[:-1]), run.completed_batches)
        ):
            raise ValueError("Trajectory batch increments are inconsistent.")


def save_trajectory_artifact(artifact: TrajectoryArtifact, path: str | Path) -> None:
    """Save the full experiment history as strict JSON."""
    content = json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False)
    Path(path).write_text(content + "\n", encoding="utf-8")


def load_trajectory_artifact(path: str | Path) -> TrajectoryArtifact:
    """Load and validate a versioned experiment history on CPU."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "config", "trajectories"
    }:
        raise ValueError("Invalid trajectory artifact envelope.")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != TRAJECTORY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported trajectory schema version.")
    config = BenchmarkExperimentConfig.from_dict(payload["config"])
    if not isinstance(payload["trajectories"], list):
        raise ValueError("Invalid trajectory list.")
    trajectories = []
    for run in payload["trajectories"]:
        if not isinstance(run, dict):
            raise ValueError("Invalid trajectory entry.")
        tensors = {name: _decode_tensor(run[name]) for name in _TENSOR_FIELDS}
        trajectories.append(
            BenchmarkTrajectory(
                seed=run["seed"],
                initial_points=run["initial_points"],
                completed_batches=tuple(run["completed_batches"]),
                candidate_seconds=tuple(run["candidate_seconds"]),
                evaluation_seconds=tuple(run["evaluation_seconds"]),
                initial_evaluation_seconds=run["initial_evaluation_seconds"],
                **tensors,
            )
        )
    result = TrajectoryArtifact(config=config, trajectories=tuple(trajectories))
    _validate_trajectories(result.config, result.trajectories)
    return result
