"""Versioned, non-executable storage for benchmark experiment trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import BenchmarkTrajectory

SCHEMA_VERSION = 1
_TENSOR_FIELDS = ("X", "Y_observed", "Y_truth", "constraints", "costs")


def _encode_tensor(value: Tensor) -> dict[str, Any]:
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("Only float32 and float64 trajectory tensors are supported.")
    if not torch.isfinite(value).all():
        raise ValueError("Stored trajectory tensors must be finite.")
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "values": value.detach().cpu().tolist(),
    }


def _decode_tensor(record: dict[str, Any]) -> Tensor:
    if not isinstance(record, dict) or set(record) != {"dtype", "shape", "values"}:
        raise ValueError("Invalid tensor record.")
    dtype_name = record["dtype"]
    if dtype_name not in ("float32", "float64"):
        raise ValueError("Unsupported tensor dtype.")
    shape = record["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(dim) is not int or dim < 0 for dim in shape)
    ):
        raise ValueError("Invalid tensor shape.")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    tensor = torch.tensor(record["values"], dtype=dtype)
    if list(tensor.shape) != shape or not torch.isfinite(tensor).all():
        raise ValueError("Invalid tensor values or shape.")
    return tensor


def _validate_trajectory(item: BenchmarkTrajectory, config: BenchmarkExperimentConfig) -> None:
    count = config.initial_points + config.evaluation_budget
    if item.initial_points != config.initial_points or item.X.ndim != 2:
        raise ValueError("Invalid initial design or X shape.")
    if item.X.shape[0] != count:
        raise ValueError("Trajectory evaluation budget mismatch.")
    for name in _TENSOR_FIELDS:
        tensor = getattr(item, name)
        if not isinstance(tensor, Tensor) or tensor.ndim != 2 or tensor.shape[0] != count:
            raise ValueError(f"Invalid trajectory tensor: {name}.")
        if tensor.dtype != config.torch_dtype:
            raise ValueError(f"Trajectory dtype mismatch: {name}.")
    if item.costs.shape[1] != 1:
        raise ValueError("Costs must have one column.")


def save_benchmark_results(
    path: str | Path,
    config: BenchmarkExperimentConfig,
    trajectories: tuple[BenchmarkTrajectory, ...],
) -> None:
    """Save full observations and evaluation histories as versioned JSON.

    JSON is intentionally used instead of pickle or torch.load to avoid
    executing arbitrary Python objects when loading result artifacts.
    """
    if tuple(item.seed for item in trajectories) != config.seeds:
        raise ValueError("Trajectory seeds must match configuration order.")
    records = []
    for item in trajectories:
        _validate_trajectory(item, config)
        record = {"seed": item.seed, "initial_points": item.initial_points}
        record.update({name: _encode_tensor(getattr(item, name)) for name in _TENSOR_FIELDS})
        records.append(record)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "trajectories": records,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_benchmark_results(
    path: str | Path,
) -> tuple[BenchmarkExperimentConfig, tuple[BenchmarkTrajectory, ...]]:
    """Read validated benchmark results on CPU without importing saved code."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "config",
        "trajectories",
    }:
        raise ValueError("Invalid benchmark result schema.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark result schema version.")
    config = BenchmarkExperimentConfig.from_dict(payload["config"])
    records = payload["trajectories"]
    if not isinstance(records, list) or len(records) != len(config.seeds):
        raise ValueError("Trajectory count does not match configured seeds.")
    trajectories = []
    for expected_seed, record in zip(config.seeds, records, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "seed",
            "initial_points",
            *_TENSOR_FIELDS,
        }:
            raise ValueError("Invalid trajectory record.")
        if type(record["seed"]) is not int or record["seed"] != expected_seed:
            raise ValueError("Trajectory seed mismatch.")
        if type(record["initial_points"]) is not int:
            raise ValueError("Invalid initial point count.")
        tensors = {name: _decode_tensor(record[name]) for name in _TENSOR_FIELDS}
        item = BenchmarkTrajectory(
            seed=expected_seed,
            initial_points=record["initial_points"],
            **tensors,
        )
        _validate_trajectory(item, config)
        trajectories.append(item)
    return config, tuple(trajectories)
