"""Portable benchmark trajectory persistence with explicit schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from robotorchan.benchmarks.runner import BenchmarkTrajectory

SCHEMA_VERSION = 1
_TENSOR_FIELDS = ("X", "Y_observed", "Y_truth", "constraints", "costs")


def _tensor_record(value: Tensor) -> dict[str, object]:
    if value.device.type != "cpu":
        value = value.detach().cpu()
    return {"dtype": str(value.dtype).removeprefix("torch."), "data": value.tolist()}


def trajectory_to_record(trajectory: BenchmarkTrajectory) -> dict[str, object]:
    """Encode a trajectory as a JSON-compatible versioned record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": trajectory.seed,
        "initial_points": trajectory.initial_points,
        "tensors": {name: _tensor_record(getattr(trajectory, name)) for name in _TENSOR_FIELDS},
    }


def trajectory_from_record(record: dict[str, object]) -> BenchmarkTrajectory:
    """Validate and decode a versioned trajectory without executing pickle."""
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark trajectory schema version.")
    tensors = record.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(_TENSOR_FIELDS):
        raise ValueError("Trajectory record must contain all tensor fields.")
    values = {}
    for name in _TENSOR_FIELDS:
        entry = tensors[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "data"}:
            raise ValueError(f"Invalid tensor record: {name}.")
        dtype_name = entry["dtype"]
        if dtype_name not in ("float32", "float64"):
            raise ValueError(f"Unsupported tensor dtype for {name}.")
        dtype = torch.float32 if dtype_name == "float32" else torch.float64
        try:
            values[name] = torch.tensor(entry["data"], dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid tensor data: {name}.") from exc
    X = values["X"]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have nonempty shape (n, d).")
    n = X.shape[0]
    if any(values[name].ndim != 2 or values[name].shape[0] != n for name in _TENSOR_FIELDS):
        raise ValueError("Trajectory tensor row counts must agree.")
    if values["Y_observed"].shape != values["Y_truth"].shape:
        raise ValueError("Observed and truth response shapes must agree.")
    if values["Y_truth"].shape[1] == 0 or values["costs"].shape[1] != 1:
        raise ValueError("Response and cost tensor widths are invalid.")
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Trajectory tensors must contain finite values.")
    seed = record.get("seed")
    initial_points = record.get("initial_points")
    if type(seed) is not int or type(initial_points) is not int or not 0 < initial_points <= n:
        raise ValueError("Invalid trajectory seed or initial_points.")
    return BenchmarkTrajectory(seed=seed, initial_points=initial_points, **values)


def save_trajectory(trajectory: BenchmarkTrajectory, path: str | Path) -> None:
    """Save a trajectory as human-readable, non-executable JSON."""
    destination = Path(path)
    destination.write_text(
        json.dumps(trajectory_to_record(trajectory), allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_trajectory(path: str | Path) -> BenchmarkTrajectory:
    """Load and validate a trajectory from a versioned JSON file."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Trajectory JSON root must be an object.")
    return trajectory_from_record(record)
