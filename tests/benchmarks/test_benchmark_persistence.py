"""Test reproducible and validated benchmark trajectory JSON persistence."""

import json

import pytest
import torch

from robotorchan.benchmarks.persistence import (
    load_trajectory,
    save_trajectory,
    trajectory_from_record,
    trajectory_to_record,
)
from robotorchan.benchmarks.runner import BenchmarkTrajectory


def _trajectory() -> BenchmarkTrajectory:
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.double)
    Y = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.double)
    return BenchmarkTrajectory(
        seed=42,
        X=X,
        Y_observed=Y.clone(),
        Y_truth=Y,
        constraints=torch.empty(2, 0, dtype=torch.double),
        costs=torch.ones(2, 1, dtype=torch.double),
        initial_points=1,
    )


def test_json_roundtrip(tmp_path) -> None:
    original = _trajectory()
    path = tmp_path / "trajectory.json"
    save_trajectory(original, path)
    loaded = load_trajectory(path)
    assert loaded.seed == original.seed
    assert loaded.initial_points == original.initial_points
    for name in ("X", "Y_observed", "Y_truth", "constraints", "costs"):
        torch.testing.assert_close(getattr(loaded, name), getattr(original, name))
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_version_mismatch_rejected() -> None:
    record = trajectory_to_record(_trajectory())
    record["schema_version"] = 999
    with pytest.raises(ValueError, match="version"):
        trajectory_from_record(record)


def test_missing_field_rejected() -> None:
    record = trajectory_to_record(_trajectory())
    del record["tensors"]["costs"]
    with pytest.raises(ValueError, match="tensor fields"):
        trajectory_from_record(record)


def test_nonfinite_tensor_rejected() -> None:
    record = trajectory_to_record(_trajectory())
    record["tensors"]["X"]["data"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trajectory_from_record(record)


def test_inconsistent_row_count_rejected() -> None:
    record = trajectory_to_record(_trajectory())
    record["tensors"]["costs"]["data"] = [[1.0]]
    with pytest.raises(ValueError, match="row counts"):
        trajectory_from_record(record)
