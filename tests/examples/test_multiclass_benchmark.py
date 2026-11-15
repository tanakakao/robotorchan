"""Smoke coverage for the multiclass benchmark and its held-out metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

BENCHMARK = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "benchmarks"
    / "multiclass_active_learning.py"
)


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("multiclass_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mixed", [False, True])
def test_multiclass_benchmark_dataset(benchmark, mixed: bool) -> None:
    X, y = benchmark.dataset(13, mixed=mixed)
    assert X.shape == (120, 2)
    assert y.shape == (120,)
    assert y.dtype == torch.long
    assert set(y.unique().tolist()).issubset({0, 1, 2})
    if mixed:
        assert set(X[:, 1].unique().tolist()).issubset({0.0, 1.0, 2.0})


@pytest.mark.parametrize("mixed", [False, True])
def test_multiclass_benchmark_smoke(benchmark, mixed: bool) -> None:
    rows = benchmark.evaluate(
        13, mixed=mixed, method="entropy", budget=0, epochs=1
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["seed"] == 13
    assert row["mixed"] == int(mixed)
    assert row["method"] == "entropy"
    assert row["labels_added"] == 0
    for metric in ("accuracy", "nll", "brier"):
        assert torch.isfinite(torch.tensor(row[metric]))
    assert 0 <= row["accuracy"] <= 1
    assert row["nll"] >= 0
    assert 0 <= row["brier"] <= 2
