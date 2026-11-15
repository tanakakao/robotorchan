"""Smoke and fairness tests for the ordinal active-learning benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

BENCHMARK = (
    Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / "ordinal_active_learning.py"
)


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("ordinal_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mixed", [False, True])
def test_ordinal_benchmark_dataset(benchmark, mixed: bool) -> None:
    X, y = benchmark.dataset(13, mixed=mixed)
    X_again, y_again = benchmark.dataset(13, mixed=mixed)
    torch.testing.assert_close(X, X_again)
    torch.testing.assert_close(y, y_again)
    assert X.shape == (120, 2)
    assert y.shape == (120,)
    assert y.dtype == torch.long
    assert set(y.unique().tolist()).issubset({0, 1, 2})
    if mixed:
        assert set(X[:, 1].unique().tolist()).issubset({0.0, 1.0, 2.0})


@pytest.mark.parametrize("mixed", [False, True])
def test_ordinal_benchmark_shared_initial_metrics(benchmark, mixed: bool) -> None:
    """All acquisition methods start with exactly the same model and holdout."""
    rows = [
        benchmark.evaluate(13, mixed=mixed, method=method, budget=0, epochs=1, num_samples=8)[0]
        for method in benchmark.METHODS
    ]
    for row in rows:
        assert row["seed"] == 13
        assert row["mixed"] == int(mixed)
        assert row["labels_added"] == 0
        for metric in ("accuracy", "nll", "brier", "ordinal_mae"):
            assert torch.isfinite(torch.tensor(row[metric]))
        assert 0 <= row["accuracy"] <= 1
        assert row["nll"] >= 0
        assert 0 <= row["brier"] <= 2
        assert 0 <= row["ordinal_mae"] <= 2
    for metric in ("accuracy", "nll", "brier", "ordinal_mae"):
        assert rows[0][metric] == pytest.approx(rows[1][metric])
        assert rows[0][metric] == pytest.approx(rows[2][metric])


@pytest.mark.parametrize(
    ("mixed", "method"),
    [(False, "random"), (False, "entropy"), (True, "entropy"), (True, "bald")],
)
def test_ordinal_benchmark_selection_step(benchmark, mixed: bool, method: str) -> None:
    rows = benchmark.evaluate(17, mixed=mixed, method=method, budget=1, epochs=1, num_samples=8)
    assert len(rows) == 2
    assert [row["labels_added"] for row in rows] == [0, 1]
    assert all(row["method"] == method for row in rows)
    assert all(torch.isfinite(torch.tensor(row["nll"])) for row in rows)


def test_ordinal_benchmark_input_validation(benchmark) -> None:
    with pytest.raises(ValueError, match="method"):
        benchmark.evaluate(1, mixed=False, method="invalid", budget=0, epochs=1)
    with pytest.raises(ValueError, match="budget"):
        benchmark.evaluate(1, mixed=False, method="random", budget=76, epochs=1)
    with pytest.raises(ValueError, match="num_samples"):
        benchmark.evaluate(1, mixed=False, method="random", budget=0, epochs=1, num_samples=1)
