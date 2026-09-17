"""Tests for the sequential continuous high-dimensional BO benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_NAME = "high_dimensional_sequential_bo"
MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / f"{MODULE_NAME}.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def test_initial_data_is_reproducible() -> None:
    X1, Y1, bounds1 = benchmark.make_initial_data(6, n_train=4, seed=3)
    X2, Y2, bounds2 = benchmark.make_initial_data(6, n_train=4, seed=3)
    assert torch.equal(X1, X2)
    assert torch.equal(Y1, Y2)
    assert torch.equal(bounds1, bounds2)


def test_random_search_runs_sequentially() -> None:
    rows = benchmark.run_strategy(
        "RandomSearch",
        6,
        n_train=4,
        n_iterations=2,
        latent_dim=2,
        random_samples=16,
        num_restarts=1,
        raw_samples=4,
        seed=1,
    )
    assert len(rows) == 2
    assert [row.iteration for row in rows] == [1, 2]
    assert rows[1].best_observed >= rows[0].best_observed
    assert rows[1].simple_regret <= rows[0].simple_regret
    assert all(row.optimization_time >= 0.0 for row in rows)


def test_latent_pca_runs_sequentially() -> None:
    rows = benchmark.run_strategy(
        "LatentPCA",
        6,
        n_train=6,
        n_iterations=1,
        latent_dim=2,
        random_samples=8,
        num_restarts=1,
        raw_samples=8,
        seed=2,
    )
    assert len(rows) == 1
    assert rows[0].strategy == "LatentPCA"
    assert rows[0].input_dim == 6


def test_aggregate_results_groups_iterations() -> None:
    rows = [
        benchmark.SequentialBOResult("RandomSearch", 6, 0, 1, -1.0, -0.5, 0.5, 0.1),
        benchmark.SequentialBOResult("RandomSearch", 6, 1, 1, -0.8, -0.4, 0.4, 0.2),
    ]
    summaries = benchmark.aggregate_results(rows)
    assert len(summaries) == 1
    assert summaries[0].n_seeds == 2
    assert summaries[0].iteration == 1
    assert summaries[0].simple_regret_mean == pytest.approx(0.45)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_iterations"):
        benchmark.run_strategy("RandomSearch", 6, n_iterations=0)
    with pytest.raises(ValueError, match="latent_dim"):
        benchmark.run_strategy("RandomSearch", 6, latent_dim=7)
