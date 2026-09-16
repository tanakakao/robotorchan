"""Tests for the continuous high-dimensional acquisition benchmark."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "high_dimensional_acqf_optimization.py"
SPEC = importlib.util.spec_from_file_location("high_dimensional_acqf_optimization", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_run_dimension_compares_shared_problem() -> None:
    results = benchmark.run_dimension(
        5,
        n_train=8,
        random_samples=32,
        num_restarts=2,
        raw_samples=16,
        seed=3,
    )

    assert {row.strategy for row in results} == {"OriginalSpace", "RandomSearch"}
    assert all(row.input_dim == 5 for row in results)
    assert all(row.seed == 3 for row in results)
    assert all(row.simple_regret >= 0.0 for row in results)
    assert all(row.optimization_time >= 0.0 for row in results)


def test_run_benchmark_and_aggregation() -> None:
    results = benchmark.run_benchmark(
        [5],
        [0, 1],
        n_train=8,
        random_samples=24,
        num_restarts=2,
        raw_samples=16,
    )
    summaries = benchmark.aggregate_results(results)

    assert len(results) == 4
    assert len(summaries) == 2
    assert {row.strategy for row in summaries} == {"OriginalSpace", "RandomSearch"}
    assert all(row.n_seeds == 2 for row in summaries)


def test_benchmark_validates_inputs() -> None:
    with pytest.raises(ValueError, match="input dimension"):
        benchmark.make_problem(4, n_train=8, seed=0)
    with pytest.raises(ValueError, match="n_train"):
        benchmark.make_problem(5, n_train=1, seed=0)
    with pytest.raises(ValueError, match="input_dims"):
        benchmark.run_benchmark([], [0])
    with pytest.raises(ValueError, match="seeds"):
        benchmark.run_benchmark([5], [])
    with pytest.raises(ValueError, match="results"):
        benchmark.aggregate_results([])
