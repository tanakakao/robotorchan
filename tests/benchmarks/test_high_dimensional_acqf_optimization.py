"""Tests for the continuous high-dimensional acquisition benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_NAME = "high_dimensional_acqf_optimization"
MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / f"{MODULE_NAME}.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def test_run_dimension_compares_shared_problem() -> None:
    results = benchmark.run_dimension(
        5,
        n_train=8,
        embedding_dim=3,
        random_samples=32,
        num_restarts=2,
        raw_samples=16,
        seed=3,
    )

    assert {row.strategy for row in results} == {"OriginalSpace", "RandomSearch", "REMBO"}
    assert all(row.input_dim == 5 for row in results)
    assert all(row.seed == 3 for row in results)
    assert all(row.simple_regret >= 0.0 for row in results)
    assert all(row.optimization_time >= 0.0 for row in results)


def test_search_rng_streams_are_independent_from_problem_rng() -> None:
    seed = 7
    n_train = 8
    input_dim = 5
    model, bounds, _ = benchmark.make_problem(input_dim, n_train=n_train, seed=seed)
    strategies = benchmark.strategy_factories(
        bounds,
        embedding_dim=3,
        random_samples=n_train,
        num_restarts=1,
        raw_samples=4,
        seed=seed,
    )
    random_strategy = strategies["RandomSearch"]
    rembo_strategy = strategies["REMBO"]

    train_X = model.train_inputs[0]
    search_X = random_strategy._sample_candidate_batches(q=1).squeeze(-2)

    assert random_strategy.seed == benchmark._search_seed(seed)
    assert random_strategy.seed != seed
    assert search_X.shape == train_X.shape
    assert not torch.equal(search_X, train_X)
    assert benchmark._embedding_seed(seed) != seed
    assert benchmark._embedding_seed(seed) != benchmark._search_seed(seed)
    assert rembo_strategy.embedding.shape == (input_dim, 3)


def test_search_seeds_are_deterministic_and_wrapped() -> None:
    assert benchmark._search_seed(11) == benchmark._search_seed(11)
    assert benchmark._embedding_seed(11) == benchmark._embedding_seed(11)
    upper_seed = benchmark._MAX_TORCH_SEED - 1
    assert 0 <= benchmark._search_seed(upper_seed) < benchmark._MAX_TORCH_SEED
    assert 0 <= benchmark._embedding_seed(upper_seed) < benchmark._MAX_TORCH_SEED


def test_run_benchmark_and_aggregation() -> None:
    results = benchmark.run_benchmark(
        [5],
        [0, 1],
        n_train=8,
        embedding_dim=3,
        random_samples=24,
        num_restarts=2,
        raw_samples=16,
    )
    summaries = benchmark.aggregate_results(results)

    assert len(results) == 6
    assert len(summaries) == 3
    assert {row.strategy for row in summaries} == {"OriginalSpace", "RandomSearch", "REMBO"}
    assert all(row.n_seeds == 2 for row in summaries)


def test_benchmark_validates_inputs() -> None:
    with pytest.raises(ValueError, match="input_dim must be at least 5"):
        benchmark.make_problem(4, n_train=8, seed=0)
    with pytest.raises(ValueError, match="n_train"):
        benchmark.make_problem(5, n_train=1, seed=0)
    bounds = torch.stack([torch.zeros(5, dtype=torch.double), torch.ones(5, dtype=torch.double)])
    with pytest.raises(ValueError, match="embedding_dim"):
        benchmark.strategy_factories(
            bounds,
            embedding_dim=6,
            random_samples=8,
            num_restarts=1,
            raw_samples=4,
            seed=0,
        )
    with pytest.raises(ValueError, match="input_dims"):
        benchmark.run_benchmark([], [0])
    with pytest.raises(ValueError, match="seeds"):
        benchmark.run_benchmark([5], [])
    with pytest.raises(ValueError, match="results"):
        benchmark.aggregate_results([])
