"""Tests for the joint q-batch high-dimensional BO benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement

from robotorchan.optim import BAxUSThompsonSamplingStrategy, HeSBOStrategy, REMBOStrategy

MODULE_NAME = "high_dimensional_batch_bo"
MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / f"{MODULE_NAME}.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def test_batch_acquisition_is_qlogei_with_current_best() -> None:
    train_X, train_Y, _ = benchmark.make_initial_data(6, n_train=6, seed=3)
    model = benchmark._fit_model(train_X, train_Y)
    acquisition = benchmark._make_acquisition(model, train_Y)
    assert isinstance(acquisition, qLogExpectedImprovement)
    torch.testing.assert_close(acquisition.best_f, train_Y.max())


def test_baxus_ts_strategy_is_available_in_batch_benchmark() -> None:
    train_X, train_Y, bounds = benchmark.make_initial_data(6, n_train=6, seed=3)
    strategy = benchmark._make_strategy(
        "BAxUSTS",
        bounds,
        train_X,
        train_Y,
        random_samples=8,
        embedding_dim=2,
        ts_candidates=16,
        num_restarts=1,
        raw_samples=4,
        search_seed=17,
        eval_budget=12,
    )
    assert isinstance(strategy, BAxUSThompsonSamplingStrategy)
    assert strategy.state.eval_budget == 12
    assert strategy.n_candidates == 16


def test_baxus_ts_batch_benchmark_uses_automatic_candidate_budget() -> None:
    train_X, train_Y, bounds = benchmark.make_initial_data(6, n_train=6, seed=3)
    strategy = benchmark._make_strategy(
        "BAxUSTS",
        bounds,
        train_X,
        train_Y,
        random_samples=8,
        embedding_dim=2,
        ts_candidates=None,
        num_restarts=1,
        raw_samples=4,
        search_seed=17,
        eval_budget=12,
    )

    assert isinstance(strategy, BAxUSThompsonSamplingStrategy)
    assert strategy.target_dim == strategy.state.initial_target_dim
    assert strategy.n_candidates == 2000


def test_fixed_embedding_strategies_are_available_in_batch_benchmark() -> None:
    train_X, train_Y, bounds = benchmark.make_initial_data(6, n_train=6, seed=3)
    kwargs = dict(
        random_samples=8,
        embedding_dim=2,
        ts_candidates=16,
        num_restarts=1,
        raw_samples=8,
        search_seed=17,
        eval_budget=12,
    )

    rembo = benchmark._make_strategy("REMBO", bounds, train_X, train_Y, **kwargs)
    hesbo = benchmark._make_strategy("HeSBO", bounds, train_X, train_Y, **kwargs)

    assert isinstance(rembo, REMBOStrategy)
    assert isinstance(hesbo, HeSBOStrategy)
    assert rembo.embedding.shape == (6, 2)
    assert hesbo.embedding.shape == (6, 2)


def test_hesbo_runs_joint_q_batch_with_qlogei() -> None:
    rows = benchmark.run_strategy(
        "HeSBO",
        6,
        q=2,
        n_train=6,
        n_iterations=1,
        random_samples=8,
        embedding_dim=2,
        ts_candidates=16,
        num_restarts=1,
        raw_samples=8,
        seed=4,
    )
    assert len(rows) == 1
    assert rows[0].strategy == "HeSBO"
    assert rows[0].q == 2


def test_random_search_runs_joint_q_batches() -> None:
    rows = benchmark.run_strategy(
        "RandomSearch",
        6,
        q=3,
        n_train=4,
        n_iterations=2,
        random_samples=16,
        embedding_dim=2,
        ts_candidates=16,
        num_restarts=1,
        raw_samples=4,
        seed=1,
    )
    assert len(rows) == 2
    assert all(row.q == 3 for row in rows)
    assert rows[1].best_observed >= rows[0].best_observed
    assert rows[1].simple_regret <= rows[0].simple_regret


def test_baxus_state_accepts_all_batch_feedback() -> None:
    train_X, train_Y, bounds = benchmark.make_initial_data(6, n_train=6, seed=3)
    strategy = benchmark._make_strategy(
        "BAxUS",
        bounds,
        train_X,
        train_Y,
        random_samples=8,
        embedding_dim=2,
        ts_candidates=16,
        num_restarts=1,
        raw_samples=4,
        search_seed=17,
        eval_budget=12,
    )
    q = 3
    target_candidates = torch.zeros(q, strategy.target_dim, dtype=torch.double)
    candidates = strategy.project(target_candidates)
    values = benchmark.objective(candidates)
    benchmark._update_stateful_strategy(
        strategy,
        candidates,
        values,
        search_metadata={"target_candidates": target_candidates},
    )
    assert strategy.target_X.shape[0] == q
    assert strategy.target_Y.shape[0] == q


def test_batch_benchmark_rejects_non_batch_q() -> None:
    with pytest.raises(ValueError, match="q must be at least 2"):
        benchmark.run_strategy("RandomSearch", 6, q=1)


def test_batch_benchmark_rejects_small_explicit_ts_pool() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        benchmark.run_strategy("HeSBO", 6, q=2, embedding_dim=7)
    with pytest.raises(ValueError, match="ts_candidates must be at least q"):
        benchmark.run_strategy("RandomSearch", 6, q=3, ts_candidates=2)



def test_run_benchmark_uses_same_problem_grid_for_each_strategy() -> None:
    rows = benchmark.run_benchmark(
        ["RandomSearch", "HeSBO"],
        [6],
        [0, 1],
        q=2,
        n_train=4,
        n_iterations=1,
        random_samples=8,
        embedding_dim=2,
        ts_candidates=8,
        num_restarts=1,
        raw_samples=4,
    )

    assert len(rows) == 4
    assert {(row.strategy, row.seed) for row in rows} == {
        ("RandomSearch", 0),
        ("RandomSearch", 1),
        ("HeSBO", 0),
        ("HeSBO", 1),
    }
    assert all(row.input_dim == 6 for row in rows)
    assert all(row.q == 2 for row in rows)


def test_batch_aggregation_keeps_q_and_iteration_separate() -> None:
    rows = [
        benchmark.BatchBOResult("RandomSearch", 6, 0, 1, 2, -0.5, -0.4, 0.4, 0.1),
        benchmark.BatchBOResult("RandomSearch", 6, 1, 1, 2, -0.3, -0.2, 0.2, 0.2),
        benchmark.BatchBOResult("RandomSearch", 6, 0, 2, 2, -0.2, -0.1, 0.1, 0.3),
    ]

    summaries = benchmark.aggregate_results(rows)

    assert len(summaries) == 2
    first = summaries[0]
    assert first.iteration == 1
    assert first.q == 2
    assert first.n_seeds == 2
    assert first.batch_best_mean == pytest.approx(-0.4)
    assert first.simple_regret_mean == pytest.approx(0.3)


def test_batch_benchmark_grid_validates_empty_inputs() -> None:
    with pytest.raises(ValueError, match="strategy_names"):
        benchmark.run_benchmark([], [6], [0])
    with pytest.raises(ValueError, match="input_dims"):
        benchmark.run_benchmark(["RandomSearch"], [], [0])
    with pytest.raises(ValueError, match="seeds"):
        benchmark.run_benchmark(["RandomSearch"], [6], [])
    with pytest.raises(ValueError, match="Unknown strategies"):
        benchmark.run_benchmark(["unknown"], [6], [0])
    with pytest.raises(ValueError, match="results"):
        benchmark.aggregate_results([])
