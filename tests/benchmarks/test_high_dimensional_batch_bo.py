"""Tests for the joint q-batch high-dimensional BO benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement

from robotorchan.optim import BAxUSThompsonSamplingStrategy

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
        ts_candidates=16,
        num_restarts=1,
        raw_samples=4,
        search_seed=17,
        eval_budget=12,
    )
    assert isinstance(strategy, BAxUSThompsonSamplingStrategy)
    assert strategy.state.eval_budget == 12


def test_random_search_runs_joint_q_batches() -> None:
    rows = benchmark.run_strategy(
        "RandomSearch",
        6,
        q=3,
        n_train=4,
        n_iterations=2,
        random_samples=16,
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


def test_batch_benchmark_rejects_small_ts_pool() -> None:
    with pytest.raises(ValueError, match="ts_candidates must be at least q"):
        benchmark.run_strategy("RandomSearch", 6, q=3, ts_candidates=2)
