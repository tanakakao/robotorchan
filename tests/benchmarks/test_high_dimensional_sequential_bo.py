"""Tests for the sequential continuous high-dimensional BO benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement

from robotorchan.models import SingleTaskGP
from robotorchan.optim import (
    BAxUSStrategy,
    HeSBOStrategy,
    PCAReconstruction,
    RandomProjectionReconstruction,
    REMBOStrategy,
    TuRBOStrategy,
)

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


def test_every_strategy_uses_original_space_single_task_gp() -> None:
    train_X, train_Y, _ = benchmark.make_initial_data(6, n_train=6, seed=3)
    model = benchmark._fit_model(train_X, train_Y)
    assert isinstance(model, SingleTaskGP)
    assert model.train_inputs[0].shape[-1] == 6


def test_acquisition_is_log_expected_improvement_with_current_best() -> None:
    train_X, train_Y, _ = benchmark.make_initial_data(6, n_train=6, seed=3)
    model = benchmark._fit_model(train_X, train_Y)
    acquisition = benchmark._make_acquisition(model, train_Y)
    assert isinstance(acquisition, LogExpectedImprovement)
    torch.testing.assert_close(acquisition.best_f, train_Y.max())


def test_search_reducers_are_independent_from_surrogate() -> None:
    train_X, _, _ = benchmark.make_initial_data(6, n_train=6, seed=3)
    pca = benchmark._make_latent_reconstruction("LatentPCA", train_X, latent_dim=2)
    rp = benchmark._make_latent_reconstruction("LatentRandomProjection", train_X, latent_dim=2)
    assert isinstance(pca, PCAReconstruction)
    assert isinstance(rp, RandomProjectionReconstruction)
    assert pca.reducer.is_fitted
    assert rp.reducer.is_fitted
    assert pca.reducer.output_dim == 2
    assert rp.reducer.output_dim == 2


def test_stateful_strategies_start_from_initial_observations() -> None:
    train_X, train_Y, bounds = benchmark.make_initial_data(6, n_train=6, seed=3)
    expected_center, expected_best = benchmark._initial_incumbent(train_X, train_Y)
    kwargs = dict(
        reconstruction=None,
        latent_dim=2,
        random_samples=8,
        num_restarts=1,
        raw_samples=4,
        search_seed=17,
        eval_budget=20,
    )
    rembo = benchmark._make_strategy("REMBO", bounds, train_X, train_Y, **kwargs)
    hesbo = benchmark._make_strategy("HeSBO", bounds, train_X, train_Y, **kwargs)
    turbo = benchmark._make_strategy("TuRBO", bounds, train_X, train_Y, **kwargs)
    baxus = benchmark._make_strategy("BAxUS", bounds, train_X, train_Y, **kwargs)
    assert isinstance(rembo, REMBOStrategy)
    assert isinstance(hesbo, HeSBOStrategy)
    assert hesbo.embedding.shape == (6, 2)
    assert isinstance(turbo, TuRBOStrategy)
    assert isinstance(baxus, BAxUSStrategy)
    torch.testing.assert_close(turbo.center, expected_center)
    assert turbo.state.best_value == pytest.approx(expected_best)
    assert baxus.state.best_value == pytest.approx(expected_best)
    assert baxus.state.eval_budget == 20
    assert baxus.target_dim == baxus.state.initial_target_dim


def test_stateful_strategy_feedback_is_persisted() -> None:
    train_X, train_Y, bounds = benchmark.make_initial_data(6, n_train=6, seed=3)
    center, best_value = benchmark._initial_incumbent(train_X, train_Y)
    candidate = torch.full((1, 6), 0.75, dtype=torch.double)
    candidate_Y = benchmark.objective(candidate)
    turbo = TuRBOStrategy(
        bounds,
        center=center,
        state=benchmark.TuRBOState(best_value=best_value),
    )
    baxus = BAxUSStrategy(
        bounds,
        state=benchmark.BAxUSState(dim=6, eval_budget=20, best_value=best_value),
        seed=5,
    )
    target_candidate = torch.zeros((1, baxus.target_dim), dtype=torch.double)
    benchmark._update_stateful_strategy(turbo, candidate, candidate_Y, search_metadata={})
    benchmark._update_stateful_strategy(
        baxus,
        candidate,
        candidate_Y,
        search_metadata={"target_candidates": target_candidate},
    )
    torch.testing.assert_close(turbo.center, candidate.squeeze(0))
    assert turbo.state.best_value == pytest.approx(0.0)
    assert baxus.state.best_value == pytest.approx(0.0)
    torch.testing.assert_close(baxus.target_X, target_candidate)


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


def test_latent_pca_runs_with_original_space_surrogate() -> None:
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


def test_rembo_runs_with_logei() -> None:
    rows = benchmark.run_strategy(
        "REMBO",
        6,
        n_train=6,
        n_iterations=1,
        latent_dim=2,
        random_samples=8,
        num_restarts=1,
        raw_samples=8,
        seed=4,
    )
    assert len(rows) == 1
    assert rows[0].strategy == "REMBO"


def test_hesbo_runs_with_logei() -> None:
    rows = benchmark.run_strategy(
        "HeSBO",
        6,
        n_train=6,
        n_iterations=1,
        latent_dim=2,
        random_samples=8,
        num_restarts=1,
        raw_samples=8,
        seed=4,
    )
    assert len(rows) == 1
    assert rows[0].strategy == "HeSBO"


def test_strategy_names_include_fixed_embedding_methods() -> None:
    assert "REMBO" in benchmark.STRATEGY_NAMES
    assert "HeSBO" in benchmark.STRATEGY_NAMES


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
    with pytest.raises(ValueError, match="Unknown strategy"):
        benchmark.run_strategy("unknown", 6)
    with pytest.raises(ValueError, match="n_iterations"):
        benchmark.run_strategy("RandomSearch", 6, n_iterations=0)
    with pytest.raises(ValueError, match="latent_dim"):
        benchmark.run_strategy("RandomSearch", 6, latent_dim=7)
