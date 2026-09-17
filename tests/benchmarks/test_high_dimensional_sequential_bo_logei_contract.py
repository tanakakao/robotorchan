"""Contract tests for the sequential high-dimensional BO acquisition."""

import importlib.util
import sys
from pathlib import Path

import torch
from botorch.acquisition.analytic import LogExpectedImprovement

MODULE_NAME = "high_dimensional_sequential_bo_logei_contract"
MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "high_dimensional_sequential_bo.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def test_logei_best_f_tracks_new_observations() -> None:
    train_X, train_Y, _ = benchmark.make_initial_data(6, n_train=6, seed=11)
    model = benchmark._fit_model(train_X, train_Y)
    first = benchmark._make_acquisition(model, train_Y)

    improved_Y = torch.cat([train_Y, train_Y.max().reshape(1, 1) + 0.25], dim=0)
    second = benchmark._make_acquisition(model, improved_Y)

    assert isinstance(first, LogExpectedImprovement)
    assert isinstance(second, LogExpectedImprovement)
    torch.testing.assert_close(first.best_f, train_Y.max())
    torch.testing.assert_close(second.best_f, improved_Y.max())
    assert second.best_f > first.best_f
