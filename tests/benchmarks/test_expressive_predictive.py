"""Structural tests for the expressive predictive benchmark."""

import importlib.util
import sys
from pathlib import Path

import torch

BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks" / "expressive_predictive.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("expressive_predictive", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load expressive predictive benchmark.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expressive_benchmark_data_are_reproducible() -> None:
    benchmark = _load_benchmark()
    first = benchmark.make_data(n_train=8, n_test=5, seed=3)
    second = benchmark.make_data(n_train=8, n_test=5, seed=3)

    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right)
    assert first[0].shape == (8, 3)
    assert first[1].shape == (8, 1)
    assert first[2].shape == (5, 3)
    assert first[3].shape == (5, 1)


def test_expressive_benchmark_builds_every_declared_model() -> None:
    benchmark = _load_benchmark()
    train_X, train_Y, _, _ = benchmark.make_data(n_train=8, n_test=3, seed=5)

    models = [benchmark.make_model(name, train_X, train_Y) for name in benchmark.MODEL_NAMES]

    assert len(models) == 5
    assert {model.__class__.__name__ for model in models} == {
        "SingleTaskGP",
        "JointEncoderGP",
        "SingleTaskDeepGP",
        "InfiniteWidthBNNGP",
        "SpectralMixtureGP",
    }


def test_expressive_benchmark_rejects_unknown_model() -> None:
    benchmark = _load_benchmark()
    train_X, train_Y, _, _ = benchmark.make_data(n_train=8, n_test=3, seed=7)

    try:
        benchmark.make_model("unknown", train_X, train_Y)
    except ValueError as error:
        assert "Unknown model" in str(error)
    else:
        raise AssertionError("Expected an unknown-model validation error.")
