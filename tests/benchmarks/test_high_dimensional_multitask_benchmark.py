import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks" / "high_dimensional_multitask.py"
SPEC = importlib.util.spec_from_file_location(
    "high_dimensional_multitask_benchmark", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_multitask_synthetic_problem_shapes_and_task_column():
    train_X, train_Y, test_X, test_Y = BENCHMARK.make_synthetic_data(
        n_train_per_task=6,
        n_test_per_task=3,
        input_dim=6,
        seed=4,
    )

    assert train_X.shape == (12, 7)
    assert train_Y.shape == (12, 1)
    assert test_X.shape == (6, 7)
    assert test_Y.shape == (6, 1)
    assert set(train_X[:, -1].tolist()) == {0.0, 1.0}


def test_multitask_synthetic_problem_rejects_too_few_dimensions():
    with pytest.raises(ValueError, match="at least 5"):
        BENCHMARK.make_synthetic_data(4, 2, 4)


def test_multitask_factories_cover_linear_reduction_baselines():
    factories = BENCHMARK.model_factories(latent_dim=2)

    assert set(factories) == {
        "MultiTaskGP",
        "PCAMultiTaskGP",
        "PLSMultiTaskGP",
        "RandomProjectionMultiTaskGP",
    }


def test_reduced_multitask_benchmark_smoke():
    train_X, train_Y, test_X, test_Y = BENCHMARK.make_synthetic_data(
        n_train_per_task=5,
        n_test_per_task=2,
        input_dim=6,
        seed=2,
    )
    factories = BENCHMARK.model_factories(latent_dim=2)

    for name in ("PCAMultiTaskGP", "PLSMultiTaskGP", "RandomProjectionMultiTaskGP"):
        model = factories[name](train_X, train_Y)
        posterior = model.posterior(test_X)
        assert posterior.mean.shape == test_Y.shape
        assert torch.isfinite(posterior.mean).all()
