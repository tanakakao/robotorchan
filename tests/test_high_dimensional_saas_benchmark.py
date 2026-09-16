import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "high_dimensional_saas.py"
SPEC = importlib.util.spec_from_file_location("high_dimensional_saas_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_sparse_synthetic_problem_shapes():
    train_X, train_Y, test_X, test_Y = BENCHMARK.make_sparse_synthetic_data(
        n_train=12,
        n_test=7,
        input_dim=10,
        seed=3,
    )

    assert train_X.shape == (12, 10)
    assert train_Y.shape == (12, 1)
    assert test_X.shape == (7, 10)
    assert test_Y.shape == (7, 1)
    assert train_X.dtype == torch.double


def test_sparse_synthetic_problem_requires_five_dimensions():
    with pytest.raises(ValueError, match="at least 5"):
        BENCHMARK.make_sparse_synthetic_data(8, 4, 4)


def test_map_saas_factories_cover_wrappers():
    factories = BENCHMARK.map_saas_factories(num_taus=2)

    assert set(factories) == {
        "AdditiveMapSaasSingleTaskGP",
        "EnsembleMapSaasSingleTaskGP",
    }


def test_map_saas_factory_retains_raw_training_data():
    train_X, train_Y, _, _ = BENCHMARK.make_sparse_synthetic_data(10, 4, 8, seed=5)
    factory = BENCHMARK.map_saas_factories(num_taus=2)["AdditiveMapSaasSingleTaskGP"]
    model = factory(train_X, train_Y)

    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert model.supports_mll


def test_fully_bayesian_wrapper_declares_non_mll_contract_without_optional_dependencies():
    assert not BENCHMARK.SaasFullyBayesianSingleTaskGP.supports_mll
