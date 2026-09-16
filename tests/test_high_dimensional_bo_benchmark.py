import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "high_dimensional_bo.py"
SPEC = importlib.util.spec_from_file_location("high_dimensional_bo_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_objective_shape_and_dimension_validation():
    X = torch.rand(7, 6, dtype=torch.double)

    assert BENCHMARK.objective(X).shape == (7, 1)
    with pytest.raises(ValueError, match="at least 5"):
        BENCHMARK.objective(torch.rand(3, 4, dtype=torch.double))


def test_core_model_factories_cover_repeated_fit_baselines():
    factories = BENCHMARK.model_factories(latent_dim=2)

    assert set(factories) == {
        "SingleTaskGP",
        "PCAGP",
        "PLSGP",
        "RandomProjectionGP",
    }


def test_select_from_pool_returns_valid_index():
    train_X = torch.rand(8, 6, dtype=torch.double)
    train_Y = BENCHMARK.objective(train_X)
    candidate_X = torch.rand(5, 6, dtype=torch.double)
    model = BENCHMARK.SingleTaskGP(train_X, train_Y)

    index = BENCHMARK.select_from_pool(model, train_Y, candidate_X, mc_samples=4, seed=3)

    assert 0 <= index < 5


def test_single_iteration_bo_smoke():
    generator = torch.Generator().manual_seed(4)
    initial_X = torch.rand(8, 6, generator=generator, dtype=torch.double)
    candidate_X = torch.rand(5, 6, generator=generator, dtype=torch.double)
    candidate_Y = BENCHMARK.objective(candidate_X)
    factory = BENCHMARK.model_factories(latent_dim=2)["SingleTaskGP"]

    results = BENCHMARK.run_model_bo(
        "SingleTaskGP",
        factory,
        initial_X,
        candidate_X,
        candidate_Y,
        n_iterations=1,
        mc_samples=4,
        seed=2,
    )

    assert len(results) == 1
    assert results[0].iteration == 1
    assert results[0].n_observations == 9
    assert results[0].simple_regret >= 0.0


def test_bo_rejects_more_iterations_than_candidates():
    initial_X = torch.rand(8, 6, dtype=torch.double)
    candidate_X = torch.rand(2, 6, dtype=torch.double)
    candidate_Y = BENCHMARK.objective(candidate_X)
    factory = BENCHMARK.model_factories(latent_dim=2)["SingleTaskGP"]

    with pytest.raises(ValueError, match="pool size"):
        BENCHMARK.run_model_bo(
            "SingleTaskGP",
            factory,
            initial_X,
            candidate_X,
            candidate_Y,
            n_iterations=3,
            mc_samples=4,
            seed=2,
        )
