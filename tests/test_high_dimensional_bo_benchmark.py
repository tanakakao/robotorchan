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


def test_extended_model_specs_cover_neural_joint_and_map_saas():
    specs = BENCHMARK.model_specs(latent_dim=2, neural_epochs=1, include_extended=True)

    assert {
        "AutoEncoderGP",
        "VAEGP",
        "SupervisedAutoEncoderGP",
        "SupervisedVAEGP",
        "JointEncoderGP",
        "HybridAutoEncoderGP",
        "JointVAEGP",
        "AdditiveMapSaasSingleTaskGP",
    } <= set(specs)
    assert specs["JointEncoderGP"].fit_policy == "joint"
    assert specs["HybridAutoEncoderGP"].fit_policy == "hybrid"
    assert specs["JointVAEGP"].fit_policy == "joint_vae"
    assert specs["AdditiveMapSaasSingleTaskGP"].fit_policy == "mll"


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
    assert results[0].seed == 2


def test_joint_encoder_fit_policy_smoke():
    train_X = torch.rand(8, 6, dtype=torch.double)
    train_Y = BENCHMARK.objective(train_X)
    spec = BENCHMARK.model_specs(2, neural_epochs=1, include_extended=True)["JointEncoderGP"]
    model = spec.factory(train_X, train_Y)

    BENCHMARK.fit_model(
        model,
        fit_policy=spec.fit_policy,
        joint_steps=1,
        joint_learning_rate=1e-2,
    )

    posterior = model.posterior(train_X[:2])
    assert posterior.mean.shape == (2, 1)


def test_aggregate_results_computes_population_statistics():
    rows = [
        BENCHMARK.BOIterationResult("GP", 1, 9, 1.0, 0.4, 0.2, seed=0),
        BENCHMARK.BOIterationResult("GP", 1, 9, 1.4, 0.2, 0.3, seed=1),
    ]

    summary = BENCHMARK.aggregate_results(rows)[0]

    assert summary.model == "GP"
    assert summary.iteration == 1
    assert summary.n_seeds == 2
    assert summary.simple_regret_mean == pytest.approx(0.3)
    assert summary.simple_regret_std == pytest.approx(0.1)
    assert summary.simple_regret_q25 == pytest.approx(0.25)
    assert summary.simple_regret_q75 == pytest.approx(0.35)
    assert summary.best_observed_mean == pytest.approx(1.2)
    assert summary.best_observed_std == pytest.approx(0.2)


def test_parse_seeds():
    assert BENCHMARK.parse_seeds("0, 2,5") == [0, 2, 5]
    with pytest.raises(Exception, match="at least one seed"):
        BENCHMARK.parse_seeds(" , ")


def test_repeated_benchmark_rejects_empty_seeds():
    with pytest.raises(ValueError, match="at least one"):
        BENCHMARK.run_repeated_benchmark([])


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
