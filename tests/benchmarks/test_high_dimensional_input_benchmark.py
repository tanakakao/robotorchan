import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks" / "high_dimensional_inputs.py"
SPEC = importlib.util.spec_from_file_location("high_dimensional_inputs_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_synthetic_high_dimensional_problem_shapes():
    train_X, train_Y, test_X, test_Y = BENCHMARK.make_synthetic_data(
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


def test_gaussian_nll_is_finite():
    mean = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    variance = torch.tensor([[0.5], [0.25]], dtype=torch.double)
    target = torch.tensor([[0.1], [0.8]], dtype=torch.double)

    value = BENCHMARK.gaussian_nll(mean, variance, target)

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_model_factories_cover_initial_phase11_comparison_set():
    factories = BENCHMARK.model_factories(latent_dim=2, neural_epochs=1)

    assert set(factories) == {
        "SingleTaskGP",
        "PCAGP",
        "PLSGP",
        "RandomProjectionGP",
        "AutoEncoderGP",
        "VAEGP",
        "SupervisedAutoEncoderGP",
        "SupervisedVAEGP",
    }


def test_joint_model_factories_cover_joint_representation_models():
    factories = BENCHMARK.joint_model_factories(latent_dim=2)

    assert set(factories) == {
        "JointEncoderGP",
        "HybridAutoEncoderGP",
        "JointVAEGP",
    }


def test_acquisition_evaluation_returns_finite_metrics():
    train_X, train_Y, _, _ = BENCHMARK.make_synthetic_data(10, 4, 6, seed=5)
    model = BENCHMARK.SingleTaskGP(train_X, train_Y)
    candidate_X = torch.rand(8, 6, dtype=torch.double)

    elapsed, value = BENCHMARK.evaluate_acquisition(
        model,
        train_Y,
        candidate_X,
        mc_samples=4,
        seed=3,
    )

    assert elapsed >= 0.0
    assert math_is_finite(value)


def test_acquisition_evaluation_rejects_zero_mc_samples():
    train_X, train_Y, _, _ = BENCHMARK.make_synthetic_data(10, 4, 6, seed=5)
    model = BENCHMARK.SingleTaskGP(train_X, train_Y)
    candidate_X = torch.rand(8, 6, dtype=torch.double)

    with pytest.raises(ValueError, match="mc_samples"):
        BENCHMARK.evaluate_acquisition(model, train_Y, candidate_X, mc_samples=0)


def test_joint_model_training_smoke():
    train_X, train_Y, test_X, test_Y = BENCHMARK.make_synthetic_data(
        n_train=10,
        n_test=4,
        input_dim=6,
        seed=7,
    )
    factory = BENCHMARK.joint_model_factories(latent_dim=2)["JointEncoderGP"]
    model = factory(train_X, train_Y)
    candidate_X = torch.rand(6, 6, dtype=torch.double)

    BENCHMARK._fit_joint_model(model, steps=1, learning_rate=1e-2)
    result = BENCHMARK._evaluate_model(
        "JointEncoderGP",
        model,
        train_Y,
        test_X,
        test_Y,
        candidate_X,
        train_seconds=0.0,
        acquisition_mc_samples=4,
        seed=7,
    )

    assert result.model == "JointEncoderGP"
    assert math_is_finite(result.rmse)
    assert math_is_finite(result.nll)
    assert result.acquisition_seconds >= 0.0
    assert math_is_finite(result.acquisition_value)


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
