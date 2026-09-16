import importlib.util
import sys
from pathlib import Path

import torch

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "high_dimensional_inputs.py"
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
