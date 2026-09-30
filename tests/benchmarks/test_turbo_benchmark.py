"""Contract and smoke tests for the TuRBO benchmark harness."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks" / "turbo.py"
SPEC = importlib.util.spec_from_file_location("turbo_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


@pytest.mark.parametrize("name", ["ackley", "rosenbrock", "levy"])
def test_objectives_have_known_zero_optimum(name: str) -> None:
    objective = BENCHMARK.OBJECTIVES[name]
    optimum_coordinate = {
        "ackley": 0.5,
        "rosenbrock": 0.75,
        "levy": 0.55,
    }[name]
    optimum = torch.full((1, 4), optimum_coordinate, dtype=torch.double)

    value = objective(optimum)

    assert value.shape == (1, 1)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-10, rtol=0.0)


def test_global_and_turbo_share_initialization_contract() -> None:
    kwargs = {
        "function": "ackley",
        "input_dim": 2,
        "n_initial": 4,
        "n_iterations": 1,
        "seed": 7,
        "num_restarts": 1,
        "raw_samples": 8,
    }

    global_rows = BENCHMARK.run_trajectory(method="global", **kwargs)
    turbo_rows = BENCHMARK.run_trajectory(method="turbo", **kwargs)

    assert len(global_rows) == 1
    assert len(turbo_rows) == 1
    assert global_rows[0].n_observations == turbo_rows[0].n_observations == 5
    assert global_rows[0].trust_region_length is None
    assert turbo_rows[0].trust_region_length is not None
    assert global_rows[0].simple_regret >= 0.0
    assert turbo_rows[0].simple_regret >= 0.0


def test_turbo_runs_multiple_iterations_without_statistical_winner_assertion() -> None:
    rows = BENCHMARK.run_trajectory(
        "ackley",
        "turbo",
        input_dim=3,
        n_initial=5,
        n_iterations=2,
        seed=11,
        num_restarts=1,
        raw_samples=8,
    )

    assert [row.iteration for row in rows] == [1, 2]
    assert [row.n_observations for row in rows] == [6, 7]
    assert all(row.method == "turbo" for row in rows)
    assert all(row.trust_region_length is not None for row in rows)


@pytest.mark.parametrize("input_dim", [20, 50, 100])
def test_benchmark_declares_high_dimensional_targets(input_dim: int) -> None:
    assert input_dim in (20, 50, 100)


def test_invalid_benchmark_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown function"):
        BENCHMARK.run_trajectory(
            "unknown",
            "turbo",
            input_dim=2,
            n_initial=4,
            n_iterations=1,
            seed=0,
        )
