"""BoTorch benchmark strategy integration tests."""

import pytest
import torch

from robotorchan.benchmarks.botorch_strategy import (
    botorch_gp_candidates,
    make_botorch_gp_strategy,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import run_benchmark


def _problem() -> BenchmarkProblem:
    return BenchmarkProblem(
        name="quadratic",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=lambda X: (X - 0.3).square(),
        directions=("minimize",),
        variable_types=("continuous",),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


@pytest.mark.parametrize("acquisition", ["qEI", "qNEI"])
def test_botorch_strategy_end_to_end(acquisition: str) -> None:
    registry = BenchmarkProblemRegistry()
    registry.register("quadratic", _problem)
    config = BenchmarkExperimentConfig(
        problem="quadratic",
        strategy=acquisition,
        seeds=(3,),
        initial_points=4,
        evaluation_budget=1,
        q=1,
    )
    strategy = make_botorch_gp_strategy(acquisition=acquisition, num_restarts=1, raw_samples=8)
    result = run_benchmark(config, strategy, registry=registry)[0]
    assert result.X.shape == (5, 1)
    assert result.Y_observed.shape == (5, 1)
    assert result.evaluation_count == 1
    assert torch.isfinite(result.X).all()


def test_strategy_rejects_invalid_acquisition() -> None:
    problem = _problem()
    X = torch.tensor([[0.1], [0.3], [0.6]], dtype=torch.double)
    Y = problem.evaluate_observation(X)
    with pytest.raises(ValueError, match="acquisition"):
        botorch_gp_candidates(
            problem, X, Y, 1, torch.Generator().manual_seed(0), acquisition="unknown"
        )
