"""Phase 13 continuous constraint benchmark smoke tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_continuous_constraints import (
    run_continuous_constraint_baseline,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


@pytest.mark.parametrize(
    "problem",
    [
        "constrained_annulus",
        "constrained_disconnected",
        "constrained_narrow_band",
        "constrained_quadratic",
    ],
)
@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_continuous_constraint_baselines(problem: str, strategy: str) -> None:
    config = BenchmarkExperimentConfig(
        problem=problem,
        strategy=strategy,
        seeds=(0, 1),
        initial_points=4,
        evaluation_budget=5,
        q=3,
    )
    result = run_continuous_constraint_baseline(config)
    assert result.evaluations.tolist() == [0, 3, 5]
    assert result.regret.shape == (2, 3)
    assert result.feasibility_rate.shape == (2, 3)
    assert ((result.feasibility_rate >= 0) & (result.feasibility_rate <= 1)).all()
    assert not torch.isnan(result.regret).any()
    assert (result.regret[:, 1:] <= result.regret[:, :-1]).all()
    assert all(run.evaluation_count == 5 for run in result.trajectories)


def test_continuous_constraint_baseline_reproducible() -> None:
    config = BenchmarkExperimentConfig(
        problem="constrained_quadratic",
        strategy="sobol",
        seeds=(0,),
        initial_points=4,
        evaluation_budget=4,
        q=1,
    )
    first = run_continuous_constraint_baseline(config)
    second = run_continuous_constraint_baseline(config)
    torch.testing.assert_close(first.trajectories[0].X, second.trajectories[0].X)
    torch.testing.assert_close(first.feasibility_rate, second.feasibility_rate)


def test_continuous_constraint_baseline_rejects_unsupported_strategy() -> None:
    config = BenchmarkExperimentConfig(
        problem="constrained_quadratic",
        strategy="qEI",
    )
    with pytest.raises(ValueError, match="random or sobol"):
        run_continuous_constraint_baseline(config)


def test_continuous_constraint_baseline_rejects_unknown_problem() -> None:
    config = BenchmarkExperimentConfig(problem="branin", strategy="random")
    with pytest.raises(ValueError, match="Unknown continuous constraint"):
        run_continuous_constraint_baseline(config)
