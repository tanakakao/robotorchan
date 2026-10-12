"""Ensure mixed empirical baselines preserve discrete domain contracts."""

import pytest
import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


@pytest.mark.parametrize("problem", ["mixed_quadratic", "categorical_switch"])
@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_mixed_smoke_baseline_candidates_are_valid(problem, strategy):
    cell = ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )
    run = run_comparative_baseline(cell)[0]
    assert run.X.shape[0] == 8
    assert torch.isfinite(run.Y_truth).all()
    assert torch.equal(run.X[:, -1], run.X[:, -1].round())


def test_mixed_gp_comparison_remains_unavailable():
    with pytest.raises(ValueError, match="not applicable"):
        ComparativeExperimentCell(
            tier="smoke",
            config=BenchmarkExperimentConfig(
                problem="mixed_quadratic",
                strategy="qEI",
                seeds=(0,),
                initial_points=4,
                evaluation_budget=4,
                q=1,
            ),
        )
