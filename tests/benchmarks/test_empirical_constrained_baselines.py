"""Validate constrained baseline-only eligibility and feasibility accounting."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.empirical_constrained_baselines import constrained_baseline_metrics
from robotorchan.benchmarks.runner import BenchmarkTrajectory


def test_constrained_quadratic_baseline_eligibility():
    for strategy in ("random", "sobol"):
        cell = ComparativeExperimentCell(
            tier="empirical_standard",
            config=BenchmarkExperimentConfig(
                problem="constrained_quadratic",
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=3,
            ),
        )
        assert cell.config.strategy == strategy
    with pytest.raises(ValueError, match="not applicable"):
        ComparativeExperimentCell(
            tier="empirical_standard",
            config=BenchmarkExperimentConfig(
                problem="constrained_quadratic",
                strategy="qEI",
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=3,
            ),
        )


def test_zero_feasible_does_not_report_infinite_regret():
    X = torch.zeros(70, 2, dtype=torch.double)
    Y = torch.zeros(70, 1, dtype=torch.double)
    run = BenchmarkTrajectory(
        seed=0,
        initial_points=20,
        X=X,
        Y_observed=Y,
        Y_truth=Y,
        constraints=-torch.ones(70, 1, dtype=torch.double),
        costs=torch.ones(70, 1, dtype=torch.double),
    )
    result = constrained_baseline_metrics(run)
    assert result["zero_feasible"] is True
    assert result["feasible_count"] == 0
    assert result["final_feasible_regret"] is None
