"""Tests for renderer-independent benchmark plot specifications."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.comparative_efficiency import BenchmarkEfficiency
from robotorchan.benchmarks.comparative_single_objective import SingleObjectiveComparison
from robotorchan.benchmarks.comparative_visualization import (
    plot_efficiency,
    plot_single_objective,
)


def test_single_objective_plot_has_mean_and_error_band() -> None:
    comparison = SingleObjectiveComparison(
        seeds=(0, 1),
        q=1,
        evaluations=torch.tensor([0, 1, 2]),
        regret_by_method={},
        mean_regret_by_method={"random": torch.tensor([3.0, 2.0, 1.0])},
        standard_error_by_method={"random": torch.tensor([0.2, 0.1, 0.1])},
    )
    plot = plot_single_objective(comparison)
    assert plot.x_label == "Evaluations"
    assert plot.series[0].x == (0.0, 1.0, 2.0)
    assert plot.series[0].lower == pytest.approx((2.8, 1.9, 0.9))
    assert plot.series[0].upper == pytest.approx((3.2, 2.1, 1.1))


def test_efficiency_plot_preserves_cost_axis() -> None:
    values = torch.tensor([[1.0, 3.0], [1.0, 5.0]])
    efficiency = BenchmarkEfficiency(
        seeds=(0, 1),
        evaluations=torch.tensor([0, 2]),
        candidate_seconds=values,
        evaluation_seconds=values,
        total_seconds=values * 2,
        evaluation_cost=values * 10,
    )
    plot = plot_efficiency({"random": efficiency}, metric="evaluation_cost")
    assert plot.y_label == "evaluation_cost"
    assert plot.series[0].y == (10.0, 40.0)


def test_invalid_efficiency_data_rejected() -> None:
    values = torch.tensor([[1.0, 2.0]])
    efficiency = BenchmarkEfficiency(
        seeds=(0,),
        evaluations=torch.tensor([0, 1]),
        candidate_seconds=values,
        evaluation_seconds=values,
        total_seconds=values,
        evaluation_cost=values,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        plot_efficiency({"random": replace(efficiency, total_seconds=-values)})
