"""Tests for extended benchmark metrics and edge-case contracts."""

import pytest
import torch

from robotorchan.benchmarks.extended_metrics import (
    area_under_curve,
    best_feasible_value_curve,
    first_feasible_evaluation,
    log_loss,
    probability_calibration_error,
)
from robotorchan.benchmarks.runner import BenchmarkTrajectory


def test_area_under_curve_with_nonuniform_budgets() -> None:
    curve = torch.tensor([0.0, 2.0, 2.0], dtype=torch.double)
    budgets = torch.tensor([0.0, 1.0, 3.0], dtype=torch.double)
    torch.testing.assert_close(area_under_curve(curve, budgets), curve.new_tensor(5.0))
    torch.testing.assert_close(area_under_curve(curve), curve.new_tensor(3.0))
    assert area_under_curve(curve[:1]) == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        area_under_curve(curve, torch.tensor([0.0, 1.0, 1.0], dtype=torch.double))


def test_feasible_incumbent_and_first_feasible() -> None:
    truth = torch.tensor([[1.0], [4.0], [3.0], [2.0]], dtype=torch.double)
    constraints = torch.tensor([[-1.0], [0.0], [1.0], [-1.0]], dtype=torch.double)
    trajectory = BenchmarkTrajectory(
        seed=0,
        X=torch.zeros(4, 1, dtype=torch.double),
        Y_observed=truth,
        Y_truth=truth,
        constraints=constraints,
        costs=torch.ones(4, 1, dtype=torch.double),
        initial_points=1,
    )
    curve = best_feasible_value_curve(trajectory)
    assert torch.isnan(curve[0])
    torch.testing.assert_close(curve[1:], truth.new_tensor([4.0, 4.0, 4.0]))
    torch.testing.assert_close(
        best_feasible_value_curve(trajectory, maximize=False)[1:],
        truth.new_tensor([4.0, 3.0, 3.0]),
    )
    assert first_feasible_evaluation(constraints) == 2
    assert first_feasible_evaluation(constraints[:1]) is None


def test_probabilistic_metrics() -> None:
    labels = torch.tensor([0.0, 1.0], dtype=torch.double)
    probabilities = torch.tensor([0.25, 0.75], dtype=torch.double)
    torch.testing.assert_close(log_loss(labels, probabilities), -probabilities[1].log())
    torch.testing.assert_close(
        probability_calibration_error(labels, probabilities, n_bins=2),
        probabilities.new_tensor(0.25),
    )
    torch.testing.assert_close(
        probability_calibration_error(labels, labels, n_bins=2),
        probabilities.new_zeros(()),
    )
    assert torch.isfinite(log_loss(labels, labels))
    with pytest.raises(ValueError, match="Probabilities"):
        log_loss(labels, probabilities + 1)
    with pytest.raises(ValueError, match="positive"):
        probability_calibration_error(labels, probabilities, n_bins=0)
