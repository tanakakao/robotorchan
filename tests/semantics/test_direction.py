"""Tests for optimization direction and sign semantics."""

import pytest
import torch

from robotorchan.semantics import ConstraintDirection, ObjectiveDirection


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (ObjectiveDirection.MAXIMIZE, [1.0, 2.0]),
        (ObjectiveDirection.MINIMIZE, [-1.0, -2.0]),
    ],
)
def test_objective_direction_normalizes_to_maximization(
    direction: ObjectiveDirection,
    expected: list[float],
) -> None:
    values = torch.tensor([1.0, 2.0])

    assert torch.equal(direction.apply(values), torch.tensor(expected))


def test_objective_direction_exposes_sign() -> None:
    assert ObjectiveDirection.MAXIMIZE.sign == 1
    assert ObjectiveDirection.MINIMIZE.sign == -1


@pytest.mark.parametrize(
    ("direction", "values", "threshold", "expected"),
    [
        (
            ConstraintDirection.LESS_THAN_OR_EQUAL,
            [80.0, 100.0, 120.0],
            100.0,
            [-20.0, 0.0, 20.0],
        ),
        (
            ConstraintDirection.GREATER_THAN_OR_EQUAL,
            [80.0, 100.0, 120.0],
            100.0,
            [20.0, 0.0, -20.0],
        ),
    ],
)
def test_constraint_direction_uses_nonpositive_feasibility(
    direction: ConstraintDirection,
    values: list[float],
    threshold: float,
    expected: list[float],
) -> None:
    residual = direction.residual(torch.tensor(values), threshold)

    assert torch.equal(residual, torch.tensor(expected))


def test_constraint_threshold_preserves_tensor_dtype_and_device() -> None:
    values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    threshold = torch.tensor(1.5, dtype=torch.float64)

    residual = ConstraintDirection.LESS_THAN_OR_EQUAL.residual(values, threshold)

    assert residual.dtype == torch.float64
    assert residual.device == values.device
