"""Focused tests for posterior safety confidence bounds."""

import pytest
import torch

from robotorchan.semantics import ConstraintDirection, ContinuousConstraint
from robotorchan.semantics.safety_bounds import safety_confidence_bounds


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _Model:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.result = _Posterior(mean, variance)

    def posterior(self, X: torch.Tensor) -> _Posterior:
        return self.result


def test_multiple_constraints_and_batched_candidates() -> None:
    mean = torch.tensor([[[1.0, 2.0], [0.0, 3.0]]], dtype=torch.double)
    variance = torch.tensor([[[0.25, 0.25], [0.0, 0.0]]], dtype=torch.double)
    constraints = (
        ContinuousConstraint(0, 0.0, ConstraintDirection.GREATER_THAN_OR_EQUAL),
        ContinuousConstraint(1, 2.5),
    )
    margins, safe = safety_confidence_bounds(
        _Model(mean, variance), torch.zeros(1, 2, 1), constraints
    )
    torch.testing.assert_close(
        margins, torch.tensor([[[0.0, -0.5], [0.0, -0.5]]], dtype=torch.double)
    )
    assert safe.tolist() == [[False, False]]


def test_zero_variance_and_boundary_are_safe() -> None:
    model = _Model(torch.tensor([[0.5]]), torch.tensor([[0.0]]))
    margins, safe = safety_confidence_bounds(
        model, torch.zeros(1, 1), (ContinuousConstraint(0, 0.5),)
    )
    assert margins.item() == 0.0
    assert safe.item()


def test_tolerance_accepts_near_boundary() -> None:
    model = _Model(torch.tensor([[0.50001]]), torch.zeros(1, 1))
    constraint = (ContinuousConstraint(0, 0.5),)
    _, strict = safety_confidence_bounds(model, torch.zeros(1, 1), constraint)
    _, relaxed = safety_confidence_bounds(
        model, torch.zeros(1, 1), constraint, tolerance=0.0001
    )
    assert not strict.item()
    assert relaxed.item()


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan")])
def test_invalid_beta(beta: float) -> None:
    model = _Model(torch.zeros(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError, match="beta"):
        safety_confidence_bounds(
            model, torch.zeros(1, 1), (ContinuousConstraint(0, 0.0),), beta=beta
        )


def test_negative_variance_rejected() -> None:
    model = _Model(torch.zeros(1, 1), -torch.ones(1, 1))
    with pytest.raises(ValueError, match="variance"):
        safety_confidence_bounds(
            model, torch.zeros(1, 1), (ContinuousConstraint(0, 0.0),)
        )
