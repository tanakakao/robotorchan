"""Tests for the observed-seed safety interface."""

import pytest
import torch

from robotorchan.semantics.direction import ConstraintDirection
from robotorchan.semantics.safety import SafetySpecification


def test_safe_seeds_support_both_directions_and_boundary() -> None:
    from robotorchan.semantics.constraints import ContinuousConstraint

    constraints = (
        ContinuousConstraint(0, 0.5),
        ContinuousConstraint(1, 1.0, ConstraintDirection.GREATER_THAN_OR_EQUAL),
    )
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.5, 1.0], [0.2, 1.5]], dtype=torch.double)
    specification = SafetySpecification(constraints, X, Y)
    assert specification.beta == 4.0


def test_unsafe_seed_is_rejected() -> None:
    from robotorchan.semantics.constraints import ContinuousConstraint

    with pytest.raises(ValueError, match="unsafe seed"):
        SafetySpecification(
            (ContinuousConstraint(0, 0.5),),
            torch.tensor([[0.0]]),
            torch.tensor([[0.6]]),
        )


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_beta_is_rejected(beta: float) -> None:
    from robotorchan.semantics.constraints import ContinuousConstraint

    with pytest.raises(ValueError, match="beta"):
        SafetySpecification(
            (ContinuousConstraint(0, 0.5),),
            torch.tensor([[0.0]]),
            torch.tensor([[0.4]]),
            beta=beta,
        )


def test_nonfinite_seed_is_rejected() -> None:
    from robotorchan.semantics.constraints import ContinuousConstraint

    with pytest.raises(ValueError, match="finite"):
        SafetySpecification(
            (ContinuousConstraint(0, 0.5),),
            torch.tensor([[0.0]]),
            torch.tensor([[float("nan")]]),
        )
