"""Focused validation checks for linear candidate constraints."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_linear_coefficients(value: float) -> None:
    with pytest.raises(ValueError, match="coefficients must be finite"):
        CandidateConstraints(
            inequality_constraints=((torch.tensor([0]), torch.tensor([value]), 0.0),),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_linear_rhs(value: float) -> None:
    with pytest.raises(ValueError, match="rhs must be finite"):
        CandidateConstraints(
            equality_constraints=((torch.tensor([0]), torch.tensor([1.0]), value),),
        )
