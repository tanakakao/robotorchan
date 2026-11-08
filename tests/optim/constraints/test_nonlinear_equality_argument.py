"""Focused tests for the public nonlinear equality contract."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation


def test_nonlinear_equality_is_converted_to_inequalities() -> None:
    constraints = resolve_candidate_constraints(
        nonlinear_equality_constraints=((lambda x: x[0] - 0.5, True),),
        nonlinear_equality_tolerance=0.05,
    )
    assert len(constraints.nonlinear_inequality_constraints) == 2
    candidates = torch.tensor([[[0.52]], [[0.7]]], dtype=torch.double)
    violation = candidate_constraint_violation(candidates, constraints)
    torch.testing.assert_close(violation[0], torch.zeros_like(violation[0]))
    assert violation[1] > 0


def test_interpoint_equality_contract() -> None:
    constraints = CandidateConstraints(
        nonlinear_equality_constraints=((lambda x: x[:, 0].sum() - 1, False),),
        nonlinear_equality_tolerance=0.01,
    )
    candidates = torch.tensor([[[0.4], [0.6]], [[0.4], [0.8]]], dtype=torch.double)
    violation = candidate_constraint_violation(candidates, constraints)
    torch.testing.assert_close(violation[0], torch.zeros_like(violation[0]))
    assert violation[1] > 0


def test_nonlinear_equality_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="nonlinear_equality_tolerance"):
        CandidateConstraints(nonlinear_equality_tolerance=-1)
