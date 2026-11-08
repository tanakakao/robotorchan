"""Regression tests for nonlinear equality constraint normalization."""

from dataclasses import replace

import pytest
import torch

from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation


def test_replacing_nonlinear_equality_does_not_duplicate_expansion() -> None:
    equality = (lambda x: x[0] - 0.5, True)
    constraints = CandidateConstraints(nonlinear_equality_constraints=(equality,))
    replaced = replace(constraints, nonlinear_equality_tolerance=0.1)
    assert len(replaced.nonlinear_inequality_constraints) == 2
    candidate = torch.tensor([[[0.55]]], dtype=torch.double)
    assert candidate_constraint_violation(candidate, replaced).item() == pytest.approx(0.0)


def test_explicit_tolerance_cannot_override_structured_constraints() -> None:
    constraints = CandidateConstraints()
    with pytest.raises(ValueError, match="nonlinear_equality_tolerance"):
        resolve_candidate_constraints(constraints, nonlinear_equality_tolerance=0.01)
