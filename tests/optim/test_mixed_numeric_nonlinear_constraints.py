"""Explicit numeric-coordinate nonlinear constraints in mixed spaces."""

import pytest
import torch

from robotorchan.optim.constraints import CandidateConstraints, NumericOnlyConstraint
from robotorchan.optim.constraints.validation import validate_mixed_constraint_dimensions
from robotorchan.optim.domains import MixedVariableSpace


def test_numeric_only_nonlinear_constraint_excludes_category() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]])
    space = MixedVariableSpace(bounds, integer_dims=(1,), categorical_values={2: (0.0, 2.0)})
    function = NumericOnlyConstraint(lambda x: x[..., 0] + x[..., 1] - 1, (0, 1))
    constraints = CandidateConstraints(nonlinear_inequality_constraints=((function, True),))
    validate_mixed_constraint_dimensions(constraints, space)
    assert function(torch.tensor([0.5, 2.0, 2.0])).item() == pytest.approx(1.5)


def test_mixed_nonlinear_constraint_requires_numeric_projection() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 2.0)})
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] - 0.5, True),)
    )
    with pytest.raises(ValueError, match="require NumericOnlyConstraint"):
        validate_mixed_constraint_dimensions(constraints, space)
    function = NumericOnlyConstraint(lambda x: x[0], (1,))
    constraints = CandidateConstraints(nonlinear_equality_constraints=((function, True),))
    with pytest.raises(ValueError, match="exclude categorical"):
        validate_mixed_constraint_dimensions(constraints, space)
