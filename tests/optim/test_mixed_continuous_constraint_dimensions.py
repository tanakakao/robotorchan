"""Mixed-space linear constraints apply only to continuous dimensions."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.constraints.validation import validate_mixed_constraint_dimensions
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def test_reject_categorical_linear_constraint_dimensions() -> None:
    space = MixedVariableSpace(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]),
        integer_dims=(1,),
        categorical_values={2: (0.0, 1.0, 2.0)},
    )
    for dim in (2,):
        constraint = (torch.tensor([dim]), torch.tensor([1.0]), 0.5)
        with pytest.raises(ValueError, match="categorical dimensions"):
            validate_mixed_constraint_dimensions(
                CandidateConstraints(inequality_constraints=(constraint,)), space
            )


def test_accept_numeric_interpoint_linear_constraint() -> None:
    space = MixedVariableSpace(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        categorical_values={1: (0.0, 1.0)},
    )
    constraint = (torch.tensor([[0, 0], [1, 0]]), torch.tensor([1.0, 1.0]), 0.5)
    validate_mixed_constraint_dimensions(
        CandidateConstraints(equality_constraints=(constraint,)), space
    )
