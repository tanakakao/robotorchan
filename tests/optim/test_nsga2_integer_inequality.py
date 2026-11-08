"""Mixed NSGA-II accepts integer linear inequalities without relaxing equalities."""

import pytest
import torch

from robotorchan.optim.backends.nsga2 import _resolve_nsga2_constraints
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.domains import MixedVariableSpace


def test_mixed_nsga2_integer_inequality_and_equality_contract() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]])
    space = MixedVariableSpace(bounds, integer_dims=(1,), categorical_values={2: (0.0, 2.0)})
    linear = ((torch.tensor([1]), torch.tensor([1.0]), 1.0),)
    kwargs = dict(
        inequality_constraints=None,
        equality_constraints=None,
        nonlinear_inequality_constraints=None,
        nonlinear_equality_constraints=None,
        nonlinear_equality_tolerance=1e-6,
        bounds=bounds,
        variable_space=space,
    )
    constraints = CandidateConstraints(inequality_constraints=linear)
    assert _resolve_nsga2_constraints(constraints, **kwargs) is constraints

    constraints = CandidateConstraints(equality_constraints=linear)
    with pytest.raises(ValueError, match="equalities require continuous"):
        _resolve_nsga2_constraints(constraints, **kwargs)
