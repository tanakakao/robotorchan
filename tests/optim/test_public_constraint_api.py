import torch

from robotorchan.optim import (
    CandidateConstraints,
    LinearConstraint,
    NonlinearConstraint,
    NonlinearConstraintCallable,
)


def test_candidate_constraint_types_are_public_from_optim_namespace() -> None:
    linear: LinearConstraint = (
        torch.tensor([0]),
        torch.tensor([1.0], dtype=torch.double),
        0.2,
    )

    def nonlinear_callable(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.8) - x[0]

    callable_alias: NonlinearConstraintCallable = nonlinear_callable
    nonlinear: NonlinearConstraint = (callable_alias, True)
    constraints = CandidateConstraints(
        inequality_constraints=(linear,),
        nonlinear_inequality_constraints=(nonlinear,),
    )

    assert constraints.inequality_constraints == (linear,)
    assert constraints.nonlinear_inequality_constraints == (nonlinear,)
    assert constraints.has_constraints
