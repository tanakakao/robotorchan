"""Constraint contracts for acquisition-function search strategies.

Candidate-space constraints are separate from output constraints used by
constrained acquisition functions. This module represents constraints on
candidate coordinates optimized by a search strategy.

The linear tuple format follows the BoTorch optimizer contract directly:
``(indices, coefficients, rhs)``. This avoids a second robotorchan-specific
constraint language and preserves intra-point and inter-point constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

LinearConstraint = tuple[Tensor, Tensor, float]
NonlinearConstraintCallable = Callable[[Tensor], Tensor]
NonlinearConstraint = tuple[NonlinearConstraintCallable, bool]


@dataclass(frozen=True)
class CandidateConstraints:
    """Candidate-space constraints for compatible search strategies.

    Linear constraints use the BoTorch ``(indices, coefficients, rhs)``
    format. Inequalities mean ``sum(X[indices] * coefficients) >= rhs``.
    One-dimensional indices describe intra-point constraints; two-dimensional
    indices may describe inter-point q-batch constraints.

    These are input constraints, not probabilistic output constraints used by
    constrained BO acquisition functions.

    Nonlinear inequalities use BoTorch's native ``(callable, is_intrapoint)``
    contract. The callable returns a scalar tensor and feasibility means
    ``callable(X) >= 0``. With ``is_intrapoint=True`` it receives ``[d]``;
    otherwise it receives the joint q-batch ``[q, d]``. Inter-point constraints
    therefore require joint q-batch optimization; BoTorch does not support them
    for greedy sequential optimization. The callable must
    preserve device and floating dtype. Gradient-based optimizers additionally
    require differentiability with respect to candidate coordinates;
    derivative-free backends do not. Python callables are runtime objects and have no
    robotorchan-specific serialization format.
    """

    inequality_constraints: tuple[LinearConstraint, ...] = ()
    equality_constraints: tuple[LinearConstraint, ...] = ()
    nonlinear_inequality_constraints: tuple[NonlinearConstraint, ...] = ()

    def __post_init__(self) -> None:
        """Validate constraint structure without evaluating user callables."""
        for name, constraints in (
            ("inequality_constraints", self.inequality_constraints),
            ("equality_constraints", self.equality_constraints),
        ):
            for constraint in constraints:
                _validate_linear_constraint(constraint, name=name)
        for constraint in self.nonlinear_inequality_constraints:
            _validate_nonlinear_constraint(constraint)

    @property
    def has_linear_constraints(self) -> bool:
        """Whether at least one linear candidate constraint is configured."""
        return bool(self.inequality_constraints or self.equality_constraints)

    @property
    def has_nonlinear_constraints(self) -> bool:
        """Whether at least one nonlinear candidate constraint is configured."""
        return bool(self.nonlinear_inequality_constraints)

    @property
    def has_constraints(self) -> bool:
        """Whether at least one candidate-space constraint is configured."""
        return self.has_linear_constraints or self.has_nonlinear_constraints


def reject_unmapped_candidate_constraints(
    constraints: CandidateConstraints | None,
    *,
    strategy_name: str,
) -> None:
    """Reject original-space constraints when a strategy changes search coordinates.

    Embedding and latent strategies optimize coordinates that are not the public
    candidate coordinates. Forwarding public-space linear tuples directly to their
    internal optimizer would therefore impose a different mathematical constraint.
    Strategies may opt into constraints only after implementing an exact mapping
    or a mathematically valid nonlinear composition.
    """
    if constraints is not None and constraints.has_constraints:
        raise NotImplementedError(
            f"{strategy_name} does not map public-space CandidateConstraints into "
            "its internal search coordinates. Use OriginalSpaceStrategy or an "
            "explicitly constraint-aware strategy instead."
        )


def _validate_linear_constraint(constraint: object, *, name: str) -> None:
    if not isinstance(constraint, (tuple, list)) or len(constraint) != 3:
        raise TypeError(f"{name} entries must be (indices, coefficients, rhs) triples.")
    indices, coefficients, rhs = constraint
    if not isinstance(indices, Tensor):
        raise TypeError(f"{name} indices must be a Tensor.")
    if not isinstance(coefficients, Tensor):
        raise TypeError(f"{name} coefficients must be a Tensor.")
    if indices.ndim not in {1, 2}:
        raise ValueError(f"{name} indices must be one- or two-dimensional.")
    if indices.ndim == 2 and indices.shape[-1] != 2:
        raise ValueError(f"{name} inter-point indices must have shape [n_terms, 2].")
    if coefficients.ndim != 1:
        raise ValueError(f"{name} coefficients must be one-dimensional.")
    if indices.shape[0] != coefficients.shape[0]:
        raise ValueError(f"{name} indices and coefficients must contain the same number of terms.")
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise TypeError(f"{name} indices must use an integer dtype.")
    if not coefficients.is_floating_point():
        raise TypeError(f"{name} coefficients must use a floating dtype.")
    if not isinstance(rhs, (int, float)):
        raise TypeError(f"{name} rhs must be a real scalar.")


def _validate_nonlinear_constraint(constraint: object) -> None:
    if not isinstance(constraint, (tuple, list)) or len(constraint) != 2:
        raise TypeError(
            "nonlinear_inequality_constraints entries must be (callable, is_intrapoint) pairs."
        )
    callable_, is_intrapoint = constraint
    if not callable(callable_):
        raise TypeError("Nonlinear candidate constraint must contain a callable.")
    if not isinstance(is_intrapoint, bool):
        raise TypeError("Nonlinear candidate constraint is_intrapoint must be bool.")
