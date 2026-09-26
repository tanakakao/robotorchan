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

    These are input constraints, not probabilistic output constraints such as
    ``g(x) <= 0`` used by constrained BO acquisition functions.

    Nonlinear inequalities use BoTorch's native ``(callable, is_intrapoint)``
    contract. The callable returns a scalar tensor and feasibility means
    ``callable(X) >= 0``. With ``is_intrapoint=True`` it receives ``[d]``;
    otherwise it receives the joint q-batch ``[q, d]``. The callable must
    preserve device and floating dtype and remain differentiable with respect
    to candidate coordinates. Python callables are runtime objects and have no
    robotorchan-specific serialization format.
    """

    inequality_constraints: tuple[LinearConstraint, ...] = ()
    equality_constraints: tuple[LinearConstraint, ...] = ()
    nonlinear_inequality_constraints: tuple[NonlinearConstraint, ...] = ()

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
