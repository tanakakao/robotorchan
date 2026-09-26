"""Constraint contracts for acquisition-function search strategies.

Candidate-space constraints are separate from output constraints used by
constrained acquisition functions. This module represents constraints on
candidate coordinates optimized by a search strategy.

The linear tuple format follows the BoTorch optimizer contract directly:
``(indices, coefficients, rhs)``. This avoids a second robotorchan-specific
constraint language and preserves intra-point and inter-point constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

LinearConstraint = tuple[Tensor, Tensor, float]


@dataclass(frozen=True)
class CandidateConstraints:
    """Candidate-space constraints for compatible search strategies.

    Linear constraints use the BoTorch ``(indices, coefficients, rhs)``
    format. Inequalities mean ``sum(X[indices] * coefficients) >= rhs``.
    One-dimensional indices describe intra-point constraints; two-dimensional
    indices may describe inter-point q-batch constraints.

    These are input constraints, not probabilistic output constraints such as
    ``g(x) <= 0`` used by constrained BO acquisition functions.

    Nonlinear candidate constraints are intentionally deferred because their
    initialization and batching requirements need a separate explicit API.
    """

    inequality_constraints: tuple[LinearConstraint, ...] = ()
    equality_constraints: tuple[LinearConstraint, ...] = ()

    @property
    def has_linear_constraints(self) -> bool:
        """Whether at least one linear candidate constraint is configured."""
        return bool(self.inequality_constraints or self.equality_constraints)
