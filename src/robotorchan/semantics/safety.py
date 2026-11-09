"""Configuration and observed safe seeds for sequential safe optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.semantics.constraints import ContinuousConstraint


@dataclass(frozen=True, slots=True)
class SafetySpecification:
    """Define continuous safety outputs and verified initial observations.

    Each column of observed_constraints corresponds to the constraint at the
    same position. A seed is accepted only when all observed constraints pass.
    This checks observations, not a probabilistic safety guarantee.
    """

    constraints: tuple[ContinuousConstraint, ...]
    initial_X: Tensor
    observed_constraints: Tensor
    beta: float = 4.0

    def __post_init__(self) -> None:
        """Reject malformed or unsafe observed seeds."""
        if not self.constraints or not all(
            isinstance(c, ContinuousConstraint) for c in self.constraints
        ):
            raise ValueError("constraints must contain ContinuousConstraint instances.")
        if not isinstance(self.beta, (int, float)) or isinstance(self.beta, bool):
            raise TypeError("beta must be a positive finite real number.")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be positive and finite.")
        X, Y = self.initial_X, self.observed_constraints
        if not isinstance(X, Tensor) or not isinstance(Y, Tensor):
            raise TypeError("initial_X and observed_constraints must be tensors.")
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("initial_X and observed_constraints must be rank-2.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("initial_X must contain at least one point and feature.")
        if Y.shape != (X.shape[0], len(self.constraints)):
            raise ValueError("observed_constraints must have shape [n, m].")
        if not X.is_floating_point() or not Y.is_floating_point():
            raise TypeError("initial observations must use floating tensors.")
        if not bool(torch.isfinite(X).all() and torch.isfinite(Y).all()):
            raise ValueError("initial observations must be finite.")
        for j, constraint in enumerate(self.constraints):
            residual = constraint.direction.residual(Y[:, j], constraint.threshold)
            if bool((residual > 0).any()):
                raise ValueError(f"initial_X contains an unsafe seed for constraint {j}.")
