"""Direction and sign conventions for optimization semantics."""

from __future__ import annotations

from enum import StrEnum

from torch import Tensor


class ObjectiveDirection(StrEnum):
    """Declare whether an objective is maximized or minimized."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    @property
    def sign(self) -> int:
        """Return the multiplier that normalizes the objective to maximization."""
        return 1 if self is ObjectiveDirection.MAXIMIZE else -1

    def apply(self, values: Tensor) -> Tensor:
        """Normalize objective values to the maximization convention."""
        return values if self is ObjectiveDirection.MAXIMIZE else -values


class ConstraintDirection(StrEnum):
    """Declare a one-sided outcome-constraint inequality direction."""

    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"

    def residual(self, values: Tensor, threshold: float | Tensor) -> Tensor:
        """Return a BoTorch-style residual where values <= 0 are feasible."""
        if self is ConstraintDirection.LESS_THAN_OR_EQUAL:
            return values - threshold
        return threshold - values
