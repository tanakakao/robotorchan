"""Risk transforms for class probabilities across explicit scenarios."""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import Tensor

from robotorchan.objectives.risk import CVaR, Expectation, VaR, WorstCase


class ClassificationProbabilityRiskType(StrEnum):
    """Supported robust aggregations of scenario-wise class probabilities."""

    EXPECTED = "expected"
    WORST_CASE = "worst_case"
    QUANTILE = "quantile"
    VAR = "var"
    CVAR = "cvar"


class RobustProbabilityTransform:
    """Aggregate scenario-wise class probabilities without reducing class axis."""

    def __init__(
        self,
        risk_type: ClassificationProbabilityRiskType | str = (
            ClassificationProbabilityRiskType.EXPECTED
        ),
        *,
        alpha: float = 0.9,
        scenario_dim: int = -2,
    ) -> None:
        """Initialize a class-probability risk transform."""
        self.risk_type = ClassificationProbabilityRiskType(risk_type)
        self.alpha = float(alpha)
        self.scenario_dim = int(scenario_dim)
        if self.risk_type in {
            ClassificationProbabilityRiskType.QUANTILE,
            ClassificationProbabilityRiskType.VAR,
            ClassificationProbabilityRiskType.CVAR,
        }:
            VaR(alpha=self.alpha)

    def __call__(self, probabilities: Tensor) -> Tensor:
        """Aggregate scenario probabilities while preserving the class dimension."""
        self._validate_probabilities(probabilities)
        scenario_dim = self.scenario_dim % probabilities.ndim
        if scenario_dim == probabilities.ndim - 1:
            raise ValueError("scenario_dim must not identify the class dimension.")
        moved = probabilities.movedim(scenario_dim, -1)

        if self.risk_type is ClassificationProbabilityRiskType.EXPECTED:
            return Expectation()(moved)
        if self.risk_type is ClassificationProbabilityRiskType.WORST_CASE:
            return WorstCase()(moved)
        if self.risk_type in {
            ClassificationProbabilityRiskType.QUANTILE,
            ClassificationProbabilityRiskType.VAR,
        }:
            return VaR(alpha=self.alpha)(moved)
        if self.risk_type is ClassificationProbabilityRiskType.CVAR:
            return CVaR(alpha=self.alpha)(moved)
        raise RuntimeError(f"Unhandled probability risk type: {self.risk_type}")

    @staticmethod
    def _validate_probabilities(probabilities: Tensor) -> None:
        if probabilities.ndim < 2:
            raise ValueError("probabilities must include scenario and class dimensions.")
        if probabilities.shape[-1] < 2:
            raise ValueError("The final probability dimension must contain at least two classes.")
        if not torch.is_floating_point(probabilities):
            raise TypeError("probabilities must use a floating-point dtype.")
        if not torch.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite.")
        if torch.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probabilities must lie in [0, 1].")
        sums = probabilities.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
            raise ValueError("Class probabilities must sum to one for every scenario.")


def robust_class_probability(
    probabilities: Tensor,
    *,
    class_index: int,
    risk_type: ClassificationProbabilityRiskType | str = (
        ClassificationProbabilityRiskType.EXPECTED
    ),
    alpha: float = 0.9,
    scenario_dim: int = -2,
) -> Tensor:
    """Aggregate one class probability across scenarios."""
    if not 0 <= class_index < probabilities.shape[-1]:
        raise ValueError("class_index is outside the class-probability dimension.")
    transformed = RobustProbabilityTransform(
        risk_type,
        alpha=alpha,
        scenario_dim=scenario_dim,
    )(probabilities)
    return transformed[..., class_index]
