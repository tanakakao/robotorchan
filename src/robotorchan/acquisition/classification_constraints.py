"""Feasibility utilities backed by probabilistic classification models."""

from __future__ import annotations

from collections.abc import Callable

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor, nn

from robotorchan.models.classification.probability import (
    ClassificationProbabilityRiskType,
    RobustProbabilityTransform,
)


class ClassificationProbabilityOfFeasibility(nn.Module):
    """Map classifier probabilities to a differentiable probability of feasibility."""

    def __init__(
        self,
        model: nn.Module,
        *,
        feasible_class: int = 1,
    ) -> None:
        """Initialize a classifier-backed PoF adapter."""
        super().__init__()
        if not hasattr(model, "predict_proba"):
            raise TypeError("model must provide predict_proba(X).")
        num_classes = getattr(model, "num_classes", None)
        if not isinstance(num_classes, int) or num_classes < 2:
            raise TypeError("model must expose an integer num_classes >= 2.")
        if not 0 <= feasible_class < num_classes:
            raise ValueError("feasible_class is outside the classifier class range.")
        self.model = model
        self.feasible_class = feasible_class

    def forward(self, X: Tensor) -> Tensor:
        """Return per-candidate P(feasible | X, D)."""
        probabilities = self.model.predict_proba(X)
        return probabilities[..., self.feasible_class]



class ClassificationProbabilityAcquisition(AcquisitionFunction):
    """Expose q=1 classifier feasibility as a BoTorch acquisition."""

    def __init__(
        self,
        probability_of_feasibility: ClassificationProbabilityOfFeasibility,
    ) -> None:
        """Initialize the q=1 acquisition adapter."""
        super().__init__(model=probability_of_feasibility.model)
        self.probability_of_feasibility = probability_of_feasibility

    def forward(self, X: Tensor) -> Tensor:
        """Return one feasibility acquisition value per t-batch."""
        if X.shape[-2] != 1:
            raise ValueError("ClassificationProbabilityAcquisition supports q=1 only.")
        return self.probability_of_feasibility(X).squeeze(-1)


class RobustClassificationProbabilityOfFeasibility(nn.Module):
    """Aggregate classifier feasibility probabilities across explicit scenarios."""

    def __init__(
        self,
        model: nn.Module,
        *,
        feasible_class: int = 1,
        risk_type: ClassificationProbabilityRiskType | str = (
            ClassificationProbabilityRiskType.WORST_CASE
        ),
        alpha: float = 0.9,
        scenario_generator: Callable[[Tensor], Tensor],
        scenario_dim: int = -2,
    ) -> None:
        """Initialize robust classifier-backed PoF."""
        super().__init__()
        self.base = ClassificationProbabilityOfFeasibility(
            model,
            feasible_class=feasible_class,
        )
        self.transform = RobustProbabilityTransform(
            risk_type,
            alpha=alpha,
            scenario_dim=scenario_dim,
        )
        self.scenario_generator = scenario_generator
        self.scenario_dim = scenario_dim

    def forward(self, X: Tensor) -> Tensor:
        """Return a lower-tail or expected feasibility score across scenarios."""
        scenario_X = self.scenario_generator(X)
        probabilities = self.base.model.predict_proba(scenario_X)
        robust = self.transform(probabilities)
        return robust[..., self.base.feasible_class]


class FeasibilityWeightedAcquisition(AcquisitionFunction):
    """Weight an arbitrary objective acquisition by classifier feasibility."""

    def __init__(
        self,
        objective_acquisition: AcquisitionFunction,
        probability_of_feasibility: nn.Module,
        *,
        q_reduction: str = "product",
    ) -> None:
        """Initialize a classifier-constrained acquisition adapter."""
        super().__init__(model=objective_acquisition.model)
        if q_reduction not in {"product", "minimum"}:
            raise ValueError("q_reduction must be 'product' or 'minimum'.")
        self.objective_acquisition = objective_acquisition
        self.probability_of_feasibility = probability_of_feasibility
        self.q_reduction = q_reduction

    def forward(self, X: Tensor) -> Tensor:
        """Return objective acquisition weighted by joint q-batch feasibility."""
        objective_value = self.objective_acquisition(X)
        feasibility = self.probability_of_feasibility(X)
        if feasibility.ndim > objective_value.ndim:
            if self.q_reduction == "product":
                feasibility = feasibility.prod(dim=-1)
            else:
                feasibility = feasibility.min(dim=-1).values
        return objective_value * feasibility
