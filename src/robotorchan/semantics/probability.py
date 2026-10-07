"""Classification probability adapters below the acquisition layer."""

from __future__ import annotations

from torch import Tensor, nn


class ClassificationProbabilityOfFeasibility(nn.Module):
    """Map classifier probabilities to a differentiable probability of feasibility."""

    def __init__(
        self,
        model: nn.Module,
        *,
        feasible_class: int = 1,
    ) -> None:
        """Initialize a classifier-backed probability adapter."""
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
