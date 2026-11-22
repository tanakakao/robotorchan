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
        feasible_classes: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize a classifier-backed probability adapter."""
        super().__init__()
        if not hasattr(model, "predict_proba"):
            raise TypeError("model must provide predict_proba(X).")
        num_classes = getattr(model, "num_classes", None)
        if not isinstance(num_classes, int) or num_classes < 2:
            raise TypeError("model must expose an integer num_classes >= 2.")
        if feasible_classes is None:
            selected = (feasible_class,)
        else:
            if not feasible_classes or any(
                not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < num_classes for index in feasible_classes
            ):
                raise ValueError("feasible_classes must contain valid class indices.")
            if len(set(feasible_classes)) != len(feasible_classes):
                raise ValueError("feasible_classes must not contain duplicates.")
            selected = tuple(feasible_classes)
        if any(not 0 <= index < num_classes for index in selected):
            raise ValueError("feasible_class is outside the classifier class range.")
        self.model = model
        self.feasible_class = feasible_class if feasible_classes is None else None
        self.feasible_classes = selected

    def forward(self, X: Tensor) -> Tensor:
        """Return per-candidate P(feasible | X, D)."""
        probabilities = self.model.predict_proba(X)
        return probabilities[..., list(self.feasible_classes)].sum(dim=-1)
