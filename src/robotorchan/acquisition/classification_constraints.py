"""Feasibility utilities backed by probabilistic classification models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor, nn

from robotorchan.models.classification.probability import (
    ClassificationProbabilityRiskType,
    RobustProbabilityTransform,
)
from robotorchan.semantics.probability import ClassificationProbabilityOfFeasibility


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


def _normalize_q_batch_probabilities(probability: Tensor, X: Tensor) -> Tensor:
    """Restore a missing singleton q axis without broadcasting t-batches."""
    if X.ndim < 2 or X.shape[-2] < 1:
        raise ValueError("X must have shape batch_shape x q x d with q >= 1.")
    if X.shape[-2] == 1 and probability.shape == X.shape[:-2]:
        probability = probability.unsqueeze(-1)
    if probability.shape != X.shape[:-1]:
        raise ValueError("Feasibility must have shape X.shape[:-1] (batch_shape x q).")
    if not torch.isfinite(probability).all():
        raise ValueError("Feasibility probabilities must be finite.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Feasibility probabilities must be in [0, 1].")
    return probability


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
        existing_pending = getattr(objective_acquisition, "X_pending", None)
        if existing_pending is not None and existing_pending.numel() > 0:
            raise NotImplementedError(
                "Classification-weighted acquisition does not support nonempty X_pending."
            )
        self.objective_acquisition = objective_acquisition
        self.probability_of_feasibility = probability_of_feasibility
        self.q_reduction = q_reduction

    def fantasize(self, *args: object, **kwargs: object) -> None:
        """Reject unsupported joint fantasy updates across model families."""
        raise NotImplementedError(
            "Classification-weighted acquisition does not support joint fantasization."
        )

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """Reject pending candidates that lack matching classifier weighting."""
        if X_pending is not None and X_pending.numel() > 0:
            raise NotImplementedError(
                "Classification-weighted acquisition does not support nonempty X_pending."
            )
        self.objective_acquisition.set_X_pending(X_pending)
        super().set_X_pending(X_pending)

    def forward(self, X: Tensor) -> Tensor:
        """Return objective acquisition weighted by joint q-batch feasibility."""
        objective_value = self.objective_acquisition(X)
        feasibility = self.probability_of_feasibility(X)
        feasibility = _normalize_q_batch_probabilities(feasibility, X)
        expected_batch_shape = X.shape[:-2]
        if objective_value.shape != expected_batch_shape and not (
            X.ndim == 2 and objective_value.shape == torch.Size([1])
        ):
            raise ValueError("Objective acquisition must return one value per t-batch.")
        if self.q_reduction == "product":
            feasibility = feasibility.prod(dim=-1)
        else:
            feasibility = feasibility.min(dim=-1).values
        return objective_value * feasibility


class IndependentFeasibilityAggregator(nn.Module):
    """Combine distinct classifier marginal PoFs under explicit independence."""

    def __init__(self, factors: Sequence[nn.Module]) -> None:
        """Validate independent factors."""
        super().__init__()
        if not factors:
            raise ValueError("At least one feasibility factor is required.")
        self.factors = nn.ModuleList(factors)

    def forward(self, X: Tensor) -> Tensor:
        """Return a per-candidate conjunction without collapsing q or t-batches."""
        probabilities = []
        for factor in self.factors:
            probability = factor(X)
            probabilities.append(_normalize_q_batch_probabilities(probability, X))
        return torch.stack(probabilities, dim=0).prod(dim=0)
