"""Mixed binary GP classification with uncertain categorical candidates."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.models.classification.binary.standard.single_task import (
    MixedBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.uncertain.base import (
    ClassificationInputUncertaintyType,
    ClassificationUncertaintyIntegration,
    ClassificationUncertaintyTarget,
    UncertainBinaryClassificationMixin,
)


def _validate_category_distribution(
    X: Tensor,
    *,
    category_values: Tensor,
    category_probabilities: Tensor,
) -> None:
    """Validate categorical support values and probabilities."""
    if category_values.ndim != 1:
        raise ValueError("category_values must be one-dimensional.")
    if category_values.numel() < 2:
        raise ValueError("category_values must contain at least two categories.")
    if category_values.device != X.device or category_values.dtype != X.dtype:
        raise ValueError("category_values must share dtype and device with X.")
    expected = (*X.shape[:-1], category_values.numel())
    if category_probabilities.shape != expected:
        raise ValueError("category_probabilities has an incompatible shape.")
    if category_probabilities.device != X.device or category_probabilities.dtype != X.dtype:
        raise ValueError("category_probabilities must share dtype and device with X.")
    if not torch.isfinite(category_values).all():
        raise ValueError("category_values must be finite.")
    if torch.unique(category_values).numel() != category_values.numel():
        raise ValueError("category_values must be unique.")
    if not torch.isfinite(category_probabilities).all():
        raise ValueError("category_probabilities must be finite.")
    if torch.any(category_probabilities < 0):
        raise ValueError("category_probabilities must be nonnegative.")
    sums = category_probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("category_probabilities must sum to one.")


class UncertainCategoricalBinarySingleTaskGPClassifier(
    UncertainBinaryClassificationMixin,
    MixedBinarySingleTaskGPClassifier,
):
    """Mixed binary classifier marginalizing one uncertain categorical feature."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        uncertain_cat_dim: int,
        cat_dims: list[int],
        **kwargs: object,
    ) -> None:
        """Initialize a mixed classifier with one uncertain categorical dimension."""
        super().__init__(train_X, train_Y, cat_dims=cat_dims, **kwargs)
        normalized = uncertain_cat_dim % train_X.shape[-1]
        if normalized not in self.cat_dims:
            raise ValueError("uncertain_cat_dim must identify a categorical dimension.")
        self.uncertain_cat_dim = normalized

    @property
    def classification_input_uncertainty(
        self,
    ) -> frozenset[ClassificationInputUncertaintyType]:
        """Return categorical input uncertainty metadata."""
        return frozenset({ClassificationInputUncertaintyType.CATEGORICAL})

    @property
    def classification_uncertainty_target(self) -> ClassificationUncertaintyTarget:
        """Return candidate-input uncertainty semantics."""
        return ClassificationUncertaintyTarget.CANDIDATE_INPUTS

    @property
    def classification_uncertainty_integration(
        self,
    ) -> ClassificationUncertaintyIntegration:
        """Return exact finite-support integration semantics."""
        return ClassificationUncertaintyIntegration.ANALYTIC

    def predict_proba_with_category_uncertainty(
        self,
        X: Tensor,
        *,
        category_values: Tensor,
        category_probabilities: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Return the exact finite-support expectation of class probabilities."""
        _validate_category_distribution(
            X,
            category_values=category_values,
            category_probabilities=category_probabilities,
        )
        num_categories = category_values.numel()
        expanded = X.unsqueeze(-2).expand(*X.shape[:-1], num_categories, X.shape[-1]).clone()
        expanded[..., self.uncertain_cat_dim] = category_values
        probabilities = super().predict_proba(expanded, **kwargs)
        weights = category_probabilities.unsqueeze(-1)
        return (probabilities * weights).sum(dim=-2)
