"""Candidate-input uncertainty for multiclass and ordinal GP classification.

Reuse the binary classifiers' Gaussian perturbation and finite categorical
support integration without creating another posterior approximation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.models.classification.binary.uncertain.base import (
    ClassificationInputUncertaintyType,
    ClassificationUncertaintyIntegration,
    ClassificationUncertaintyTarget,
    UncertainBinaryClassificationMixin,
)
from robotorchan.models.classification.binary.uncertain.uncertain_categorical import (
    _validate_category_distribution,
)
from robotorchan.models.classification.binary.uncertain.uncertain_input import (
    _validate_candidate_uncertainty,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    MixedOrdinalSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)


class _GaussianCandidateInputMixin(UncertainBinaryClassificationMixin):
    """Sample candidate perturbations using the binary GP validation contract."""

    @property
    def classification_input_uncertainty(
        self,
    ) -> frozenset[ClassificationInputUncertaintyType]:
        return frozenset({ClassificationInputUncertaintyType.CONTINUOUS})

    @property
    def classification_uncertainty_target(self) -> ClassificationUncertaintyTarget:
        return ClassificationUncertaintyTarget.CANDIDATE_INPUTS

    @property
    def classification_uncertainty_integration(
        self,
    ) -> ClassificationUncertaintyIntegration:
        return ClassificationUncertaintyIntegration.MONTE_CARLO

    def sample_uncertain_inputs(
        self,
        X: Tensor,
        *,
        input_std: Tensor | None = None,
        input_covar: Tensor | None = None,
        num_samples: int = 128,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        covariance = _validate_candidate_uncertainty(
            X,
            input_std=input_std,
            input_covar=input_covar,
        )
        d = X.shape[-1]
        jitter = torch.finfo(X.dtype).eps
        identity = torch.eye(d, dtype=X.dtype, device=X.device)
        chol = torch.linalg.cholesky(covariance + jitter * identity)
        noise = torch.randn(
            (num_samples, *X.shape),
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        )
        perturbation = torch.matmul(chol.unsqueeze(0), noise.unsqueeze(-1)).squeeze(-1)
        return X.unsqueeze(0) + perturbation

    def predict_proba_with_input_uncertainty(
        self,
        X: Tensor,
        *,
        input_std: Tensor | None = None,
        input_covar: Tensor | None = None,
        num_samples: int = 128,
        generator: torch.Generator | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Compute E_epsilon[P(class | X + epsilon, observations)] over candidates."""
        perturbed = self.sample_uncertain_inputs(
            X,
            input_std=input_std,
            input_covar=input_covar,
            num_samples=num_samples,
            generator=generator,
        )
        # Delegate to the existing classification family without modifying
        # the latent posterior, variational objective, or class link.
        probabilities = super().predict_proba(perturbed, **kwargs)
        return probabilities.mean(dim=0)


class ContinuousUncertainInputMulticlassGPClassifier(
    _GaussianCandidateInputMixin, MulticlassSingleTaskGPClassifier
):
    """Multiclass GP with Gaussian candidate-design input uncertainty."""


class ContinuousUncertainInputOrdinalGPClassifier(
    _GaussianCandidateInputMixin, OrdinalSingleTaskGPClassifier
):
    """Ordinal-probit GP with Gaussian candidate-design input uncertainty."""


class _CategoricalCandidateInputMixin(UncertainBinaryClassificationMixin):
    """Integrate one uncertain categorical input over its finite support."""

    uncertain_cat_dim: int

    def _validate_uncertain_cat_dim(self, train_X: Tensor, uncertain_cat_dim: int) -> None:
        if not isinstance(uncertain_cat_dim, int):
            raise TypeError("uncertain_cat_dim must be an integer.")
        if not -train_X.shape[-1] <= uncertain_cat_dim < train_X.shape[-1]:
            raise ValueError("uncertain_cat_dim is out of range.")
        normalized = uncertain_cat_dim % train_X.shape[-1]
        if normalized not in self.cat_dims:
            raise ValueError("uncertain_cat_dim must identify a categorical dimension.")
        self.uncertain_cat_dim = normalized

    @property
    def classification_input_uncertainty(
        self,
    ) -> frozenset[ClassificationInputUncertaintyType]:
        return frozenset({ClassificationInputUncertaintyType.CATEGORICAL})

    @property
    def classification_uncertainty_target(self) -> ClassificationUncertaintyTarget:
        return ClassificationUncertaintyTarget.CANDIDATE_INPUTS

    @property
    def classification_uncertainty_integration(
        self,
    ) -> ClassificationUncertaintyIntegration:
        return ClassificationUncertaintyIntegration.ANALYTIC

    def predict_proba_with_category_uncertainty(
        self,
        X: Tensor,
        *,
        category_values: Tensor,
        category_probabilities: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Exact finite-support average of class probabilities, not latent means."""
        _validate_category_distribution(
            X,
            category_values=category_values,
            category_probabilities=category_probabilities,
        )
        num_categories = category_values.numel()
        expanded = X.unsqueeze(-2).expand(
            *X.shape[:-1], num_categories, X.shape[-1]
        ).clone()
        expanded[..., self.uncertain_cat_dim] = category_values
        probabilities = super().predict_proba(expanded, **kwargs)
        return (probabilities * category_probabilities.unsqueeze(-1)).sum(dim=-2)


class UncertainCategoricalMulticlassGPClassifier(
    _CategoricalCandidateInputMixin, MixedMulticlassSingleTaskGPClassifier
):
    """Mixed multiclass GP with one uncertain candidate categorical feature."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        uncertain_cat_dim: int,
        cat_dims: list[int],
        num_classes: int,
        **kwargs: object,
    ) -> None:
        super().__init__(
            train_X, train_Y, num_classes=num_classes, cat_dims=cat_dims, **kwargs
        )
        self._validate_uncertain_cat_dim(train_X, uncertain_cat_dim)


class UncertainCategoricalOrdinalGPClassifier(
    _CategoricalCandidateInputMixin, MixedOrdinalSingleTaskGPClassifier
):
    """Mixed ordinal GP with one uncertain candidate categorical feature."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        uncertain_cat_dim: int,
        cat_dims: list[int],
        num_classes: int,
        **kwargs: object,
    ) -> None:
        super().__init__(
            train_X, train_Y, num_classes=num_classes, cat_dims=cat_dims, **kwargs
        )
        self._validate_uncertain_cat_dim(train_X, uncertain_cat_dim)
