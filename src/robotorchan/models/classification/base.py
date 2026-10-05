"""Common contracts for classification surrogate models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import torch
from botorch.posteriors import Posterior
from torch import Tensor


class ClassificationLikelihoodFamily(StrEnum):
    """Likelihood families exposed by classification models."""

    BERNOULLI = "bernoulli"
    CATEGORICAL = "categorical"
    CUSTOM = "custom"


class LatentOutputStructure(StrEnum):
    """Structure of the latent function represented by ``posterior``."""

    SINGLE = "single"
    PER_CLASS = "per_class"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ClassificationMetadata:
    """Stable metadata describing a classification prediction contract."""

    num_classes: int
    class_labels: tuple[object, ...]
    likelihood_family: ClassificationLikelihoodFamily
    latent_output_structure: LatentOutputStructure


class ClassificationModelMixin(ABC):
    """Common prediction contract for classification surrogate models.

    ``posterior(X)`` remains the latent-function posterior. Class probabilities
    and discrete predictions are exposed separately so latent uncertainty is
    not confused with predictive label uncertainty.
    """

    task_type: ClassVar[str] = "classification"
    is_classification: ClassVar[bool] = True

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of predictive classes."""

    @property
    @abstractmethod
    def class_labels(self) -> tuple[object, ...]:
        """Return labels ordered as the final ``predict_proba`` dimension."""

    @property
    @abstractmethod
    def likelihood_family(self) -> ClassificationLikelihoodFamily:
        """Return the observation likelihood family."""

    @property
    @abstractmethod
    def latent_output_structure(self) -> LatentOutputStructure:
        """Return how latent outputs correspond to the classification task."""

    @property
    def classification_metadata(self) -> ClassificationMetadata:
        """Return immutable metadata for the classification prediction contract."""
        return ClassificationMetadata(
            num_classes=self.num_classes,
            class_labels=self.class_labels,
            likelihood_family=self.likelihood_family,
            latent_output_structure=self.latent_output_structure,
        )

    def latent_posterior(self, X: Tensor, **kwargs: object) -> Posterior:
        """Return the latent-function posterior using the BoTorch contract.

        This is an explicit semantic alias for ``posterior(X)``. It never
        applies a classification likelihood or converts latent values into
        class probabilities.
        """
        posterior = self.posterior(X, **kwargs)
        return cast(Posterior, posterior)

    def sample_latent(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw reparameterized samples from the latent BoTorch posterior."""
        resolved_shape = torch.Size() if sample_shape is None else sample_shape
        return self.latent_posterior(X, **kwargs).rsample(sample_shape=resolved_shape)

    @abstractmethod
    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw class-probability samples induced by latent posterior samples."""

    def latent_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return posterior variance of the latent classification function."""
        return self.latent_posterior(X, **kwargs).variance

    @abstractmethod
    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return observation-space variance for each predictive class."""

    @abstractmethod
    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return entropy of the posterior-predictive class distribution."""

    @abstractmethod
    def predictive_distribution(self, X: Tensor, **kwargs: object) -> object:
        """Return the observation-space predictive distribution.

        Implementations must apply the classification likelihood or link to
        latent uncertainty. The returned object is distinct from the latent
        BoTorch ``Posterior`` exposed by ``posterior(X)``.
        """

    @abstractmethod
    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return posterior-predictive class probabilities.

        The final dimension must have size ``num_classes`` and follow
        ``class_labels`` ordering. Implementations must integrate latent
        uncertainty rather than treating a latent posterior mean as a class
        probability.
        """

    @abstractmethod
    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return discrete class predictions."""
