"""Common contracts for classification surrogate models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from botorch.posteriors import Posterior
from torch import Tensor


class ClassificationModelMixin(ABC):
    """Common prediction contract for classification surrogate models.

    ``posterior(X)`` remains the latent-function posterior. Class probabilities
    and discrete predictions are exposed separately so latent uncertainty is
    not confused with predictive label uncertainty.
    """

    task_type: ClassVar[str] = "classification"

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of predictive classes."""

    @property
    @abstractmethod
    def class_labels(self) -> tuple[object, ...]:
        """Return labels ordered as the final ``predict_proba`` dimension."""

    def latent_posterior(self, X: Tensor, **kwargs: object) -> Posterior:
        """Return the latent-function posterior using the BoTorch contract."""
        return self.posterior(X, **kwargs)

    @abstractmethod
    def predictive_distribution(self, X: Tensor, **kwargs: object) -> object:
        """Return the observation-space predictive distribution."""

    @abstractmethod
    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return class probabilities with classes on the final dimension."""

    @abstractmethod
    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return discrete class predictions."""


class BinaryClassificationMixin(ClassificationModelMixin):
    """Binary specialization without defining a likelihood or link function."""

    num_classes: ClassVar[int] = 2
    class_labels: ClassVar[tuple[int, int]] = (0, 1)

    @abstractmethod
    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary predictions using a model-specific probability path."""
