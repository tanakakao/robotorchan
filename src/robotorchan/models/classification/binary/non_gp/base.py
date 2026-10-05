"""Common contract for non-GP binary classifiers."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import torch
from torch import Tensor

from robotorchan.models.classification.binary.base import BinaryClassificationMixin


class NonGPBinaryClassificationMixin(BinaryClassificationMixin):
    """Binary classification contract for models without a latent posterior."""

    is_non_gp_classification: ClassVar[bool] = True
    has_latent_posterior: ClassVar[bool] = False

    def latent_posterior(self, X: Tensor, **kwargs: object):
        """Reject latent-posterior access for discriminative non-GP models."""
        del X, kwargs
        raise NotImplementedError("Non-GP classifiers do not expose a latent BoTorch posterior.")

    def sample_latent(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Reject latent sampling when no latent posterior exists."""
        del X, sample_shape, kwargs
        raise NotImplementedError("Non-GP classifiers do not expose latent samples.")

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Reject epistemic probability sampling for a single fitted estimator."""
        del X, sample_shape, kwargs
        raise NotImplementedError(
            "A single non-GP classifier has no posterior probability samples. "
            "Use an ensemble classifier for epistemic uncertainty."
        )

    def latent_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Reject latent variance when no latent function is modeled."""
        del X, kwargs
        raise NotImplementedError("Non-GP classifiers do not expose latent variance.")

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Reject epistemic variance for a single fitted estimator."""
        del X, num_samples, kwargs
        raise NotImplementedError(
            "A single non-GP classifier has no epistemic probability variance."
        )

    def expected_class_entropy(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Reject expected conditional entropy without posterior members."""
        del X, num_samples, kwargs
        raise NotImplementedError(
            "A single non-GP classifier has no posterior-member entropy."
        )

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Reject BALD-style mutual information without posterior members."""
        del X, num_samples, kwargs
        raise NotImplementedError(
            "A single non-GP classifier has no posterior-member mutual information."
        )

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli observation variance for each class."""
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return entropy of the fitted predictive class distribution."""
        probabilities = self.predict_proba(X, **kwargs)
        tiny = torch.finfo(probabilities.dtype).tiny
        probabilities = probabilities.clamp_min(tiny)
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)

    def predictive_distribution(self, X: Tensor, **kwargs: object):
        """Return a categorical predictive distribution."""
        return torch.distributions.Categorical(probs=self.predict_proba(X, **kwargs))

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary predictions from positive-class probability."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        return (self.predict_proba(X, **kwargs)[..., 1] >= threshold).long()

    @abstractmethod
    def fit(self) -> None:
        """Fit the underlying non-GP classifier."""
