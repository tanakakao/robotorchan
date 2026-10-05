"""Calibrated classification model adapters."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.calibration.base import ProbabilityCalibrator


class CalibratedBinaryClassifier(BinaryClassificationMixin, nn.Module):
    """Apply a post-hoc probability calibrator to a binary classifier."""

    def __init__(self, model: nn.Module, calibrator: ProbabilityCalibrator) -> None:
        super().__init__()
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must expose predict_proba(X).")
        if getattr(model, "num_classes", None) != self.num_classes:
            raise ValueError("model must expose two predictive classes.")
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return calibrated predictive probabilities."""
        return self.calibrator(self.model.predict_proba(X, **kwargs))

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return class labels from calibrated positive-class probability."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(X, **kwargs)[..., 1] >= threshold).long()

    def latent_posterior(self, X: Tensor, **kwargs: object):
        """Delegate latent posterior without claiming that calibration changes it."""
        return self.model.latent_posterior(X, **kwargs)

    def sample_latent(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Delegate latent sampling to the underlying classifier."""
        return self.model.sample_latent(X, sample_shape=sample_shape, **kwargs)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Calibrate every sampled probability vector."""
        samples = self.model.sample_class_probabilities(
            X,
            sample_shape=sample_shape,
            **kwargs,
        )
        return self.calibrator(samples)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        """Return categorical distribution from calibrated probabilities."""
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli observation variance after calibration."""
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return entropy of calibrated predictive probabilities."""
        probabilities = self.predict_proba(X, **kwargs)
        safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        return -torch.special.xlogy(safe, safe).sum(dim=-1)

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Estimate calibrated class-probability epistemic variance."""
        samples = self.sample_class_probabilities(
            X,
            sample_shape=torch.Size([num_samples]),
            **kwargs,
        )
        return samples.var(dim=0, unbiased=False)

    def expected_class_entropy(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Estimate expected entropy after calibrating probability samples."""
        samples = self.sample_class_probabilities(
            X,
            sample_shape=torch.Size([num_samples]),
            **kwargs,
        )
        safe = samples.clamp_min(torch.finfo(samples.dtype).tiny)
        entropy = -torch.special.xlogy(safe, safe).sum(dim=-1)
        return entropy.mean(dim=0)

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return calibrated BALD-style disagreement."""
        return (
            self.predictive_entropy(X, **kwargs)
            - self.expected_class_entropy(X, num_samples=num_samples, **kwargs)
        ).clamp_min(0.0)
