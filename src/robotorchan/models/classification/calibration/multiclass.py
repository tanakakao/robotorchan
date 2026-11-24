"""Multiclass and ordinal probability calibration without altering latent GP models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.calibration.base import ProbabilityCalibrator
from robotorchan.models.classification.posterior.ensemble import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)


class CalibratedMulticlassClassifier(ClassificationModelMixin, nn.Module):
    """Apply an existing post-hoc calibrator to predictive probability vectors.

    This class does not re-train the underlying surrogate and does not claim
    that a single non-GP estimator has an epistemic probability posterior.
    Calibrated probability samples are available only when the base classifier
    implements sample_class_probabilities().
    """

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.CUSTOM

    def __init__(self, model: nn.Module, calibrator: ProbabilityCalibrator) -> None:
        super().__init__()
        num_classes = getattr(model, "num_classes", None)
        if not isinstance(num_classes, int) or num_classes < 3:
            raise ValueError("model must expose at least three predictive classes.")
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must implement predict_proba(X).")
        if not isinstance(calibrator, ProbabilityCalibrator):
            raise TypeError("calibrator must implement ProbabilityCalibrator.")
        labels = tuple(getattr(model, "class_labels", ()))
        if labels != tuple(range(num_classes)):
            raise ValueError("model class_labels must be ordered 0 through num_classes-1.")
        self.model = model
        self.calibrator = calibrator
        self._num_classes = num_classes
        self._class_labels = labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_labels(self) -> tuple[int, ...]:
        return self._class_labels

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.calibrator(self.model.predict_proba(X, **kwargs))

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        return self.calibrator(
            self.model.sample_class_probabilities(X, sample_shape=sample_shape, **kwargs)
        )

    def latent_posterior(self, X: Tensor, **kwargs: object):
        """Forward the uncalibrated latent posterior when the model has one."""
        return self.model.latent_posterior(X, **kwargs)

    def sample_latent(
        self, X: Tensor, sample_shape: torch.Size | None = None, **kwargs: object
    ) -> Tensor:
        return self.model.sample_latent(X, sample_shape=sample_shape, **kwargs)

    def probability_posterior(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> ClassificationEnsemblePosterior:
        """Empirical distribution of calibrated class-probability samples.

        This Monte Carlo posterior integrates the underlying predictive
        probability samples rather than transforming their posterior mean.
        No single deterministic non-GP estimator is assigned a fictitious
        epistemic posterior.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        return make_classification_ensemble_posterior(samples)

    def probability_variance(
        self, X: Tensor, *, num_samples: int = 128, **kwargs: object
    ) -> Tensor:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        return samples.var(dim=0, unbiased=False)

    def expected_class_entropy(
        self, X: Tensor, *, num_samples: int = 128, **kwargs: object
    ) -> Tensor:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        probabilities = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1).mean(dim=0)

    def mutual_information(self, X: Tensor, *, num_samples: int = 128, **kwargs: object) -> Tensor:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        probabilities = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        predictive = probabilities.mean(dim=0)
        total = -torch.special.xlogy(predictive, predictive).sum(dim=-1)
        conditional = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
        return (total - conditional.mean(dim=0)).clamp_min(0.0)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)


class CalibratedOrdinalClassifier(CalibratedMulticlassClassifier):
    """Calibrated ordinal-class probabilities with unchanged class order.

    Temperature scaling is a post-hoc operation and does not impose ordinal
    threshold or stochastic-order constraints on arbitrary underlying models.
    """
