"""Binary classification likelihoods with explicit label corruption."""

from __future__ import annotations

import math

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
from torch import Tensor, nn
from torch.distributions import Bernoulli

from robotorchan.models.classification.binary.robust.base import (
    ClassificationRobustnessType,
    RobustBinaryClassificationMixin,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


def _validate_flip_probability(value: float, *, name: str) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 <= value < 0.5:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 0.5.")


class LabelNoiseBernoulliLikelihood(BernoulliLikelihood):
    """Bernoulli likelihood with symmetric or asymmetric label flips."""

    def __init__(
        self,
        *,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        learn_flip_probabilities: bool = False,
    ) -> None:
        super().__init__()
        _validate_flip_probability(false_positive_rate, name="false_positive_rate")
        _validate_flip_probability(false_negative_rate, name="false_negative_rate")
        rates = torch.tensor([false_positive_rate, false_negative_rate])
        logits = torch.logit(rates.clamp(1e-6, 0.5 - 1e-6) / 0.5)
        self.raw_flip_logits = nn.Parameter(logits, requires_grad=learn_flip_probabilities)

    @property
    def flip_probabilities(self) -> Tensor:
        """Return false-positive and false-negative rates."""
        return 0.5 * self.raw_flip_logits.sigmoid()

    @property
    def false_positive_rate(self) -> Tensor:
        """Return P(observed=1 | clean=0)."""
        return self.flip_probabilities[0]

    @property
    def false_negative_rate(self) -> Tensor:
        """Return P(observed=0 | clean=1)."""
        return self.flip_probabilities[1]

    def corrupt_positive_probability(self, clean_probability: Tensor) -> Tensor:
        """Map clean positive-class probabilities to observed-label probabilities."""
        false_positive = self.false_positive_rate.to(clean_probability)
        false_negative = self.false_negative_rate.to(clean_probability)
        return false_positive + (1.0 - false_positive - false_negative) * clean_probability

    def forward(self, function_samples: Tensor, *args: object, **kwargs: object) -> Bernoulli:
        """Return corrupted Bernoulli distributions for latent samples."""
        clean = super().forward(function_samples, *args, **kwargs).probs
        return Bernoulli(probs=self.corrupt_positive_probability(clean))

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *args: object,
        **kwargs: object,
    ) -> Bernoulli:
        """Return the corrupted posterior-predictive Bernoulli distribution."""
        clean = super().marginal(function_dist, *args, **kwargs).probs
        return Bernoulli(probs=self.corrupt_positive_probability(clean))

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        """Integrate corrupted Bernoulli log probability over the latent Gaussian."""

        def log_prob(function_samples: Tensor) -> Tensor:
            return self.forward(function_samples).log_prob(observations)

        return self.quadrature(log_prob, function_dist)


class LabelNoiseBinarySingleTaskGPClassifier(
    RobustBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Binary variational GP classifier with explicit label-flip noise."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        flip_probability: float | None = None,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        learn_flip_probabilities: bool = False,
        **kwargs: object,
    ) -> None:
        if flip_probability is not None:
            _validate_flip_probability(flip_probability, name="flip_probability")
            if false_positive_rate != 0.0 or false_negative_rate != 0.0:
                raise ValueError("flip_probability cannot be combined with asymmetric flip rates.")
            false_positive_rate = flip_probability
            false_negative_rate = flip_probability
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self.likelihood = LabelNoiseBernoulliLikelihood(
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            learn_flip_probabilities=learn_flip_probabilities,
        )

    def predict_clean_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return posterior-predictive probabilities before label corruption."""
        latent = self.latent_posterior(X, **kwargs)
        positive = BernoulliLikelihood.marginal(self.likelihood, latent.distribution).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def sample_clean_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw clean class probabilities before label corruption."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        positive = BernoulliLikelihood.forward(self.likelihood, latent_samples).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return label-noise robustness metadata."""
        return frozenset({ClassificationRobustnessType.LABEL_NOISE})

    @property
    def models_observed_label_process(self) -> bool:
        """Return whether observed-label corruption is represented."""
        return True

    @property
    def flip_probabilities(self) -> Tensor:
        """Return current false-positive and false-negative probabilities."""
        return self.likelihood.flip_probabilities
