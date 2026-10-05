"""Binary classification with an explicit contamination-label process."""

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


def _validate_probability(value: float, *, name: str, open_interval: bool = False) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    valid = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not valid:
        interval = "strictly between 0 and 1" if open_interval else "between 0 and 1"
        raise ValueError(f"{name} must be {interval}.")


class ContaminatedBernoulliLikelihood(BernoulliLikelihood):
    """Bernoulli likelihood mixed with an input-independent contaminant source."""

    def __init__(
        self,
        *,
        contamination_probability: float = 0.05,
        contaminant_positive_probability: float = 0.5,
        learn_contamination_probability: bool = False,
    ) -> None:
        super().__init__()
        _validate_probability(
            contamination_probability,
            name="contamination_probability",
            open_interval=True,
        )
        _validate_probability(
            contaminant_positive_probability,
            name="contaminant_positive_probability",
        )
        contamination = torch.tensor(contamination_probability)
        contaminant = torch.tensor(contaminant_positive_probability)
        self.raw_contamination_logit = nn.Parameter(
            torch.logit(contamination),
            requires_grad=learn_contamination_probability,
        )
        self.register_buffer("contaminant_positive_probability", contaminant)

    @property
    def contamination_probability(self) -> Tensor:
        """Return the current probability of the contaminant label process."""
        return self.raw_contamination_logit.sigmoid()

    def contaminate_positive_probability(self, clean_probability: Tensor) -> Tensor:
        """Mix clean and contaminant positive-class probabilities."""
        contamination = self.contamination_probability.to(clean_probability)
        contaminant = self.contaminant_positive_probability.to(clean_probability)
        return (1.0 - contamination) * clean_probability + contamination * contaminant

    def forward(self, function_samples: Tensor, *args: object, **kwargs: object) -> Bernoulli:
        """Return contaminated Bernoulli distributions for latent samples."""
        clean = super().forward(function_samples, *args, **kwargs).probs
        return Bernoulli(probs=self.contaminate_positive_probability(clean))

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *args: object,
        **kwargs: object,
    ) -> Bernoulli:
        """Return the contaminated posterior-predictive Bernoulli distribution."""
        clean = super().marginal(function_dist, *args, **kwargs).probs
        return Bernoulli(probs=self.contaminate_positive_probability(clean))

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        """Integrate contaminated Bernoulli log probability over the latent Gaussian."""

        def log_prob(function_samples: Tensor) -> Tensor:
            return self.forward(function_samples).log_prob(observations)

        return self.quadrature(log_prob, function_dist)


class ContaminatedBinarySingleTaskGPClassifier(
    RobustBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Binary variational GP robust to labels from a contaminant process."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        contamination_probability: float = 0.05,
        contaminant_positive_probability: float = 0.5,
        learn_contamination_probability: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self.likelihood = ContaminatedBernoulliLikelihood(
            contamination_probability=contamination_probability,
            contaminant_positive_probability=contaminant_positive_probability,
            learn_contamination_probability=learn_contamination_probability,
        )

    def predict_clean_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return posterior-predictive probabilities before contamination."""
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
        """Draw clean class probabilities before contamination."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        positive = BernoulliLikelihood.forward(self.likelihood, latent_samples).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return contamination robustness metadata."""
        return frozenset({ClassificationRobustnessType.CONTAMINATION})

    @property
    def models_observed_label_process(self) -> bool:
        """Return whether the contaminant label process is represented."""
        return True
