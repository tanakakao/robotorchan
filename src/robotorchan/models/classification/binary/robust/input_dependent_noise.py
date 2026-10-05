"""Binary classification with input-dependent label-flip probabilities."""

from __future__ import annotations

from collections.abc import Callable

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

NoiseRateFunction = Callable[[Tensor], Tensor]


class InputDependentLabelNoiseLikelihood(BernoulliLikelihood):
    """Bernoulli likelihood whose false-positive/negative rates depend on input."""

    def __init__(
        self,
        train_X: Tensor,
        noise_rate_model: nn.Module | NoiseRateFunction,
        *,
        max_flip_probability: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < max_flip_probability <= 0.5:
            raise ValueError("max_flip_probability must satisfy 0 < value <= 0.5.")
        if isinstance(noise_rate_model, nn.Module):
            self.noise_rate_model = noise_rate_model
            self._noise_rate_function = None
        elif callable(noise_rate_model):
            self.noise_rate_model = None
            self._noise_rate_function = noise_rate_model
        else:
            raise TypeError("noise_rate_model must be an nn.Module or callable.")
        self.register_buffer("train_X", train_X.detach().clone())
        self.max_flip_probability = float(max_flip_probability)

    def flip_probabilities(self, X: Tensor) -> Tensor:
        """Return false-positive and false-negative probabilities in the final axis."""
        model = self.noise_rate_model
        rate_function = self._noise_rate_function
        rates = model(X) if model is not None else rate_function(X)
        if rates.shape != (*X.shape[:-1], 2):
            raise ValueError("noise_rate_model must return shape [..., 2].")
        if not torch.isfinite(rates).all():
            raise ValueError("noise_rate_model output must be finite.")
        if torch.any(rates < 0.0) or torch.any(rates >= self.max_flip_probability):
            raise ValueError(
                "noise_rate_model output must satisfy 0 <= rate < max_flip_probability."
            )
        return rates.to(dtype=X.dtype, device=X.device)

    def corrupt_positive_probability(
        self,
        clean_probability: Tensor,
        X: Tensor,
    ) -> Tensor:
        """Apply the input-dependent binary label-flip channel."""
        rates = self.flip_probabilities(X)
        false_positive = rates[..., 0]
        false_negative = rates[..., 1]
        if clean_probability.shape[-1:] == (1,):
            clean_probability = clean_probability.squeeze(-1)
        return false_positive + (1.0 - false_positive - false_negative) * clean_probability

    def forward(
        self,
        function_samples: Tensor,
        *,
        X: Tensor | None = None,
        **kwargs: object,
    ) -> Bernoulli:
        """Return corrupted Bernoulli distributions for latent samples."""
        inputs = self.train_X if X is None else X
        clean = BernoulliLikelihood.forward(self, function_samples, **kwargs).probs
        return Bernoulli(probs=self.corrupt_positive_probability(clean, inputs))

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *,
        X: Tensor | None = None,
        **kwargs: object,
    ) -> Bernoulli:
        """Return input-dependent corrupted posterior predictions."""
        inputs = self.train_X if X is None else X
        clean = BernoulliLikelihood.marginal(self, function_dist, **kwargs).probs
        return Bernoulli(probs=self.corrupt_positive_probability(clean, inputs))

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        """Integrate corrupted log probability using the stored training inputs."""

        def log_prob(function_samples: Tensor) -> Tensor:
            return self.forward(function_samples).log_prob(observations)

        return self.quadrature(log_prob, function_dist)


class InputDependentLabelNoiseBinarySingleTaskGPClassifier(
    RobustBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Binary GP classifier with an externally composable input-dependent noise model."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        noise_rate_model: nn.Module | NoiseRateFunction,
        max_flip_probability: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self.likelihood = InputDependentLabelNoiseLikelihood(
            train_X,
            noise_rate_model,
            max_flip_probability=max_flip_probability,
        )

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Bernoulli:
        """Return observed-label predictions using input-dependent flip rates."""
        latent = self.latent_posterior(X, **kwargs)
        return self.likelihood.marginal(latent.distribution, X=X)

    def predict_clean_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return posterior probabilities before the label-flip channel."""
        latent = self.latent_posterior(X, **kwargs)
        positive = BernoulliLikelihood.marginal(self.likelihood, latent.distribution).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw observed-label probabilities with input-dependent flip rates."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        clean = BernoulliLikelihood.forward(self.likelihood, latent_samples).probs
        positive = self.likelihood.corrupt_positive_probability(clean, X)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def sample_clean_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw clean class probabilities before input-dependent corruption."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        positive = BernoulliLikelihood.forward(self.likelihood, latent_samples).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def flip_probabilities(self, X: Tensor) -> Tensor:
        """Return input-dependent false-positive and false-negative rates."""
        return self.likelihood.flip_probabilities(X)

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return input-dependent label-noise robustness metadata."""
        return frozenset({ClassificationRobustnessType.INPUT_DEPENDENT_LABEL_NOISE})

    @property
    def models_observed_label_process(self) -> bool:
        """Return whether the observed-label process is explicitly represented."""
        return True
