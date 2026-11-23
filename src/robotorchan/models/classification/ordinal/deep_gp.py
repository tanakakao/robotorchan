"""Ordinal classification with the existing stochastic Deep Gaussian process."""

from __future__ import annotations

import gpytorch
import torch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch import Tensor
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.ordinal.likelihood import OrderedProbitLikelihood
from robotorchan.models.classification.posterior.probability import (
    ClassificationProbabilityPosterior,
)
from robotorchan.models.classification.validation import validate_class_indices
from robotorchan.models.expressive.deep_gp import SingleTaskDeepGP


class OrdinalSingleTaskDeepGPClassifier(ClassificationModelMixin, SingleTaskDeepGP):
    """Ordered-probit classifier with stochastic DeepGP latent representations.

    Class probabilities are integrated over empirical latent trajectories rather
    than applying Gaussian-probit identities to a non-Gaussian DeepGP posterior.
    """

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.SINGLE

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        **kwargs: object,
    ) -> None:
        if num_classes < 3:
            raise ValueError("num_classes must be at least 3.")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y
        validate_class_indices(labels, num_classes=num_classes)
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[-2]:
            raise ValueError("train_Y must have shape [n] or [n, 1].")
        model_Y = labels.to(dtype=train_X.dtype, device=train_X.device).unsqueeze(-1)
        super().__init__(train_X=train_X, train_Y=model_Y, **kwargs)
        self._num_classes = num_classes
        self.likelihood = OrderedProbitLikelihood(num_classes).to(train_X)
        self._store_raw_tensor("train_Y", labels.detach().clone().long())

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_labels(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    def make_mll(self, num_data: int | None = None) -> DeepApproximateMLL:
        if num_data is None:
            num_data = self.raw_train_X.shape[-2]
        if num_data <= 0:
            raise ValueError("num_data must be positive.")
        return DeepApproximateMLL(
            VariationalELBO(likelihood=self.likelihood, model=self, num_data=num_data)
        )

    def training_loss(
        self,
        X: Tensor | None = None,
        Y: Tensor | None = None,
        *,
        num_data: int | None = None,
        num_likelihood_samples: int = 8,
    ) -> Tensor:
        X = self.raw_train_X if X is None else X
        Y = self.raw_train_Y if Y is None else Y
        labels = Y.squeeze(-1) if Y.ndim == 2 and Y.shape[-1] == 1 else Y
        validate_class_indices(labels, num_classes=self.num_classes)
        if labels.ndim != 1 or labels.shape[0] != X.shape[-2]:
            raise ValueError("X and Y must contain the same number of rows.")
        if num_likelihood_samples <= 0:
            raise ValueError("num_likelihood_samples must be positive.")
        with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            return -self.make_mll(num_data=num_data)(self(X), labels.long())

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        *,
        num_samples: int | None = None,
    ) -> Tensor:
        latent = self.sample_latent(X, sample_shape=sample_shape, num_samples=num_samples)
        return self.likelihood.class_probabilities(latent.squeeze(-1))

    def predict_proba(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        resolved = self.posterior_samples if num_samples is None else int(num_samples)
        if resolved < 2:
            raise ValueError("num_samples must be at least 2.")
        # Posterior stores correlated DeepGP trajectories. Average class
        # probabilities over these trajectories, not over a Gaussian surrogate.
        latent = self.latent_posterior(X, num_samples=resolved)
        return self.likelihood.class_probabilities(
            latent._samples.squeeze(-1)
        ).mean(dim=0)

    def probability_posterior(
        self,
        X: Tensor,
        *,
        num_samples: int | None = None,
    ) -> ClassificationProbabilityPosterior:
        resolved = self.posterior_samples if num_samples is None else int(num_samples)
        latent = self.latent_posterior(X, num_samples=resolved)
        samples = self.likelihood.class_probabilities(latent._samples.squeeze(-1))
        return ClassificationProbabilityPosterior(
            latent,
            sample_transform=lambda values: self.likelihood.class_probabilities(
                values.squeeze(-1)
            ),
            mean=samples.mean(dim=0),
            variance=samples.var(dim=0, unbiased=False),
        )

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
