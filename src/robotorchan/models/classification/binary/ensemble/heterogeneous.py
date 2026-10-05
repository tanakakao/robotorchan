"""Heterogeneous ensembles for binary classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.posterior.ensemble import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)


class HeterogeneousBinaryClassificationEnsemble(BinaryClassificationMixin, nn.Module):
    """Ensemble complete classifiers that share binary class semantics."""

    def __init__(self, *members: nn.Module) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("Heterogeneous classification ensembles require at least two members.")
        for member in members:
            if not callable(getattr(member, "predict_proba", None)):
                raise TypeError("Every ensemble member must expose predict_proba(X).")
            if getattr(member, "num_classes", None) != self.num_classes:
                raise ValueError("Every ensemble member must expose two predictive classes.")
            if tuple(getattr(member, "class_labels", ())) != self.class_labels:
                raise ValueError("Every ensemble member must use class_labels=(0, 1).")
        self.members = nn.ModuleList(members)

    def probability_posterior(
        self,
        X: Tensor,
        **kwargs: object,
    ) -> ClassificationEnsemblePosterior:
        """Return empirical probability posterior across heterogeneous members."""
        probabilities = torch.stack(
            [member.predict_proba(X, **kwargs) for member in self.members],
            dim=0,
        )
        return make_classification_ensemble_posterior(probabilities)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return equally weighted mean member probability."""
        return self.probability_posterior(X, **kwargs).mean

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Sample complete member probabilities from the empirical posterior."""
        return self.probability_posterior(X, **kwargs).rsample(sample_shape)

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return between-member epistemic probability variance."""
        del num_samples
        return self.probability_posterior(X, **kwargs).variance

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        """Return categorical distribution from ensemble-mean probabilities."""
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli observation variance from ensemble-mean probabilities."""
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return entropy of ensemble-mean predictive probabilities."""
        return self.probability_posterior(X, **kwargs).predictive_entropy

    def expected_class_entropy(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return equally weighted mean member entropy."""
        del num_samples
        return self.probability_posterior(X, **kwargs).expected_class_entropy

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return BALD-style disagreement across heterogeneous members."""
        del num_samples
        return self.probability_posterior(X, **kwargs).mutual_information

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary class labels from ensemble-mean probability."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not torch.isfinite(torch.tensor(threshold)):
            raise ValueError("threshold must be finite.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(X, **kwargs)[..., 1] >= threshold).long()

    def latent_posterior(self, X: Tensor, **kwargs: object):
        """Reject a fictitious common latent posterior across backends."""
        del X, kwargs
        raise NotImplementedError(
            "Heterogeneous classification ensembles do not share one latent posterior. "
            "Use probability_posterior() or inspect member models directly."
        )

    def sample_latent(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Reject latent sampling across heterogeneous model families."""
        del X, sample_shape, kwargs
        raise NotImplementedError(
            "Heterogeneous classification ensembles do not share a latent sample space."
        )
