"""Homogeneous GP ensembles for binary classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.posterior.ensemble import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)


class GPBinaryClassificationEnsemble(BinaryClassificationMixin, nn.Module):
    """Model-uncertainty ensemble of independently fitted binary GP classifiers."""

    def __init__(self, *members: BinarySingleTaskGPClassifier) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("GP classification ensembles require at least two members.")
        if not all(isinstance(member, BinarySingleTaskGPClassifier) for member in members):
            raise TypeError("All members must be BinarySingleTaskGPClassifier instances.")
        reference_X = members[0].raw_train_X
        reference_Y = members[0].raw_train_Y
        for member in members[1:]:
            if member.raw_train_X.shape != reference_X.shape:
                raise ValueError("All GP ensemble members must use matching training input shapes.")
            if member.raw_train_Y.shape != reference_Y.shape:
                raise ValueError("All GP ensemble members must use matching training label shapes.")
            if not torch.equal(member.raw_train_X, reference_X):
                raise ValueError("All GP ensemble members must use the same training inputs.")
            if not torch.equal(member.raw_train_Y, reference_Y):
                raise ValueError("All GP ensemble members must use the same training labels.")
        self.members = nn.ModuleList(members)

    @property
    def raw_train_X(self) -> Tensor:
        """Return the shared caller-supplied training inputs."""
        return self.members[0].raw_train_X

    @property
    def raw_train_Y(self) -> Tensor:
        """Return the shared caller-supplied training labels."""
        return self.members[0].raw_train_Y

    def probability_posterior(
        self,
        X: Tensor,
        **kwargs: object,
    ) -> ClassificationEnsemblePosterior:
        """Return the empirical posterior over complete GP member predictions."""
        probabilities = torch.stack(
            [member.predict_proba(X, **kwargs) for member in self.members],
            dim=0,
        )
        return make_classification_ensemble_posterior(probabilities)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return the ensemble-mean predictive class probability."""
        return self.probability_posterior(X, **kwargs).mean

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Sample complete GP member predictive probabilities."""
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
        """Return a categorical distribution from ensemble-mean probabilities."""
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli observation variance from ensemble-mean probabilities."""
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return entropy of the ensemble-mean predictive distribution."""
        return self.probability_posterior(X, **kwargs).predictive_entropy

    def expected_class_entropy(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return mean predictive entropy across complete GP members."""
        del num_samples
        return self.probability_posterior(X, **kwargs).expected_class_entropy

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return between-member BALD-style disagreement."""
        del num_samples
        return self.probability_posterior(X, **kwargs).mutual_information

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary labels from ensemble-mean positive probability."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not torch.isfinite(torch.tensor(threshold)):
            raise ValueError("threshold must be finite.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        positive = self.predict_proba(X, **kwargs)[..., 1]
        return (positive >= threshold).to(dtype=torch.long)

    def latent_posterior(self, X: Tensor, **kwargs: object):
        """Reject a fictitious single latent posterior for the ensemble."""
        del X, kwargs
        raise NotImplementedError(
            "A GP classification ensemble has multiple independent latent posteriors. "
            "Use member_latent_posteriors() or probability_posterior()."
        )

    def member_latent_posteriors(self, X: Tensor, **kwargs: object) -> tuple[object, ...]:
        """Return each GP member latent posterior without collapsing semantics."""
        return tuple(member.latent_posterior(X, **kwargs) for member in self.members)

    def make_mlls(self) -> tuple[object, ...]:
        """Construct independent variational objectives for all GP members."""
        return tuple(member.make_mll() for member in self.members)
