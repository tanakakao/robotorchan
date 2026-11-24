"""Multiclass and ordinal-label heterogeneous classification ensembles.

An ensemble posterior is an empirical distribution over *member class
probabilities*, not an invented shared GP latent posterior.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.multiclass.non_gp import (
    GradientBoostingMulticlassClassifier,
    GradientBoostingOrdinalClassifier,
)
from robotorchan.models.classification.posterior.ensemble import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)


class MulticlassClassificationEnsemble(ClassificationModelMixin, nn.Module):
    """Weighted ensemble of fitted GP and non-GP multiclass classifiers."""

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.CUSTOM

    def __init__(self, *members: nn.Module, weights: Tensor | None = None) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("Classification ensembles require at least two members.")
        reference_classes = getattr(members[0], "num_classes", None)
        reference_labels = getattr(members[0], "class_labels", None)
        if not isinstance(reference_classes, int) or reference_classes < 3:
            raise ValueError("Members must expose at least three classes.")
        for member in members:
            if not callable(getattr(member, "predict_proba", None)):
                raise TypeError("Every ensemble member must expose predict_proba(X).")
            if getattr(member, "num_classes", None) != reference_classes:
                raise ValueError("Every ensemble member must use the same num_classes.")
            if getattr(member, "class_labels", None) != reference_labels:
                raise ValueError("Every ensemble member must use identical class_labels.")
        self._num_classes = reference_classes
        self._class_labels = tuple(reference_labels)
        self.members = nn.ModuleList(members)
        if weights is None:
            weights = torch.ones(len(members), dtype=torch.get_default_dtype())
        if weights.ndim != 1 or weights.shape != (len(members),):
            raise ValueError("weights must have shape (num_members,).")
        if not torch.is_floating_point(weights) or not torch.isfinite(weights).all():
            raise ValueError("weights must contain finite floating-point values.")
        if torch.any(weights < 0) or not bool(weights.sum() > 0):
            raise ValueError("weights must be nonnegative with a positive sum.")
        self.register_buffer("member_weights", weights / weights.sum())

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_labels(self) -> tuple[object, ...]:
        return self._class_labels

    def probability_posterior(self, X: Tensor, **kwargs: object) -> ClassificationEnsemblePosterior:
        target = (*X.shape[:-1], self.num_classes)
        normalized = []
        for member in self.members:
            probabilities = member.predict_proba(X, **kwargs)
            if probabilities.shape != target:
                if (
                    X.ndim >= 3
                    and X.shape[-2] == 1
                    and probabilities.shape == (*X.shape[:-2], self.num_classes)
                ):
                    probabilities = probabilities.unsqueeze(-2)
                else:
                    raise ValueError("Ensemble member probability shape must preserve candidates.")
            normalized.append(probabilities)
        stacked = torch.stack(normalized)
        weights = self.member_weights.to(dtype=stacked.dtype, device=stacked.device)
        return make_classification_ensemble_posterior(stacked, weights=weights)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.probability_posterior(X, **kwargs).mean

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        return self.probability_posterior(X, **kwargs).rsample(sample_shape)

    def probability_variance(
        self, X: Tensor, *, num_samples: int = 128, **kwargs: object
    ) -> Tensor:
        del num_samples
        return self.probability_posterior(X, **kwargs).variance

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.probability_posterior(X, **kwargs).predictive_entropy

    def expected_class_entropy(
        self, X: Tensor, *, num_samples: int = 128, **kwargs: object
    ) -> Tensor:
        del num_samples
        return self.probability_posterior(X, **kwargs).expected_class_entropy

    def mutual_information(self, X: Tensor, *, num_samples: int = 128, **kwargs: object) -> Tensor:
        del num_samples
        return self.probability_posterior(X, **kwargs).mutual_information

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)

    def latent_posterior(self, X: Tensor, **kwargs: object):
        del X, kwargs
        raise NotImplementedError(
            "Ensemble members do not share a GP latent posterior; "
            "use probability_posterior() or inspect individual members."
        )

    def sample_latent(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        del X, sample_shape, kwargs
        raise NotImplementedError("Ensembles do not share a latent sample space.")


class OrdinalClassificationEnsemble(MulticlassClassificationEnsemble):
    """Ensemble over ordered category labels; does not assume latent GP coherence."""


class BootstrapMulticlassClassificationEnsemble(MulticlassClassificationEnsemble):
    """Bootstrap complete non-GP classifiers and retain member epistemic spread."""

    member_class: ClassVar[type[nn.Module]] = GradientBoostingMulticlassClassifier

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        n_members: int = 8,
        random_state: int | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 else train_Y
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[0]:
            raise ValueError("train_Y must contain one label per row.")
        if not isinstance(n_members, int) or n_members < 2:
            raise ValueError("n_members must be at least two.")
        generator = torch.Generator(device="cpu")
        if random_state is None:
            generator.seed()
        else:
            generator.manual_seed(random_state)
        members = []
        cpu_labels = labels.detach().cpu()
        for index in range(n_members):
            for _ in range(100):
                sampled = torch.randint(len(labels), (len(labels),), generator=generator)
                if torch.unique(cpu_labels[sampled]).numel() == num_classes:
                    break
            else:
                raise RuntimeError("Could not draw a bootstrap sample containing all classes.")
            indices = sampled.to(device=train_X.device)
            kwargs = dict(estimator_kwargs)
            kwargs.setdefault(
                "random_state",
                None if random_state is None else random_state + index,
            )
            members.append(
                self.member_class(
                    train_X[indices], labels[indices], num_classes=num_classes, **kwargs
                )
            )
        super().__init__(*members)
        self.n_members = n_members

    def fit(self) -> None:
        """Fit all bootstrap members on their independently resampled data."""
        for member in self.members:
            member.fit()


class BootstrapGradientBoostingMulticlassClassifier(BootstrapMulticlassClassificationEnsemble):
    """Bootstrap gradient boosting with an empirical multiclass posterior."""


class BootstrapGradientBoostingOrdinalClassifier(BootstrapMulticlassClassificationEnsemble):
    """Bootstrap gradient boosting over ordinal labels (without ordered thresholds)."""

    member_class = GradientBoostingOrdinalClassifier
