"""Single-task multiclass variational Gaussian process classifier."""

from __future__ import annotations

import torch
from gpytorch.likelihoods import SoftmaxLikelihood
from torch import Tensor
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.validation import validate_class_indices
from robotorchan.models.standard.variational import SingleTaskVariationalGP


class MulticlassSingleTaskGPClassifier(ClassificationModelMixin, SingleTaskVariationalGP):
    """Variational multiclass classifier with one latent GP per class."""

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.PER_CLASS

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        if num_classes < 3:
            raise ValueError("num_classes must be at least 3.")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y
        validate_class_indices(labels, num_classes=num_classes)
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[-2]:
            raise ValueError("train_Y must have shape [n] or [n, 1].")
        self._num_classes = num_classes
        super().__init__(
            train_X=train_X,
            train_Y=labels.to(dtype=train_X.dtype).unsqueeze(-1),
            likelihood=SoftmaxLikelihood(
                num_features=num_classes,
                num_classes=num_classes,
                mixing_weights=False,
            ),
            num_outputs=num_classes,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self._store_raw_tensor("train_Y", labels.detach().clone().long())

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_labels(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        latent = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        return latent.softmax(dim=-1)

    def predict_proba(
        self,
        X: Tensor,
        *,
        num_samples: int = 256,
        **kwargs: object,
    ) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        return samples.mean(dim=0)

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
