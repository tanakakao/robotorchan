"""Independent classification-model composition."""

from __future__ import annotations

import torch
from botorch.models.model import ModelList
from botorch.posteriors import PosteriorList
from torch import Tensor

from robotorchan.models.classification.base import ClassificationModelMixin


class ClassificationModelList(ModelList):
    """BoTorch model list for independent classification surrogate models.

    Child models keep their own likelihoods, ELBOs, class metadata, and
    training data. The container only composes latent posteriors and
    classification predictions; it does not invent a shared likelihood.
    """

    def __init__(self, *models: ClassificationModelMixin) -> None:
        """Initialize a heterogeneous list of classification models."""
        if not models:
            raise ValueError("ClassificationModelList requires at least one model.")
        if not all(isinstance(model, ClassificationModelMixin) for model in models):
            raise TypeError("All child models must implement ClassificationModelMixin.")
        super().__init__(*models)

    @property
    def classification_metadata(self) -> tuple[object, ...]:
        """Return classification metadata for every child model."""
        return tuple(model.classification_metadata for model in self.models)

    @property
    def raw_train_Xs(self) -> tuple[Tensor | None, ...]:
        """Return each child model caller-supplied training inputs."""
        return tuple(getattr(model, "raw_train_X", None) for model in self.models)

    @property
    def raw_train_Ys(self) -> tuple[Tensor | None, ...]:
        """Return each child model caller-supplied training labels."""
        return tuple(getattr(model, "raw_train_Y", None) for model in self.models)

    def latent_posterior(self, X: Tensor, **kwargs: object) -> PosteriorList:
        """Return the independent latent posterior from every child model."""
        return PosteriorList(*(model.latent_posterior(X, **kwargs) for model in self.models))

    def make_mlls(self) -> tuple[object, ...]:
        """Construct each child model training objective independently."""
        return tuple(model.make_mll() for model in self.models)

    def predict_proba(self, X: Tensor, **kwargs: object) -> tuple[Tensor, ...]:
        """Return class probabilities for every child without forcing shapes."""
        return tuple(model.predict_proba(X, **kwargs) for model in self.models)

    def predict_class(self, X: Tensor, **kwargs: object) -> tuple[Tensor, ...]:
        """Return discrete predictions for every child model."""
        return tuple(model.predict_class(X, **kwargs) for model in self.models)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> tuple[Tensor, ...]:
        """Return predictive entropy for every child model."""
        return tuple(model.predictive_entropy(X, **kwargs) for model in self.models)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, ...]:
        """Draw class-probability samples independently from every child."""
        return tuple(
            model.sample_class_probabilities(X, sample_shape=sample_shape, **kwargs)
            for model in self.models
        )
