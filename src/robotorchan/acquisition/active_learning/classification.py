"""Active-learning acquisitions for probabilistic classification models."""

from __future__ import annotations

from typing import Protocol, cast

import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


class ClassificationAcquisitionModel(Protocol):
    """Prediction surface required by classification acquisitions."""

    num_classes: int

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor: ...

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor: ...

    def latent_posterior(self, X: Tensor, **kwargs: object) -> object: ...


def _classification_model(model: Model) -> ClassificationAcquisitionModel:
    required = ("predict_proba", "sample_class_probabilities", "latent_posterior")
    if not all(callable(getattr(model, name, None)) for name in required):
        raise TypeError("model must implement the robotorchan classification prediction contract")
    return cast(ClassificationAcquisitionModel, model)


def _require_q_one(X: Tensor, name: str) -> None:
    if X.shape[-2] != 1:
        raise ValueError(f"{name} supports q=1.")


def _entropy(probabilities: Tensor) -> Tensor:
    probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)


class PredictiveEntropy(AcquisitionFunction):
    """Prefer inputs with high posterior-predictive class entropy."""

    def __init__(self, model: Model) -> None:
        super().__init__(model=model)
        self.classification_model = _classification_model(model)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate predictive entropy for q=1 candidates."""
        _require_q_one(X, self.__class__.__name__)
        probabilities = self.classification_model.predict_proba(X)
        return _entropy(probabilities).squeeze(-1)


class MarginUncertainty(AcquisitionFunction):
    """Prefer small gaps between the two most probable classes."""

    def __init__(self, model: Model) -> None:
        super().__init__(model=model)
        self.classification_model = _classification_model(model)

    def forward(self, X: Tensor) -> Tensor:
        """Return one minus the top-two probability margin."""
        _require_q_one(X, self.__class__.__name__)
        probabilities = self.classification_model.predict_proba(X)
        if probabilities.shape[-1] < 2:
            raise ValueError("MarginUncertainty requires at least two classes.")
        top_two = probabilities.topk(k=2, dim=-1).values
        score = 1.0 - (top_two[..., 0] - top_two[..., 1])
        return score.squeeze(-1)


class ProbabilityVariance(AcquisitionFunction):
    """Prefer inputs with high posterior variance of class probabilities."""

    def __init__(self, model: Model, *, num_samples: int = 128) -> None:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        super().__init__(model=model)
        self.classification_model = _classification_model(model)
        self.num_samples = num_samples

    def forward(self, X: Tensor) -> Tensor:
        """Average class-probability variance across classes."""
        _require_q_one(X, self.__class__.__name__)
        samples = self.classification_model.sample_class_probabilities(
            X, sample_shape=torch.Size([self.num_samples])
        )
        return samples.var(dim=0, unbiased=False).mean(dim=-1).squeeze(-1)


class BALD(AcquisitionFunction):
    """Bayesian active learning by disagreement for classification."""

    def __init__(self, model: Model, *, num_samples: int = 128) -> None:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        super().__init__(model=model)
        self.classification_model = _classification_model(model)
        self.num_samples = num_samples

    def forward(self, X: Tensor) -> Tensor:
        """Estimate mutual information between labels and latent parameters."""
        _require_q_one(X, self.__class__.__name__)
        samples = self.classification_model.sample_class_probabilities(
            X, sample_shape=torch.Size([self.num_samples])
        )
        predictive_entropy = _entropy(samples.mean(dim=0))
        expected_entropy = _entropy(samples).mean(dim=0)
        return (predictive_entropy - expected_entropy).clamp_min(0.0).squeeze(-1)


class LatentStraddle(AcquisitionFunction):
    """Binary classification straddle score around the latent decision boundary."""

    def __init__(self, model: Model, *, beta: float = 1.96) -> None:
        if not math.isfinite(beta):
            raise ValueError("beta must be finite.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        super().__init__(model=model)
        self.classification_model = _classification_model(model)
        if self.classification_model.num_classes != 2:
            raise ValueError("LatentStraddle currently requires binary classification.")
        self.beta = beta

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate straddle around latent zero for q=1 candidates."""
        _require_q_one(X, self.__class__.__name__)
        posterior = self.classification_model.latent_posterior(X)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(0.0)
        score = self.beta * variance.sqrt() - mean.abs()
        return score.squeeze(-1).squeeze(-1)
