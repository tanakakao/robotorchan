"""Active-learning acquisitions for probabilistic classification models."""

from __future__ import annotations

import math
from typing import Protocol, cast

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

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor: ...

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor: ...


def _classification_model(
    model: Model,
    *,
    required_methods: tuple[str, ...] = (),
) -> ClassificationAcquisitionModel:
    required = (
        "predict_proba",
        "sample_class_probabilities",
        "latent_posterior",
        *required_methods,
    )
    missing = tuple(name for name in required if not callable(getattr(model, name, None)))
    if missing:
        names = ", ".join(missing)
        raise TypeError(
            "model must implement the robotorchan classification prediction "
            f"contract; missing methods: {names}"
        )
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
        self.classification_model = _classification_model(
            model,
            required_methods=("probability_variance",),
        )
        self.num_samples = num_samples

    def forward(self, X: Tensor) -> Tensor:
        """Average class-probability variance across classes."""
        _require_q_one(X, self.__class__.__name__)
        variance = self.classification_model.probability_variance(
            X,
            num_samples=self.num_samples,
        )
        return variance.mean(dim=-1).squeeze(-1)


class BALD(AcquisitionFunction):
    """Bayesian active learning by disagreement for classification."""

    def __init__(self, model: Model, *, num_samples: int = 128) -> None:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        super().__init__(model=model)
        self.classification_model = _classification_model(
            model,
            required_methods=("mutual_information",),
        )
        self.num_samples = num_samples

    def forward(self, X: Tensor) -> Tensor:
        """Estimate mutual information between labels and latent parameters."""
        _require_q_one(X, self.__class__.__name__)
        mutual_information = self.classification_model.mutual_information(
            X,
            num_samples=self.num_samples,
        )
        return mutual_information.squeeze(-1)


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
