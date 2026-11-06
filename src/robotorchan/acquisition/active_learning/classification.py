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
    required = required_methods
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
        self.classification_model = _classification_model(
            model,
            required_methods=("predict_proba",),
        )

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate predictive entropy for q=1 candidates."""
        _require_q_one(X, self.__class__.__name__)
        probabilities = self.classification_model.predict_proba(X)
        return _entropy(probabilities).squeeze(-1)


class MarginUncertainty(AcquisitionFunction):
    """Prefer small gaps between the two most probable classes."""

    def __init__(self, model: Model) -> None:
        super().__init__(model=model)
        self.classification_model = _classification_model(
            model,
            required_methods=("predict_proba",),
        )

    def forward(self, X: Tensor) -> Tensor:
        """Return one minus the top-two probability margin."""
        _require_q_one(X, self.__class__.__name__)
        probabilities = self.classification_model.predict_proba(X)
        if probabilities.shape[-1] < 2:
            raise ValueError("MarginUncertainty requires at least two classes.")
        top_two = probabilities.topk(k=2, dim=-1).values
        score = 1.0 - (top_two[..., 0] - top_two[..., 1])
        return score.squeeze(-1)


class qPredictiveEntropy(PredictiveEntropy):
    """Mean marginal class entropy over a batch; not joint label entropy."""

    def forward(self, X: Tensor) -> Tensor:
        probabilities = self.classification_model.predict_proba(X)
        return _entropy(probabilities).reshape(X.shape[:-1]).mean(dim=-1)


class qMarginUncertainty(MarginUncertainty):
    """Mean marginal top-two class uncertainty over a batch."""

    def forward(self, X: Tensor) -> Tensor:
        probabilities = self.classification_model.predict_proba(X)
        if probabilities.shape[-1] < 2:
            raise ValueError("qMarginUncertainty requires at least two classes.")
        top_two = probabilities.topk(k=2, dim=-1).values
        score = 1.0 - (top_two[..., 0] - top_two[..., 1])
        return score.reshape(X.shape[:-1]).mean(dim=-1)


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


class qProbabilityVariance(ProbabilityVariance):
    """Mean marginal class-probability variance; no joint batch information."""

    def forward(self, X: Tensor) -> Tensor:
        variance = self.classification_model.probability_variance(
            X,
            num_samples=self.num_samples,
        )
        return variance.mean(dim=-1).reshape(X.shape[:-1]).mean(dim=-1)


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


class qBALD(BALD):
    """Joint binary-label information gain using coherent posterior samples.

    Enumerates 2**q label vectors, so q is limited to eight. Unlike a
    marginal BALD average, this accounts for redundant candidate information.
    """

    def forward(self, X: Tensor) -> Tensor:
        q = X.shape[-2]
        if not 1 <= q <= 8:
            raise ValueError("qBALD requires 1 <= q <= 8 for exact label enumeration.")
        if self.classification_model.num_classes != 2:
            raise ValueError("qBALD currently requires binary classification.")
        samples = self.classification_model.sample_class_probabilities(
            X, sample_shape=torch.Size([self.num_samples])
        )
        if samples.shape[-2:] != (q, 2):
            raise ValueError("sample_class_probabilities must preserve q and class dimensions.")
        labels = torch.cartesian_prod(
            *[torch.arange(2, device=X.device) for _ in range(q)]
        ).reshape(-1, q)
        selected = torch.where(
            labels == 1,
            samples[..., 1].unsqueeze(-2),
            samples[..., 0].unsqueeze(-2),
        )
        conditional_joint = selected.prod(dim=-1)
        predictive_joint = conditional_joint.mean(dim=0)
        predictive_entropy = -torch.special.xlogy(predictive_joint, predictive_joint).sum(dim=-1)
        conditional_entropy = (
            -torch.special.xlogy(conditional_joint, conditional_joint).sum(dim=-1).mean(dim=0)
        )
        return (predictive_entropy - conditional_entropy).clamp_min(0.0)


class LatentStraddle(AcquisitionFunction):
    """Binary classification straddle score around the latent decision boundary."""

    def __init__(self, model: Model, *, beta: float = 1.96) -> None:
        if not math.isfinite(beta):
            raise ValueError("beta must be finite.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        super().__init__(model=model)
        self.classification_model = _classification_model(
            model,
            required_methods=("latent_posterior",),
        )
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


class qLatentStraddle(LatentStraddle):
    """Mean marginal latent decision-boundary straddle over q candidates."""

    def forward(self, X: Tensor) -> Tensor:
        posterior = self.classification_model.latent_posterior(X)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(0.0)
        score = self.beta * variance.sqrt() - mean.abs()
        return score.reshape(X.shape[:-1]).mean(dim=-1)
