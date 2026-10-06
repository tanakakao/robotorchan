"""Reliability diagnostics for probabilistic classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClassificationReliabilityEvaluator(nn.Module):
    """Evaluate distinct predictive, epistemic, and input-space reliability signals."""

    def __init__(
        self,
        model: nn.Module,
        *,
        reference_X: Tensor | None = None,
        covariance_regularization: float = 1e-6,
    ) -> None:
        super().__init__()
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must expose predict_proba(X).")
        if covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive.")
        self.model = model
        self.covariance_regularization = covariance_regularization
        if reference_X is None:
            reference_X = getattr(model, "raw_train_X", None)
        if reference_X is None:
            self.register_buffer("reference_mean", None)
            self.register_buffer("reference_precision", None)
            return
        if reference_X.ndim != 2:
            raise ValueError("reference_X must have shape n x d.")
        if reference_X.shape[0] < 2:
            raise ValueError("reference_X must contain at least two rows.")
        if not torch.is_floating_point(reference_X) or not torch.isfinite(reference_X).all():
            raise ValueError("reference_X must be a finite floating-point tensor.")
        mean = reference_X.mean(dim=0)
        centered = reference_X - mean
        covariance = centered.transpose(-1, -2) @ centered / (reference_X.shape[0] - 1)
        eye = torch.eye(
            reference_X.shape[-1],
            dtype=reference_X.dtype,
            device=reference_X.device,
        )
        precision = torch.linalg.pinv(
            covariance + covariance_regularization * eye,
            hermitian=True,
        )
        self.register_buffer("reference_mean", mean)
        self.register_buffer("reference_precision", precision)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return predictive class entropy as an ambiguity signal."""
        method = getattr(self.model, "predictive_entropy", None)
        if callable(method):
            return method(X, **kwargs)
        probabilities = self.model.predict_proba(X, **kwargs)
        safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        return -torch.special.xlogy(safe, safe).sum(dim=-1)

    def epistemic_disagreement(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return mutual information when the model exposes epistemic uncertainty."""
        method = getattr(self.model, "mutual_information", None)
        if not callable(method):
            raise NotImplementedError(
                "The model does not expose mutual_information; "
                "epistemic disagreement is unavailable."
            )
        return method(X, num_samples=num_samples, **kwargs)

    def input_ood_score(self, X: Tensor) -> Tensor:
        """Return Mahalanobis distance from the reference input distribution."""
        if self.reference_mean is None or self.reference_precision is None:
            raise NotImplementedError(
                "Input OOD scoring requires reference_X or model.raw_train_X."
            )
        if X.shape[-1] != self.reference_mean.shape[-1]:
            raise ValueError("X and reference_X must have the same feature dimension.")
        delta = X - self.reference_mean.to(X)
        precision = self.reference_precision.to(X)
        squared = torch.einsum("...d,de,...e->...", delta, precision, delta)
        return squared.clamp_min(0.0).sqrt()


def max_probability_reliability(probabilities: Tensor) -> Tensor:
    """Return maximum predicted class probability as a confidence diagnostic."""
    if probabilities.ndim < 1 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities must contain at least two classes.")
    if not torch.is_floating_point(probabilities) or not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite floating-point values.")
    if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("Class probabilities must sum to one.")
    return probabilities.max(dim=-1).values
