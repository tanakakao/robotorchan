"""Split-conformal prediction sets for probabilistic classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def classification_nonconformity_scores(
    probabilities: Tensor,
    targets: Tensor,
) -> Tensor:
    """Return inverse-probability nonconformity scores for observed labels."""
    if probabilities.ndim < 2 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities must have shape ... x num_classes with num_classes >= 2.")
    if not torch.is_floating_point(probabilities) or not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite floating-point values.")
    if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("Class probabilities must sum to one.")
    if targets.shape != probabilities.shape[:-1]:
        raise ValueError("targets must match the leading probability dimensions.")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must contain integer class indices.")
    if targets.device != probabilities.device:
        raise ValueError("targets and probabilities must be on the same device.")
    if torch.any((targets < 0) | (targets >= probabilities.shape[-1])):
        raise ValueError("targets contain an invalid class index.")
    observed = probabilities.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    return 1.0 - observed


def conformal_quantile(scores: Tensor, *, alpha: float) -> Tensor:
    """Return the finite-sample split-conformal quantile."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor.")
    if not torch.is_floating_point(scores) or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite floating-point values.")
    if not isinstance(alpha, int | float) or isinstance(alpha, bool):
        raise TypeError("alpha must be a real number.")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    n = scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(rank, n)
    return scores.sort().values[rank - 1]


class SplitConformalClassifier(nn.Module):
    """Wrap a probabilistic classifier with split-conformal prediction sets."""

    def __init__(self, model: nn.Module, *, alpha: float = 0.1) -> None:
        super().__init__()
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must expose predict_proba(X).")
        if not isinstance(alpha, int | float) or isinstance(alpha, bool):
            raise TypeError("alpha must be a real number.")
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        self.model = model
        self.alpha = float(alpha)
        self.register_buffer("quantile", None)

    @property
    def is_calibrated(self) -> bool:
        """Whether a held-out conformal calibration set has been supplied."""
        return self.quantile is not None

    def calibrate(self, calibration_X: Tensor, calibration_Y: Tensor) -> None:
        """Fit the conformal threshold on held-out exchangeable observations."""
        with torch.no_grad():
            probabilities = self.model.predict_proba(calibration_X)
            scores = classification_nonconformity_scores(probabilities, calibration_Y)
            quantile = conformal_quantile(scores.reshape(-1), alpha=self.alpha)
        self.quantile = quantile.detach()

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Delegate predictive probabilities without changing the base model."""
        return self.model.predict_proba(X, **kwargs)

    def prediction_set(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return a boolean class-membership mask for each candidate."""
        if self.quantile is None:
            raise RuntimeError("Call calibrate() before prediction_set().")
        probabilities = self.predict_proba(X, **kwargs)
        threshold = 1.0 - self.quantile.to(probabilities)
        return probabilities >= threshold

    def prediction_set_size(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return the number of classes retained in each prediction set."""
        return self.prediction_set(X, **kwargs).sum(dim=-1)
