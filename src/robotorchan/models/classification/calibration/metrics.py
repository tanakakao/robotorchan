"""Metrics for evaluating classification probability calibration."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_probabilities_and_targets(
    probabilities: Tensor,
    targets: Tensor,
) -> None:
    if probabilities.ndim < 2 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities must have shape (..., num_classes) with num_classes >= 2.")
    if not torch.is_floating_point(probabilities):
        raise ValueError("probabilities must use a floating-point dtype.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite.")
    if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("Class probabilities must sum to one.")
    if targets.shape != probabilities.shape[:-1]:
        raise ValueError("targets must match probability leading dimensions.")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must contain integer class indices.")
    if torch.any((targets < 0) | (targets >= probabilities.shape[-1])):
        raise ValueError("targets contain an invalid class index.")


def classification_nll(probabilities: Tensor, targets: Tensor) -> Tensor:
    """Return mean negative log likelihood for class probabilities."""
    _validate_probabilities_and_targets(probabilities, targets)
    tiny = torch.finfo(probabilities.dtype).tiny
    log_probabilities = probabilities.clamp_min(tiny).log()
    return F.nll_loss(
        log_probabilities.reshape(-1, probabilities.shape[-1]),
        targets.reshape(-1),
    )


def brier_score(probabilities: Tensor, targets: Tensor) -> Tensor:
    """Return multiclass Brier score averaged over observations."""
    _validate_probabilities_and_targets(probabilities, targets)
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities)
    return (probabilities - one_hot).square().sum(dim=-1).mean()


def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    n_bins: int = 15,
) -> Tensor:
    """Return confidence-based expected calibration error."""
    _validate_probabilities_and_targets(probabilities, targets)
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    flat_targets = targets.reshape(-1)
    confidence, predictions = flat_probabilities.max(dim=-1)
    correct = predictions.eq(flat_targets).to(probabilities.dtype)
    boundaries = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    bin_indices = torch.bucketize(confidence, boundaries[1:-1], right=True)
    error = probabilities.new_zeros(())
    count = confidence.numel()
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if mask.any():
            weight = mask.sum().to(probabilities.dtype) / count
            accuracy = correct[mask].mean()
            mean_confidence = confidence[mask].mean()
            error = error + weight * (accuracy - mean_confidence).abs()
    return error


def maximum_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    n_bins: int = 15,
) -> Tensor:
    """Return maximum confidence-calibration gap across non-empty bins."""
    _validate_probabilities_and_targets(probabilities, targets)
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    flat_targets = targets.reshape(-1)
    confidence, predictions = flat_probabilities.max(dim=-1)
    correct = predictions.eq(flat_targets).to(probabilities.dtype)
    boundaries = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    bin_indices = torch.bucketize(confidence, boundaries[1:-1], right=True)
    maximum = probabilities.new_zeros(())
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if mask.any():
            gap = (correct[mask].mean() - confidence[mask].mean()).abs()
            maximum = torch.maximum(maximum, gap)
    return maximum
