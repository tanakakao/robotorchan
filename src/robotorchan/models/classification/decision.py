"""Bayes decisions for multiclass and ordered-label classifications.

Cost matrices use rows for possible predictions and columns for true labels.
"""

from __future__ import annotations

import torch
from torch import Tensor


def multiclass_expected_decision_cost(
    probabilities: Tensor, *, cost_matrix: Tensor
) -> Tensor:
    """Return expected cost per action from class posterior probabilities.

    Given C[a,y] = loss for action a when truth is y, this returns
    E[C[a,Y] | x] for each candidate action a.
    """
    if probabilities.ndim < 1 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have at least three classes.")
    if not torch.is_floating_point(probabilities):
        raise ValueError("probabilities must be floating point.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite.")
    if torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1].")
    if not torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]),
        atol=1e-6, rtol=1e-6,
    ):
        raise ValueError("Class probabilities must sum to one.")
    num_classes = probabilities.shape[-1]
    if cost_matrix.shape != (num_classes, num_classes):
        raise ValueError("cost_matrix must have shape [num_classes, num_classes].")
    if cost_matrix.dtype != probabilities.dtype or cost_matrix.device != probabilities.device:
        raise ValueError("cost_matrix must match probability dtype and device.")
    if not torch.isfinite(cost_matrix).all() or torch.any(cost_matrix < 0):
        raise ValueError("cost_matrix must contain finite nonnegative costs.")
    return probabilities @ cost_matrix.transpose(-1, -2)


def multiclass_cost_sensitive_prediction(
    probabilities: Tensor, *, cost_matrix: Tensor
) -> Tensor:
    """Select the action with lowest posterior expected misclassification cost."""
    return multiclass_expected_decision_cost(
        probabilities, cost_matrix=cost_matrix
    ).argmin(dim=-1)


def ordinal_cost_matrix(
    num_classes: int,
    *,
    power: float = 1.0,
    reference: Tensor | None = None,
) -> Tensor:
    """Return absolute-distance or squared-distance ordinal costs.

    power=1 yields the posterior weighted-median decision, whereas power=2
    yields the discrete class nearest the posterior mean when ties are absent.
    """
    if not isinstance(num_classes, int) or num_classes < 3:
        raise ValueError("num_classes must be at least 3.")
    if power not in (1.0, 2.0):
        raise ValueError("power must equal 1 or 2.")
    kwargs = {}
    if reference is not None:
        kwargs = {"dtype": reference.dtype, "device": reference.device}
    labels = torch.arange(num_classes, **kwargs)
    distance = (labels[:, None] - labels[None, :]).abs()
    return distance.pow(power)


def ordinal_cost_sensitive_prediction(
    probabilities: Tensor, *, power: float = 1.0
) -> Tensor:
    """Choose the lowest expected ordinal-distance cost action."""
    cost_matrix = ordinal_cost_matrix(
        probabilities.shape[-1], power=power, reference=probabilities
    )
    return multiclass_cost_sensitive_prediction(probabilities, cost_matrix=cost_matrix)
