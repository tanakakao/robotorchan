"""Decision utilities for imbalanced and cost-sensitive classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def binary_cost_sensitive_threshold(
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Return the Bayes-optimal positive-class threshold for binary costs."""
    for name, value in (
        ("false_positive_cost", false_positive_cost),
        ("false_negative_cost", false_negative_cost),
    ):
        if not isinstance(value, int | float):
            raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    total = false_positive_cost + false_negative_cost
    if total <= 0.0:
        raise ValueError("At least one misclassification cost must be positive.")
    return false_positive_cost / total


def binary_expected_decision_cost(
    probabilities: Tensor,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> Tensor:
    """Return expected cost for decisions 0 and 1 along the final dimension."""
    threshold = binary_cost_sensitive_threshold(
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    del threshold
    if probabilities.ndim < 1 or probabilities.shape[-1] != 2:
        raise ValueError("probabilities must have final dimension 2.")
    if not torch.is_floating_point(probabilities):
        raise ValueError("probabilities must use a floating-point dtype.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite.")
    if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("Class probabilities must sum to one.")
    p_negative = probabilities[..., 0]
    p_positive = probabilities[..., 1]
    cost_predict_negative = false_negative_cost * p_positive
    cost_predict_positive = false_positive_cost * p_negative
    return torch.stack((cost_predict_negative, cost_predict_positive), dim=-1)


def binary_cost_sensitive_prediction(
    probabilities: Tensor,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> Tensor:
    """Return the minimum-expected-cost binary decision."""
    expected_cost = binary_expected_decision_cost(
        probabilities,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    return expected_cost.argmin(dim=-1)


def inverse_frequency_class_weights(targets: Tensor) -> Tensor:
    """Return normalized inverse-frequency weights for binary labels."""
    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional.")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must contain integer binary labels.")
    if targets.numel() == 0:
        raise ValueError("targets must not be empty.")
    if torch.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must contain only binary labels 0 and 1.")
    counts = torch.bincount(targets.to(torch.long), minlength=2)
    if torch.any(counts == 0):
        raise ValueError("Both binary classes must be present.")
    weights = targets.new_tensor(targets.numel(), dtype=torch.get_default_dtype()) / (
        2.0 * counts.to(torch.get_default_dtype())
    )
    return weights / weights.mean()
