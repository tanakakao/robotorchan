"""Finite empirical context distributions and policy-value evaluation.

This module evaluates supplied conditional outcomes, not a surrogate posterior.
It deliberately does not optimize an acquisition function or retrain a model.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_distribution(weights: Tensor) -> Tensor:
    if not isinstance(weights, Tensor) or weights.ndim != 1 or not weights.numel():
        raise ValueError("weights must be a nonempty one-dimensional tensor.")
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise ValueError("weights must contain finite floating-point values.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive total mass.")
    return weights / weights.sum()


def expected_policy_value(outcomes: Tensor, weights: Tensor) -> Tensor:
    """Return expected objective values for one or more policies.

    outcomes: [n_contexts] for one policy or [n_policies, n_contexts].
    weights: nonnegative context masses [n_contexts], normalized internally.
    The returned value is scalar or [n_policies], respectively.
    """
    probabilities = _validate_distribution(weights)
    if not isinstance(outcomes, Tensor) or outcomes.ndim not in (1, 2):
        raise ValueError("outcomes must have shape [n_contexts] or [n_policies, n_contexts].")
    if outcomes.shape[-1] != probabilities.numel():
        raise ValueError("outcomes and weights must share the context dimension.")
    if not outcomes.is_floating_point() or not torch.isfinite(outcomes).all():
        raise ValueError("outcomes must contain finite floating-point values.")
    if outcomes.dtype != probabilities.dtype or outcomes.device != probabilities.device:
        raise ValueError("outcomes and weights must share dtype and device.")
    return (outcomes * probabilities).sum(dim=-1)


def monte_carlo_policy_value(
    outcomes: Tensor,
    weights: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Estimate policy values by sampling contexts from an empirical distribution.

    Shared context indices across policies enable paired comparisons.
    """
    probabilities = _validate_distribution(weights)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(outcomes, Tensor) or outcomes.ndim not in (1, 2):
        raise ValueError("outcomes must have shape [n_contexts] or [n_policies, n_contexts].")
    if outcomes.shape[-1] != probabilities.numel():
        raise ValueError("outcomes and weights must share the context dimension.")
    if not outcomes.is_floating_point() or not torch.isfinite(outcomes).all():
        raise ValueError("outcomes must contain finite floating-point values.")
    if outcomes.dtype != probabilities.dtype or outcomes.device != probabilities.device:
        raise ValueError("outcomes and weights must share dtype and device.")
    indices = torch.multinomial(
        probabilities, num_samples=n_samples, replacement=True, generator=generator
    )
    return outcomes.index_select(-1, indices).mean(dim=-1)
