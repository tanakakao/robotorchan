"""Posterior confidence-bound screening for continuous safety outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from robotorchan.semantics.constraints import ContinuousConstraint
from robotorchan.semantics.direction import ConstraintDirection


def safety_confidence_bounds(
    model: object,
    X: Tensor,
    constraints: Sequence[ContinuousConstraint],
    *,
    beta: float = 4.0,
    tolerance: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return safety margins and a joint-safe mask for posterior candidates.

    Margins >= -tolerance are accepted. For lower constraints the margin is
    mu - sqrt(beta)*sigma - threshold; for upper constraints it is
    threshold - mu - sqrt(beta)*sigma. The last axis indexes constraints.
    This is a posterior screening rule, not a formal safety guarantee.
    """
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise TypeError("beta must be a real scalar.")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite.")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be a real scalar.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if not constraints or not all(isinstance(c, ContinuousConstraint) for c in constraints):
        raise ValueError("constraints must contain ContinuousConstraint instances.")
    if X.ndim < 2:
        raise ValueError("X must have shape [..., q, d].")

    posterior = model.posterior(X)
    mean, variance = posterior.mean, posterior.variance
    if mean.shape != variance.shape or mean.ndim < 2:
        raise ValueError("posterior mean and variance must have matching [..., q, m] shape.")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
        raise ValueError("posterior moments must be finite.")
    if bool((variance < 0).any()):
        raise ValueError("posterior variance must be non-negative.")

    margins = []
    for constraint in constraints:
        if isinstance(constraint.output, str):
            if not hasattr(model, "resolve_output"):
                raise TypeError("Named safety outputs require model.resolve_output.")
            index = model.resolve_output(constraint.output)
        else:
            index = constraint.output
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Safety output index must be an integer.")
        if index < 0 or index >= mean.shape[-1]:
            raise IndexError("Safety output index is outside posterior outputs.")
        mu = mean[..., index]
        uncertainty = math.sqrt(beta) * variance[..., index].sqrt()
        if constraint.direction is ConstraintDirection.GREATER_THAN_OR_EQUAL:
            margin = mu - uncertainty - constraint.threshold
        else:
            margin = constraint.threshold - mu - uncertainty
        margins.append(margin)
    stacked = torch.stack(margins, dim=-1)
    return stacked, (stacked >= -tolerance).all(dim=-1)
