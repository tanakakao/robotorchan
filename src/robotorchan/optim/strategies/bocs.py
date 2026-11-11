"""Sequential binary combinatorial Bayesian optimization using BOCS."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from torch import Tensor

from robotorchan.models.standard.bocs import BOCSSurrogate


def propose_bocs_candidate(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    beta: float = 0.2,
    maximize: bool = True,
    prior_precision: float = 1.0,
    noise_variance: float = 1e-2,
    max_dimensions: int = 16,
) -> Tensor:
    """Propose an unobserved binary point by exhaustive q=1 UCB search.

    The exact enumeration is deliberately limited to modest binary domains.
    Returns a candidate of shape [1, d]; raises when the domain is exhausted.
    """
    if train_X.ndim != 2 or train_X.shape[1] < 1:
        raise ValueError("train_X must have shape [n, d].")
    n, d = train_X.shape
    if d > max_dimensions or max_dimensions < 1 or max_dimensions > 20:
        raise ValueError("Binary dimension exceeds the configured enumeration limit.")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")
    model = BOCSSurrogate(
        train_X,
        train_Y,
        prior_precision=prior_precision,
        noise_variance=noise_variance,
    )
    indices = torch.arange(2**d, device=train_X.device)
    bits = torch.arange(d, device=train_X.device)
    candidates = ((indices[:, None] >> bits) & 1).to(dtype=train_X.dtype)
    unseen = ~(candidates[:, None, :] == train_X[None, :, :]).all(dim=-1).any(dim=-1)
    if not bool(unseen.any()):
        raise ValueError("The binary search space is exhausted.")
    candidates = candidates[unseen]
    acquisition = UpperConfidenceBound(model, beta=beta, maximize=maximize)
    with torch.no_grad():
        values = acquisition(candidates.unsqueeze(-2))
    return candidates[torch.argmax(values)].unsqueeze(0)


def run_bocs_sequential(
    train_X: Tensor,
    train_Y: Tensor,
    objective: Callable[[Tensor], Tensor],
    *,
    steps: int,
    beta: float = 0.2,
    maximize: bool = True,
    prior_precision: float = 1.0,
    noise_variance: float = 1e-2,
    max_dimensions: int = 16,
) -> tuple[Tensor, Tensor]:
    """Refit BOCS and evaluate one new candidate per sequential iteration.

    The objective accepts a single candidate [1, d] and returns [1, 1].
    All observations, including the initial data, are returned in order.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative.")
    X, Y = train_X.detach().clone(), train_Y.detach().clone()
    for _ in range(steps):
        candidate = propose_bocs_candidate(
            X,
            Y,
            beta=beta,
            maximize=maximize,
            prior_precision=prior_precision,
            noise_variance=noise_variance,
            max_dimensions=max_dimensions,
        )
        value = objective(candidate)
        if value.shape != (1, 1):
            raise ValueError("objective must return a tensor of shape [1, 1].")
        if value.dtype != Y.dtype or value.device != Y.device:
            raise ValueError("objective output must match train_Y dtype and device.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("objective must return a finite observation.")
        X = torch.cat((X, candidate), dim=0)
        Y = torch.cat((Y, value.detach()), dim=0)
    return X, Y
