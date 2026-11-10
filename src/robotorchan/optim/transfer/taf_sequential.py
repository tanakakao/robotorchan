"""Sequential q=1 candidate proposal with a transfer acquisition function."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.optim.transfer.taf import TransferAcquisitionFunction


def propose_taf_candidate(
    source_models: Sequence[Model],
    target_x: Tensor,
    target_y: Tensor,
    bounds: Tensor,
    weights: Tensor,
    *,
    maximize: bool = True,
    raw_samples: int = 128,
    num_restarts: int = 8,
    seed: int = 0,
) -> Tensor:
    """Fit the target GP and propose one candidate using weighted TAF.

    The supplied weights are fixed during this proposal and must have one
    entry per source model plus one for the target model. Evaluate the
    candidate externally, append the observation, and call again for
    sequential BO. Continuous inputs, scalar output, and q=1 only.
    """
    if target_x.ndim != 2 or target_x.shape[0] < 2:
        raise ValueError("target_x must have shape (n, d) with n >= 2")
    n, d = target_x.shape
    if target_y.shape not in ((n,), (n, 1)):
        raise ValueError("target_y must have shape (n,) or (n, 1)")
    if bounds.shape != (2, d) or not bool((bounds[0] < bounds[1]).all()):
        raise ValueError("bounds must have shape (2, d) with increasing limits")
    if not bool(torch.isfinite(target_x).all() and torch.isfinite(target_y).all()):
        raise ValueError("target observations must be finite")
    if not bool(torch.isfinite(bounds).all()):
        raise ValueError("bounds must be finite")
    if not bool(((target_x >= bounds[0]) & (target_x <= bounds[1])).all()):
        raise ValueError("target_x must lie within bounds")
    if raw_samples < 1 or num_restarts < 1:
        raise ValueError("raw_samples and num_restarts must be positive")

    y = target_y.reshape(n, 1)
    target_model = SingleTaskGP(target_x, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(target_model.likelihood, target_model))
    target_model.eval()
    incumbent = y.max().item() if maximize else y.min().item()
    acquisition = TransferAcquisitionFunction(
        source_models,
        target_model,
        weights,
        best_f=incumbent,
        maximize=maximize,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    return candidate.detach()
