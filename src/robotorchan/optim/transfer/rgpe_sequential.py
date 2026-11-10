"""Sequential candidate proposal using a ranking-weighted GP ensemble."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.optim.transfer.loo import target_loo_predictions
from robotorchan.optim.transfer.rgpe import rgpe_weights
from robotorchan.optim.transfer.rgpe_model import RGPEModel


def propose_rgpe_candidate(
    source_models: Sequence[Model],
    target_x: Tensor,
    target_y: Tensor,
    bounds: Tensor,
    *,
    maximize: bool = True,
    n_bootstrap: int = 256,
    raw_samples: int = 128,
    num_restarts: int = 8,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Propose one candidate from fitted source models and target observations.

    Each call fits a target SingleTaskGP and independent LOO target GPs,
    computes RGPE ranking weights, then maximizes analytic EI. The caller
    evaluates the proposed candidate and appends the observation before
    the next call. Only continuous, single-output, q=1 BO is supported.
    Returns (candidate of shape (1, d), model weights).
    """
    if target_x.ndim != 2 or target_x.shape[0] < 3:
        raise ValueError("target_x must have shape (n, d) with n >= 3")
    n, d = target_x.shape
    if target_y.shape not in ((n,), (n, 1)):
        raise ValueError("target_y must have shape (n,) or (n, 1)")
    if bounds.shape != (2, d) or not bool((bounds[0] < bounds[1]).all()):
        raise ValueError("bounds must have shape (2, d) with increasing limits")
    if not bool(torch.isfinite(target_x).all() and torch.isfinite(target_y).all()):
        raise ValueError("target observations must be finite")
    if not bool(((target_x >= bounds[0]) & (target_x <= bounds[1])).all()):
        raise ValueError("target_x must lie within bounds")
    if raw_samples < 1 or num_restarts < 1:
        raise ValueError("raw_samples and num_restarts must be positive")

    y = target_y.reshape(n, 1)
    target_model = SingleTaskGP(target_x, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(target_model.likelihood, target_model))
    target_model.eval()
    loo = target_loo_predictions(target_x, y)
    weights = rgpe_weights(
        source_models,
        target_x,
        y,
        loo,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    ensemble = RGPEModel(source_models, target_model, weights)
    best_f = y.max().item() if maximize else y.min().item()
    acquisition = ExpectedImprovement(ensemble, best_f=best_f, maximize=maximize)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    return candidate.detach(), weights.detach()
