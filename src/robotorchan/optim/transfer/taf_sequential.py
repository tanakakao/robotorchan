"""Sequential q=1 candidate proposal with a transfer acquisition function."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf, optimize_acqf_mixed
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
    q: int = 1,
    cat_dims: list[int] | None = None,
    fixed_features_list: list[dict[int, float]] | None = None,
) -> Tensor:
    """Fit the target GP and propose one candidate using weighted TAF.

    The supplied weights are fixed during this proposal and must have one
    entry per source model plus one for the target model. Evaluate the
    candidate externally, append the observation, and call again for
    sequential BO. Mixed inputs require cat_dims and fixed_features_list.
    For q>1, greedily condition on posterior-mean fantasies after each
    proposal, keeping source weights fixed. Scalar output only.
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
    if q < 1:
        raise ValueError("q must be positive")
    if raw_samples < 1 or num_restarts < 1:
        raise ValueError("raw_samples and num_restarts must be positive")

    if (cat_dims is None) != (fixed_features_list is None):
        raise ValueError("cat_dims and fixed_features_list must be provided together")
    if cat_dims is not None:
        if not cat_dims or len(set(cat_dims)) != len(cat_dims):
            raise ValueError("cat_dims must contain unique categorical dimensions")
        if any(index < 0 or index >= d for index in cat_dims):
            raise ValueError("cat_dims must be valid input dimensions")
        if not fixed_features_list or any(
            set(features) != set(cat_dims) for features in fixed_features_list
        ):
            raise ValueError("fixed_features_list must specify every categorical dimension")
        if any(
            not any(
                all(bool(target_x[row, index] == features[index]) for index in cat_dims)
                for features in fixed_features_list
            )
            for row in range(n)
        ):
            raise ValueError("target_x contains unsupported categorical assignments")

    y = target_y.reshape(n, 1)
    target_model = (
        MixedSingleTaskGP(target_x, y, cat_dims=cat_dims)
        if cat_dims is not None
        else SingleTaskGP(target_x, y)
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(target_model.likelihood, target_model))
    target_model.eval()
    incumbent = y.max().item() if maximize else y.min().item()
    candidates = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(q):
            acquisition = TransferAcquisitionFunction(
                source_models,
                target_model,
                weights,
                best_f=incumbent,
                maximize=maximize,
            )
            if fixed_features_list is None:
                candidate, _ = optimize_acqf(
                    acquisition,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
            else:
                candidate, _ = optimize_acqf_mixed(
                    acquisition,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features_list=fixed_features_list,
                )
            candidates.append(candidate.detach())
            if len(candidates) == q:
                break
            with torch.no_grad():
                fantasy = target_model.posterior(candidate).mean.detach()
            target_model = target_model.condition_on_observations(
                X=candidate, Y=fantasy
            )
            fantasy_value = fantasy.item()
            incumbent = max(incumbent, fantasy_value) if maximize else min(
                incumbent, fantasy_value
            )
    return torch.cat(candidates, dim=0)
