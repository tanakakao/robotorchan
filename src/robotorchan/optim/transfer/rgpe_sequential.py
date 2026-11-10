"""Sequential candidate proposal using a ranking-weighted GP ensemble."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
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
    q: int = 1,
    batch_strategy: str = "sequential",
    mc_samples: int = 128,
    cat_dims: list[int] | None = None,
    fixed_features_list: list[dict[int, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Propose one candidate from fitted source models and target observations.

    Each call fits a target SingleTaskGP and independent LOO target GPs,
    computes RGPE ranking weights, then maximizes analytic EI. The caller
    evaluates the proposed candidate and appends the observation before
    the next call. Mixed inputs require cat_dims and fixed_features_list.
    For sequential q>1, greedily condition on posterior-mean fantasies after each
    proposal, without updating RGPE ranking weights. Single-output only.
    Returns (candidate of shape (q, d), model weights).
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
    if batch_strategy not in ("sequential", "joint"):
        raise ValueError("batch_strategy must be 'sequential' or 'joint'")
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
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
    loo = (
        target_loo_predictions(target_x, y, cat_dims=cat_dims)
        if cat_dims is not None
        else target_loo_predictions(target_x, y)
    )
    weights = rgpe_weights(
        source_models,
        target_x,
        y,
        loo,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    best_f = y.max().item() if maximize else y.min().item()
    if batch_strategy == "joint" and q > 1:
        ensemble = RGPEModel(source_models, target_model, weights)
        objective = (
            None if maximize else GenericMCObjective(lambda samples, X=None: -samples.squeeze(-1))
        )
        acquisition = qExpectedImprovement(
            ensemble,
            best_f=best_f if maximize else -best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples])),
            objective=objective,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if fixed_features_list is None:
                candidate, _ = optimize_acqf(
                    acquisition,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
            else:
                candidate, _ = optimize_acqf_mixed(
                    acquisition,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features_list=fixed_features_list,
                )
        return candidate.detach(), weights.detach()

    candidates = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(q):
            ensemble = RGPEModel(source_models, target_model, weights)
            acquisition = ExpectedImprovement(ensemble, best_f=best_f, maximize=maximize)
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
            target_model = target_model.condition_on_observations(X=candidate, Y=fantasy)
            fantasy_value = fantasy.item()
            best_f = max(best_f, fantasy_value) if maximize else min(best_f, fantasy_value)
    return torch.cat(candidates, dim=0), weights.detach()
