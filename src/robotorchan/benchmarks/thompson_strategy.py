"""Posterior Thompson sampling benchmark with independently seeded candidate pools."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem


def thompson_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    q: int,
    generator: torch.Generator,
    *,
    pool_size: int = 256,
) -> Tensor:
    """Draw a joint posterior sample and choose its top q distinct pool locations."""
    if problem.n_objectives != 1 or problem.n_constraints:
        raise ValueError("Thompson benchmark requires an unconstrained scalar objective.")
    if any(kind != "continuous" for kind in problem.variable_types):
        raise ValueError("Thompson benchmark requires continuous variables.")
    if type(pool_size) is not int or pool_size < 2:
        raise ValueError("pool_size must be an integer >= 2.")
    if type(q) is not int or not 1 <= q <= pool_size:
        raise ValueError("q must be between 1 and pool_size.")
    if X.ndim != 2 or Y.shape != (X.shape[0], 1):
        raise ValueError("Expected X=(n,d) and Y=(n,1).")
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    devices = [X.device.index or 0] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        pool = torch.quasirandom.SobolEngine(
            dimension=X.shape[-1], scramble=True, seed=seed
        ).draw(pool_size).to(dtype=X.dtype, device=X.device)
        pool = bounds[0] + (bounds[1] - bounds[0]) * pool
        direction = 1 if problem.directions[0] == "maximize" else -1
        model = SingleTaskGP(X, Y * direction)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        with torch.no_grad():
            sample = model.posterior(pool).rsample().reshape(pool_size)
        if not torch.isfinite(sample).all():
            raise ValueError("Nonfinite Thompson posterior sample.")
        indices = torch.topk(sample, k=q).indices
    return pool[indices].detach()
