"""Reproducible single-region versus TuRBO-m benchmark.

Run: python examples/turbo_multi_benchmark.py
Both configurations use the same regional Thompson sampler for a controlled
one-versus-multiple trust-region comparison, not a TuRBOStrategy API benchmark.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import Ackley, Levy, Rosenbrock
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from robotorchan.optim.trust_region import TuRBOMultiRegionState


def _evaluate(function, X: torch.Tensor) -> torch.Tensor:
    return -function(X).reshape(-1, 1)


def _initial_points(bounds: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    unit = SobolEngine(bounds.shape[-1], scramble=True, seed=seed).draw(n).to(bounds)
    return bounds[0] + unit * (bounds[1] - bounds[0])


def _fit_local_models(manager: TuRBOMultiRegionState) -> None:
    for index, region in enumerate(manager.regions):
        if region.train_X.shape[0] < 2:
            raise ValueError("each region requires at least two initial observations.")
        model = SingleTaskGP(region.train_X, region.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        manager.set_model(index, model)


def run_experiment(
    function,
    *,
    n_regions: int,
    q: int = 2,
    n_steps: int = 8,
    initial_per_region: int = 5,
    seed: int = 0,
) -> list[float]:
    """Run maximization of negative benchmark values; return best objective history."""
    if n_regions < 1 or q < 1 or n_steps < 0 or initial_per_region < 2:
        raise ValueError("invalid benchmark configuration.")
    bounds = function.bounds.to(dtype=torch.double)
    centers = _initial_points(bounds, n_regions, seed)
    manager = TuRBOMultiRegionState(bounds, centers, batch_size=q)
    for index in range(n_regions):
        X = _initial_points(bounds, initial_per_region, seed + 100 + index)
        Y = _evaluate(function, X)
        manager.append_observations(index, X, Y)
        region = manager.region(index)
        best_index = int(Y.argmax().item())
        region.center = X[best_index].clone()
        region.state = replace(
            region.state,
            best_value=float(Y.max().item()),
            observed_best_value=float(Y.max().item()),
        )
    history = [max(float(r.train_Y.max()) for r in manager.regions)]
    for step in range(n_steps):
        for index, region in enumerate(manager.regions):
            if region.state.restart_triggered:
                manager.restart_region(index, seed=seed + 1000 + step * n_regions + index)
                X = _initial_points(
                    bounds, initial_per_region, seed + 2000 + step * n_regions + index
                )
                manager.append_observations(index, X, _evaluate(function, X))
                region.center = X[int(region.train_Y.argmax().item())].clone()
                best = float(region.train_Y.max().item())
                region.state = replace(region.state, best_value=best, observed_best_value=best)
        _fit_local_models(manager)
        X, region_ids, _ = manager.propose_batch(
            q=q, n_candidates=128, seed=seed + 3000 + step
        )
        manager.update_regions(X, _evaluate(function, X), region_ids)
        history.append(max(float(r.history_Y.max()) for r in manager.regions))
    return history


def main() -> None:
    torch.set_default_dtype(torch.double)
    problems = (Ackley(dim=2), Rosenbrock(dim=2), Levy(dim=2))
    for problem in problems:
        print(type(problem).__name__)
        for n_regions in (1, 3):
            history = run_experiment(
                problem, n_regions=n_regions, initial_per_region=15 // n_regions
            )
            print(f"  regions={n_regions}: initial={history[0]:.4f}, final={history[-1]:.4f}")
    print("Scores are negative function values; larger is better.")


if __name__ == "__main__":
    main()
