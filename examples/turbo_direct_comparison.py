"""Matched-budget TuRBOStrategy (TuRBO-1) versus TuRBO-m comparison.

Run manually: python examples/turbo_direct_comparison.py
This benchmark is intentionally not part of the default Notebook CI.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import Ackley, Levy, Rosenbrock
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from robotorchan.optim.trust_region import TuRBOMultiRegionState, TuRBOState, TuRBOStrategy


def _evaluate(problem, X: torch.Tensor) -> torch.Tensor:
    return -problem(X).reshape(-1, 1)


def _fit(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def _initial(bounds: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    unit = SobolEngine(bounds.shape[-1], scramble=True, seed=seed).draw(n).to(bounds)
    return bounds[0] + unit * (bounds[1] - bounds[0])


def compare(
    problem,
    *,
    seed: int,
    n_regions: int = 3,
    initial_budget: int = 15,
    q: int = 2,
    steps: int = 8,
) -> dict[str, list[float]]:
    """Return incumbent traces after equal numbers of objective evaluations."""
    if n_regions < 2 or initial_budget < 2 * n_regions or q < 1 or steps < 0:
        raise ValueError("invalid comparison configuration")
    bounds = problem.bounds.to(dtype=torch.double)
    X0 = _initial(bounds, initial_budget, seed)
    Y0 = _evaluate(problem, X0)
    best0 = float(Y0.max().item())

    # The actual TuRBO-1 public strategy, not a one-region TuRBO-m surrogate.
    best_idx = int(Y0.argmax().item())
    state = replace(
        TuRBOState(dim=bounds.shape[-1], batch_size=q),
        best_value=best0,
        observed_best_value=best0,
    )
    single = TuRBOStrategy(bounds, center=X0[best_idx], state=state, seed=seed)
    single_X, single_Y = X0.clone(), Y0.clone()
    single_trace = [best0]

    # Same initial objective evaluations, partitioned between independent regions.
    indices = [torch.arange(i, initial_budget, n_regions) for i in range(n_regions)]
    centers = torch.stack([X0[ix[int(Y0[ix].argmax().item())]] for ix in indices])
    multi = TuRBOMultiRegionState(bounds, centers, batch_size=q)
    for region_id, ix in enumerate(indices):
        multi.append_observations(region_id, X0[ix], Y0[ix])
        region = multi.region(region_id)
        regional_best = float(Y0[ix].max().item())
        region.state = replace(
            region.state, best_value=regional_best, observed_best_value=regional_best
        )
    multi_trace = [best0]

    for step in range(steps):
        if single.state.restart_triggered:
            single.restart(seed=seed + 1000 + step)
        model = _fit(single_X, single_Y)
        single.seed = seed + 2000 + step
        X = single.thompson_sample(model, q=q, n_candidates=128).candidates
        Y = _evaluate(problem, X)
        single.update_state(Y, candidates=X)
        single_X = torch.cat((single_X, X))
        single_Y = torch.cat((single_Y, Y))
        single_trace.append(float(single_Y.max().item()))

        for region_id, region in enumerate(multi.regions):
            if region.state.restart_triggered:
                multi.restart_region(region_id, seed=seed + 3000 + step * n_regions + region_id)
                X_init = _initial(
                    bounds, len(indices[region_id]), seed + 4000 + step * n_regions + region_id
                )
                Y_init = _evaluate(problem, X_init)
                multi.append_observations(region_id, X_init, Y_init)
                idx = int(Y_init.argmax().item())
                region.center = X_init[idx].clone()
                best = float(Y_init.max().item())
                region.state = replace(
                    region.state, best_value=best, observed_best_value=best
                )
            multi.set_model(region_id, _fit(region.train_X, region.train_Y))
        X, ids, _ = multi.propose_batch(q=q, n_candidates=128, seed=seed + 5000 + step)
        Y = _evaluate(problem, X)
        multi.update_regions(X, Y, ids)
        multi_trace.append(max(float(r.history_Y.max().item()) for r in multi.regions))

    return {"TuRBO-1": single_trace, "TuRBO-m": multi_trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--regions", type=int, default=3)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    torch.set_default_dtype(torch.double)
    for problem in (Ackley(dim=2), Rosenbrock(dim=2), Levy(dim=2)):
        results = [
            compare(problem, seed=seed, n_regions=args.regions, steps=args.steps)
            for seed in range(args.seeds)
        ]
        print(type(problem).__name__)
        for method in ("TuRBO-1", "TuRBO-m"):
            finals = torch.tensor([run[method][-1] for run in results])
            print(
                f"  {method}: final best mean={finals.mean().item():.6f}, "
                f"std={finals.std(unbiased=False).item():.6f}"
            )
    print("Higher is better (negative objective); descriptive statistics only.")


if __name__ == "__main__":
    main()
