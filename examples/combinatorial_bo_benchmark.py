"""Seed-aligned binary combinatorial BO benchmark (Phase 9).

Run: python examples/combinatorial_bo_benchmark.py --seeds 3 --steps 5
All methods start from identical observations and use the same oracle budget.
The local/TR methods use COMBO's acquisition, so they compare candidate
search strategies rather than independent surrogate families.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.models.standard import COMBOSurrogate
from robotorchan.optim import (
    DiscreteTrustRegionState,
    DiscreteTrustRegionStrategy,
    HammingLocalSearchStrategy,
    MixedVariableSpace,
    propose_bocs_candidate,
    propose_combo_candidate,
)

METHODS = ("random", "sobol", "mixed_gp", "hamming", "discrete_tr", "bocs", "combo")


def oracle(X: Tensor) -> Tensor:
    """Deceptive binary landscape with a unique all-ones global optimum."""
    count = X.sum(dim=-1, keepdim=True)
    return torch.where(count == X.shape[-1], count + 2, X.shape[-1] - count)


def binary_domain(d: int) -> Tensor:
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=d)), dtype=torch.double)


def propose(
    method: str, X: Tensor, Y: Tensor, domain: Tensor, seed: int, iteration: int
) -> Tensor:
    unseen = ~(domain[:, None, :] == X[None, :, :]).all(dim=-1).any(dim=-1)
    available = domain[unseen]
    if available.shape[0] == 0:
        raise ValueError("Search space exhausted.")
    if method == "bocs":
        return propose_bocs_candidate(X, Y)
    if method == "combo":
        return propose_combo_candidate(X, Y)
    if method == "random":
        generator = torch.Generator().manual_seed(seed * 1000 + iteration)
        return available[torch.randint(len(available), (1,), generator=generator)]
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(X.shape[-1], scramble=True, seed=seed)
        draws = engine.draw(iteration + 1).to(dtype=X.dtype)
        target = (draws[-1] >= 0.5).to(dtype=X.dtype)
        distances = (available != target).sum(dim=-1)
        return available[torch.argmin(distances)].unsqueeze(0)

    if method == "mixed_gp":
        model = SingleTaskGP(X, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    else:
        model = COMBOSurrogate(X, Y)
        model.fit()
    model.eval()
    acquisition = UpperConfidenceBound(model, beta=0.2)
    if method == "mixed_gp":
        with torch.no_grad():
            values = acquisition(available.unsqueeze(-2))
        return available[torch.argmax(values)].unsqueeze(0)

    bounds = torch.stack((torch.zeros(X.shape[-1]), torch.ones(X.shape[-1]))).double()
    space = MixedVariableSpace(bounds=bounds, integer_dims=tuple(range(X.shape[-1])))
    excluded = tuple(tuple(row.tolist()) for row in X)
    if method == "hamming":
        strategy = HammingLocalSearchStrategy(space, available, excluded=excluded, max_steps=10)
    elif method == "discrete_tr":
        center = X[torch.argmax(Y.flatten())]
        radius = min(2, X.shape[-1])
        starts = available[(available != center).sum(dim=-1) <= radius]
        strategy = DiscreteTrustRegionStrategy(
            space,
            center=center,
            state=DiscreteTrustRegionState(radius=radius, max_radius=radius),
            initial_points=starts,
            excluded=excluded,
            max_steps=10,
        )
    else:
        raise ValueError(f"Unknown benchmark method: {method}")
    return strategy.optimize(acquisition).candidates


def run_benchmark(
    *, seeds: int = 3, steps: int = 5, dimensions: int = 4, initial: int = 3
) -> list[dict[str, float | int | str]]:
    """Return one row per seed, method and evaluation count."""
    if seeds < 1 or steps < 1 or dimensions < 2 or dimensions > 8:
        raise ValueError("Require seeds/steps >= 1 and 2 <= dimensions <= 8.")
    domain = binary_domain(dimensions)
    if initial < 2 or initial + steps > len(domain):
        raise ValueError("Require at least two initial points and a feasible budget.")
    rows: list[dict[str, float | int | str]] = []
    for seed in range(seeds):
        generator = torch.Generator().manual_seed(seed)
        initial_X = domain[torch.randperm(len(domain), generator=generator)[:initial]]
        initial_Y = oracle(initial_X)
        for method in METHODS:
            X, Y = initial_X.clone(), initial_Y.clone()
            start = time.perf_counter()
            for iteration in range(steps):
                candidate = propose(method, X, Y, domain, seed, iteration)
                if bool((candidate == X).all(dim=-1).any()):
                    raise RuntimeError(f"{method} proposed an already observed candidate.")
                X = torch.cat((X, candidate), dim=0)
                Y = torch.cat((Y, oracle(candidate)), dim=0)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "evaluations": initial + iteration + 1,
                        "best_value": float(Y.max()),
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--initial", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("combinatorial_bo_results.csv"))
    args = parser.parse_args()
    rows = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        dimensions=args.dimensions,
        initial=args.initial,
    )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in METHODS:
        final = [
            row["best_value"]
            for row in rows
            if row["method"] == method and row["evaluations"] == args.initial + args.steps
        ]
        print(f"{method}: mean best={sum(final) / len(final):.3f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
