"""Small reproducible transfer BO comparison including negative transfer.

Run with: python examples/benchmarks/transfer_bo.py
This is an illustrative benchmark, not a statistical performance claim.
"""

from __future__ import annotations

import argparse

import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer.rgpe_sequential import propose_rgpe_candidate
from robotorchan.optim.transfer.taf_sequential import propose_taf_candidate


def objective(x: torch.Tensor, optimum: float) -> torch.Tensor:
    """One-dimensional maximization objective."""
    return -((x - optimum) ** 2)


def run(
    method: str,
    *,
    aligned: bool,
    steps: int,
    seed: int,
) -> list[float]:
    """Return best observed objective after initialization and each proposal."""
    dtype = torch.double
    bounds = torch.tensor([[0.0], [1.0]], dtype=dtype)
    x = torch.tensor([[0.0], [0.3], [1.0]], dtype=dtype)
    y = objective(x, 0.75)
    source_optimum = 0.75 if aligned else 0.05
    source_x = torch.linspace(0, 1, 6, dtype=dtype).unsqueeze(-1)
    source = SingleTaskGP(source_x, objective(source_x, source_optimum))
    source_models = [source]
    history = [y.max().item()]

    for iteration in range(steps):
        if method == "rgpe":
            candidate, _ = propose_rgpe_candidate(
                source_models,
                x,
                y,
                bounds,
                n_bootstrap=16,
                raw_samples=32,
                num_restarts=2,
                seed=seed + iteration,
            )
        elif method in ("target", "taf"):
            weights = (
                torch.tensor([0.0, 1.0], dtype=dtype)
                if method == "target"
                else torch.tensor([0.5, 0.5], dtype=dtype)
            )
            candidate = propose_taf_candidate(
                source_models,
                x,
                y,
                bounds,
                weights,
                raw_samples=32,
                num_restarts=2,
                seed=seed + iteration,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        x = torch.cat([x, candidate], dim=0)
        y = torch.cat([y, objective(candidate, 0.75)], dim=0)
        history.append(y.max().item())
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    print("scenario,method,iteration,best_observed")
    for aligned, scenario in ((True, "aligned"), (False, "misaligned")):
        for method in ("target", "taf", "rgpe"):
            for iteration, value in enumerate(
                run(method, aligned=aligned, steps=args.steps, seed=args.seed)
            ):
                print(f"{scenario},{method},{iteration},{value:.8f}")


if __name__ == "__main__":
    main()
