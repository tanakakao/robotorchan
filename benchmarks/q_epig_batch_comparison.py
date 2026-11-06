"""Compare marginal, joint, and pending-aware Gaussian qEPIG batch selection."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from robotorchan.acquisition import (
    qConditionalExpectedPredictiveInformationGain,
    qExpectedPredictiveInformationGain,
    qJointExpectedPredictiveInformationGain,
)


def run_benchmark(seed: int, q: int) -> list[dict[str, float | int | str]]:
    """Optimize equal-budget batches and report measured acquisition diagnostics."""
    if q < 1:
        raise ValueError("q must be positive.")
    torch.manual_seed(seed)
    train_X = torch.linspace(0.0, 1.0, 9, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(6.0 * train_X)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    target_X = torch.linspace(0.0, 1.0, 31, dtype=torch.double).unsqueeze(-1)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    pending = torch.tensor([[0.35]], dtype=torch.double)
    noise = 1e-3
    acquisitions = {
        "marginal": qExpectedPredictiveInformationGain(model, target_X, observation_noise=noise),
        "joint": qJointExpectedPredictiveInformationGain(model, target_X, observation_noise=noise),
        "conditional": qConditionalExpectedPredictiveInformationGain(
            model, target_X, X_pending=pending, observation_noise=noise
        ),
    }
    joint = acquisitions["joint"]
    conditional = acquisitions["conditional"]
    results = []
    for name, acquisition in acquisitions.items():
        torch.manual_seed(seed)
        start = perf_counter()
        candidates, value = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=q,
            num_restarts=4,
            raw_samples=32,
            options={"maxiter": 50},
        )
        elapsed = perf_counter() - start
        with torch.no_grad():
            batch = candidates.unsqueeze(0)
            distances = torch.pdist(candidates)
            results.append(
                {
                    "strategy": name,
                    "seed": seed,
                    "q": q,
                    "acquisition_value": float(value.item()),
                    "joint_epig": float(joint(batch).item()),
                    "incremental_epig": float(conditional(batch).item()),
                    "min_pairwise_distance": (
                        float(distances.min().item()) if distances.numel() else 0.0
                    ),
                    "optimization_seconds": elapsed,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q", type=int, default=2)
    args = parser.parse_args()
    for result in run_benchmark(seed=args.seed, q=args.q):
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
