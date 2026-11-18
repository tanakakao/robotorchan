"""Reproducible synthetic contextual-policy benchmark (not a BO convergence test).

Run: python benchmarks/contextual_policy.py
The context-aware oracle is an analytical upper bound, not a learned BO policy.
"""

from __future__ import annotations

import time

import torch

from robotorchan.optim.contextual_policy import (
    expected_policy_value,
    monte_carlo_policy_value,
)


def reward(decisions: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
    """Smooth contextual objective, maximized at decision == context."""
    return 1.0 - (decisions - contexts).square()


def run(seed: int = 42, n_samples: int = 4096) -> dict[str, float]:
    """Compare a fixed decision with the context-aware analytical oracle."""
    contexts = torch.linspace(0.0, 1.0, 21, dtype=torch.double)
    weights = torch.ones_like(contexts)
    fixed = torch.full_like(contexts, 0.5)
    oracle = contexts.clone()
    outcomes = torch.stack((reward(fixed, contexts), reward(oracle, contexts)))

    started = time.perf_counter()
    exact = expected_policy_value(outcomes, weights)
    sampled = monte_carlo_policy_value(
        outcomes,
        weights,
        n_samples=n_samples,
        generator=torch.Generator().manual_seed(seed),
    )
    elapsed = time.perf_counter() - started
    return {
        "fixed_expected": exact[0].item(),
        "oracle_expected": exact[1].item(),
        "oracle_gain": (exact[1] - exact[0]).item(),
        "fixed_mc": sampled[0].item(),
        "oracle_mc": sampled[1].item(),
        "evaluation_seconds": elapsed,
    }


if __name__ == "__main__":
    for name, value in run().items():
        print(f"{name}: {value:.6f}")
