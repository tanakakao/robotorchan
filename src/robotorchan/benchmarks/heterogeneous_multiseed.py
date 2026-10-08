"""Multi-seed summaries of fixed-budget synthetic optimization trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from robotorchan.benchmarks.heterogeneous_baselines import run_strategy


def summarize_trajectories(curves: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return mean, sample standard deviation, and standard error over seeds."""
    if curves.ndim != 2 or curves.shape[0] < 2 or curves.shape[1] < 1:
        raise ValueError("curves must have shape (at least 2 seeds, at least 1 step).")
    if not curves.is_floating_point() or not torch.isfinite(curves).all():
        raise ValueError("curves must be finite floating-point values.")
    mean = curves.mean(dim=0)
    std = curves.std(dim=0, unbiased=True)
    sem = std / curves.shape[0] ** 0.5
    return mean, std, sem


def compare_strategies(
    seeds: Sequence[int],
    *,
    initial_points: int = 12,
    steps: int = 3,
) -> dict[str, Tensor]:
    """Collect best-so-far trajectories with paired initial designs per seed."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required.")
    if initial_points < 2 or steps < 1:
        raise ValueError("initial_points must be >= 2 and steps must be >= 1.")
    curves: dict[str, list[Tensor]] = {
        "random": [],
        "sobol": [],
        "qei": [],
    }
    for seed in seeds:
        initial_X = torch.rand(
            initial_points,
            3,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.double,
        )
        for strategy in curves:
            _, best = run_strategy(strategy, initial_X, seed=seed, steps=steps)
            curves[strategy].append(best)
    return {strategy: torch.stack(values) for strategy, values in curves.items()}
