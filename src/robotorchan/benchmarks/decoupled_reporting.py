"""Cost-aligned reporting for independent-output benchmark trajectories."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import isfinite

import torch
from torch import Tensor

from robotorchan.benchmarks.multiseed_statistics import (
    bootstrap_curves,
    compare_paired_bootstrap,
)


def align_cost_curves(
    records: Iterable[dict], metric: str, cost_grid: Sequence[float]
) -> dict[str, dict[int, Tensor]]:
    """Carry the latest available metric forward without using future observations.

    Input must describe one problem and one experimental configuration.
    Missing metrics are rejected rather than silently dropping failed recommendations.
    """
    grid = torch.as_tensor(cost_grid, dtype=torch.double)
    if (
        grid.ndim != 1
        or not grid.numel()
        or not torch.isfinite(grid).all()
        or (grid < 0).any()
        or (grid.diff() <= 0).any()
    ):
        raise ValueError("Cost grid must be finite, nonnegative and strictly increasing.")
    groups = defaultdict(list)
    for row in records:
        value = row.get(metric)
        if value is None or not isfinite(float(value)):
            raise ValueError(f"Missing or nonfinite metric: {metric}")
        cost = float(row["cost"])
        if not isfinite(cost) or cost < 0:
            raise ValueError("Invalid cumulative cost.")
        groups[row["method"], int(row["seed"])].append((cost, float(value)))
    if not groups:
        raise ValueError("No benchmark records.")
    result = defaultdict(dict)
    for (method, seed), rows in groups.items():
        rows.sort()
        costs, values = torch.tensor(rows, dtype=torch.double).T
        if (costs.diff() <= 0).any():
            raise ValueError("Duplicate cost within method and seed.")
        if grid[0] < costs[0]:
            raise ValueError("Cost grid precedes initial observations.")
        indices = torch.searchsorted(costs.contiguous(), grid, right=True) - 1
        result[method][seed] = values[indices]
    return dict(result)


def summarize_cost_curves(
    records: Iterable[dict],
    metric: str,
    cost_grid: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    baseline: str | None = None,
) -> list[dict]:
    """Return JSON-compatible pointwise summaries and optional paired differences."""
    curves = align_cost_curves(records, metric, cost_grid)
    if baseline is not None and baseline not in curves:
        raise ValueError("Unknown baseline.")
    output = []
    for method, seeded in sorted(curves.items()):
        summary = bootstrap_curves(seeded, confidence=confidence, n_resamples=n_resamples)
        paired = None
        if baseline is not None and method != baseline:
            paired = compare_paired_bootstrap(
                seeded, curves[baseline], confidence=confidence, n_resamples=n_resamples
            ).difference
        for i, cost in enumerate(cost_grid):
            row = {
                "method": method,
                "metric": metric,
                "cost": float(cost),
                "n_seeds": len(summary.seeds),
                "mean": float(summary.mean[i]),
                "lower": float(summary.lower[i]),
                "upper": float(summary.upper[i]),
                "confidence": confidence,
            }
            if paired is not None:
                row.update(
                    baseline=baseline,
                    difference=float(paired.mean[i]),
                    difference_lower=float(paired.lower[i]),
                    difference_upper=float(paired.upper[i]),
                )
            output.append(row)
    return output
