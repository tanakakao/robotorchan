"""Seed-level descriptive statistics and paired bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.metric_curves import MetricCurves


@dataclass(frozen=True)
class CurveStatistics:
    """Summary of one metric curve across independent experimental seeds."""

    evaluations: tuple[int, ...]
    n_seeds: int
    mean: Tensor
    std: Tensor
    sem: Tensor
    ci_lower: Tensor
    ci_upper: Tensor
    confidence_level: float
    bootstrap_samples: int


def summarize_metric_curves(
    curves: MetricCurves,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> CurveStatistics:
    """Compute pointwise percentile bootstrap intervals by resampling whole seeds.

    Intervals describe uncertainty in the mean curve and are not simultaneous
    confidence bands. Infinite regret before feasibility is not supported.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer.")
    values = curves.values
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("At least two seed curves are required.")
    if values.shape[1] != len(curves.evaluations) or not torch.isfinite(values).all():
        raise ValueError("Curve values must be finite and aligned with evaluations.")
    n = values.shape[0]
    std = values.std(dim=0, correction=1)
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    indices = torch.randint(n, (bootstrap_samples, n), generator=generator)
    boot_means = values[indices.to(device=values.device)].mean(dim=1)
    alpha = (1 - confidence_level) / 2
    return CurveStatistics(
        evaluations=curves.evaluations,
        n_seeds=n,
        mean=values.mean(dim=0),
        std=std,
        sem=std / n**0.5,
        ci_lower=torch.quantile(boot_means, alpha, dim=0),
        ci_upper=torch.quantile(boot_means, 1 - alpha, dim=0),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
    )


def paired_metric_difference(
    first: MetricCurves,
    second: MetricCurves,
) -> MetricCurves:
    """Return first-minus-second differences paired by seed, not tuple order."""
    if first.metric != second.metric or first.evaluations != second.evaluations:
        raise ValueError("Paired curves must share metric and evaluation axis.")
    if len(set(first.seeds)) != len(first.seeds) or len(set(second.seeds)) != len(
        second.seeds
    ):
        raise ValueError("Paired curve seeds must be unique.")
    if set(first.seeds) != set(second.seeds):
        raise ValueError("Paired curves must share identical seed sets.")
    if first.values.shape != (len(first.seeds), len(first.evaluations)):
        raise ValueError("First curve shape is inconsistent.")
    if second.values.shape != (len(second.seeds), len(second.evaluations)):
        raise ValueError("Second curve shape is inconsistent.")
    by_seed = {seed: i for i, seed in enumerate(second.seeds)}
    reordered = second.values[[by_seed[seed] for seed in first.seeds]]
    return MetricCurves(
        metric=first.metric,
        seeds=first.seeds,
        evaluations=first.evaluations,
        values=first.values - reordered,
    )
