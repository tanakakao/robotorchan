"""Seed-aligned benchmark curve aggregation and paired strategy comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import torch
from torch import Tensor


@dataclass(frozen=True)
class CurveSummary:
    """Pointwise mean, sample std and normal-approximation confidence bounds."""

    seeds: tuple[int, ...]
    mean: Tensor
    std: Tensor
    lower: Tensor
    upper: Tensor


@dataclass(frozen=True)
class PairedCurveComparison:
    """Paired strategy difference (first minus second) over matching seeds."""

    seeds: tuple[int, ...]
    difference: CurveSummary


def summarize_curves(
    curves: dict[int, Tensor],
    *,
    confidence: float = 0.95,
) -> CurveSummary:
    """Aggregate equal-length 1D curves keyed by unique experiment seeds.

    Confidence intervals use a normal approximation with sample standard
    deviation; they are not exact small-sample Student-t intervals.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    if not curves:
        raise ValueError("At least one seeded curve is required.")
    seeds = tuple(sorted(curves))
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("Seed keys must be integers.")
    first = curves[seeds[0]]
    if not isinstance(first, Tensor) or first.ndim != 1 or first.numel() == 0:
        raise ValueError("Each curve must be a nonempty one-dimensional tensor.")
    if not first.is_floating_point() or not torch.isfinite(first).all():
        raise ValueError("Curves must contain finite floating-point values.")
    for seed in seeds[1:]:
        curve = curves[seed]
        if (
            not isinstance(curve, Tensor)
            or curve.shape != first.shape
            or curve.dtype != first.dtype
            or curve.device != first.device
            or not torch.isfinite(curve).all()
        ):
            raise ValueError("Curves must share shape, dtype, device and finite values.")
    stacked = torch.stack([curves[seed] for seed in seeds])
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=True) if len(seeds) > 1 else torch.zeros_like(mean)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    half_width = z * std / sqrt(len(seeds))
    return CurveSummary(
        seeds=seeds,
        mean=mean,
        std=std,
        lower=mean - half_width,
        upper=mean + half_width,
    )


def compare_paired_curves(
    first: dict[int, Tensor],
    second: dict[int, Tensor],
    *,
    confidence: float = 0.95,
) -> PairedCurveComparison:
    """Compare strategies using differences from identical experiment seeds."""
    if not first or first.keys() != second.keys():
        raise ValueError("Paired comparisons require identical nonempty seed sets.")
    differences = {seed: first[seed] - second[seed] for seed in first}
    summary = summarize_curves(differences, confidence=confidence)
    return PairedCurveComparison(seeds=summary.seeds, difference=summary)
