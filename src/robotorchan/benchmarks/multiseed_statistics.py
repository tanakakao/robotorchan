"""Seed-aware bootstrap uncertainty and paired benchmark statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparison import summarize_curves


@dataclass(frozen=True)
class BootstrapCurveSummary:
    """Pointwise percentile bootstrap confidence intervals over independent seeds."""

    seeds: tuple[int, ...]
    mean: Tensor
    lower: Tensor
    upper: Tensor
    confidence: float
    n_resamples: int


@dataclass(frozen=True)
class PairedBootstrapComparison:
    """Bootstrap uncertainty of first-minus-second paired seed differences."""

    seeds: tuple[int, ...]
    difference: BootstrapCurveSummary
    probability_positive: Tensor


def bootstrap_curves(
    curves: dict[int, Tensor],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCurveSummary:
    """Resample whole seed trajectories, preserving within-curve correlation.

    Percentile bounds describe between-seed uncertainty of the mean.
    They are not simultaneous confidence bands or significance tests.
    """
    summary = summarize_curves(curves, confidence=confidence)
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    stacked = torch.stack([curves[key] for key in summary.seeds])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        stacked.shape[0],
        (n_resamples, stacked.shape[0]),
        generator=generator,
    ).to(device=stacked.device)
    resampled = stacked[indices].mean(dim=1)
    tail = (1 - confidence) / 2
    return BootstrapCurveSummary(
        seeds=summary.seeds,
        mean=summary.mean,
        lower=torch.quantile(resampled, tail, dim=0),
        upper=torch.quantile(resampled, 1 - tail, dim=0),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def compare_paired_bootstrap(
    first: dict[int, Tensor],
    second: dict[int, Tensor],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedBootstrapComparison:
    """Bootstrap paired differences, not independent strategy samples."""
    if not first or first.keys() != second.keys():
        raise ValueError("Paired comparisons require identical nonempty seed sets.")
    differences = {key: first[key] - second[key] for key in first}
    summary = bootstrap_curves(
        differences,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    # Resample the paired differences using the same reproducible seed.
    stacked = torch.stack([differences[key] for key in summary.seeds])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        stacked.shape[0],
        (n_resamples, stacked.shape[0]),
        generator=generator,
    ).to(device=stacked.device)
    probability_positive = (stacked[indices].mean(dim=1) > 0).to(stacked.dtype).mean(dim=0)
    return PairedBootstrapComparison(
        seeds=summary.seeds,
        difference=summary,
        probability_positive=probability_positive,
    )
