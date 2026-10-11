"""Paired strategy comparisons on a common empirical evaluation axis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.curve_statistics import (
    CurveStatistics,
    paired_metric_difference,
    summarize_metric_curves,
)
from robotorchan.benchmarks.metric_curves import MetricCurves


@dataclass(frozen=True)
class PairedComparison:
    """First-minus-second metric difference and its uncertainty."""

    first: str
    second: str
    statistics: CurveStatistics
    final_differences: Tensor
    final_win_rate: Tensor
    final_tie_rate: Tensor
    lower_is_better: bool


def compare_paired_strategies(
    curves: Mapping[str, MetricCurves],
    *,
    reference: str,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> tuple[PairedComparison, ...]:
    """Compare each alternative against a reference with paired seed bootstrap.

    Differences are alternative minus reference. For regret, negative differences
    favor the alternative; for hypervolume and feasibility, positive differences
    favor the alternative. Win rates are descriptive, not hypothesis tests.
    """
    if reference not in curves:
        raise ValueError("Reference strategy must exist in curves.")
    if len(curves) < 2:
        raise ValueError("At least two strategies are required.")
    baseline = curves[reference]
    lower_is_better = baseline.metric == "simple_regret"
    results = []
    for name, candidate in curves.items():
        if name == reference:
            continue
        differences = paired_metric_difference(candidate, baseline)
        stats = summarize_metric_curves(
            differences,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        final = differences.final_values
        wins = final < 0 if lower_is_better else final > 0
        results.append(
            PairedComparison(
                first=name,
                second=reference,
                statistics=stats,
                final_differences=final,
                final_win_rate=wins.to(dtype=final.dtype).mean(),
                final_tie_rate=(final == 0).to(dtype=final.dtype).mean(),
                lower_is_better=lower_is_better,
            )
        )
    return tuple(results)
