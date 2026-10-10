"""Phase 21 renderer-independent benchmark visualization data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_constrained import ConstrainedComparison
from robotorchan.benchmarks.comparative_efficiency import BenchmarkEfficiency
from robotorchan.benchmarks.comparative_single_objective import SingleObjectiveComparison


@dataclass(frozen=True)
class PlotSeries:
    """Validated CPU-side values for one line with optional uncertainty bands."""

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    lower: tuple[float, ...] | None = None
    upper: tuple[float, ...] | None = None


@dataclass(frozen=True)
class BenchmarkPlot:
    """Plot specification independent of Matplotlib or Plotly."""

    title: str
    x_label: str
    y_label: str
    series: tuple[PlotSeries, ...]


def _series(
    name: str,
    x: Tensor,
    y: Tensor,
    *,
    error: Tensor | None = None,
) -> PlotSeries:
    if x.ndim != 1 or y.ndim != 1 or x.numel() != y.numel() or not x.numel():
        raise ValueError("Plot coordinates must be nonempty aligned vectors.")
    if not torch.isfinite(x).all():
        raise ValueError("Plot checkpoints must be finite.")
    if x.numel() > 1 and (x[1:] <= x[:-1]).any():
        raise ValueError("Plot checkpoints must increase.")
    if torch.isnan(y).any() or torch.isneginf(y).any():
        raise ValueError("Plot values must not contain NaN or negative infinity.")
    lower = upper = None
    if error is not None:
        if error.shape != y.shape or (error < 0).any():
            raise ValueError("Uncertainty must be nonnegative and aligned.")
        if torch.isinf(error).any():
            raise ValueError("Uncertainty must not be infinite.")
        # A pre-feasibility +inf regret has an undefined standard error.
        # Preserve the score and omit the uncertainty band at that checkpoint.
        invalid_band = torch.isnan(error)
        if (invalid_band & torch.isfinite(y)).any():
            raise ValueError("Finite scores require finite uncertainty.")
        if invalid_band.any():
            lower = upper = None
        else:
            lower = y - error
            upper = y + error
    return PlotSeries(
        name=name,
        x=tuple(float(v) for v in x.detach().cpu().tolist()),
        y=tuple(float(v) for v in y.detach().cpu().tolist()),
        lower=None if lower is None else tuple(float(v) for v in lower.detach().cpu().tolist()),
        upper=None if upper is None else tuple(float(v) for v in upper.detach().cpu().tolist()),
    )


def plot_single_objective(comparison: SingleObjectiveComparison) -> BenchmarkPlot:
    """Build mean regret curves with one-standard-error bands."""
    series = tuple(
        _series(
            method,
            comparison.evaluations,
            values,
            error=comparison.standard_error_by_method[method],
        )
        for method, values in sorted(comparison.mean_regret_by_method.items())
    )
    return BenchmarkPlot("Single-objective comparison", "Evaluations", "Simple regret", series)


def plot_constrained(comparison: ConstrainedComparison) -> BenchmarkPlot:
    """Build feasible regret or hypervolume curves."""
    series = tuple(
        _series(
            method,
            comparison.evaluations,
            values,
            error=comparison.standard_error_by_method[method],
        )
        for method, values in sorted(comparison.mean_score_by_method.items())
    )
    return BenchmarkPlot("Constrained comparison", "Evaluations", comparison.metric, series)


def plot_efficiency(
    methods: dict[str, BenchmarkEfficiency],
    *,
    metric: str = "total_seconds",
) -> BenchmarkPlot:
    """Plot per-method seed-mean wall time or explicit evaluation cost."""
    if metric not in (
        "candidate_seconds",
        "evaluation_seconds",
        "total_seconds",
        "evaluation_cost",
    ):
        raise ValueError("Unknown efficiency metric.")
    if not methods:
        raise ValueError("At least one method is required.")
    series = []
    for method, result in sorted(methods.items()):
        values = getattr(result, metric)
        if values.ndim != 2 or values.shape != (
            len(result.seeds),
            result.evaluations.numel(),
        ):
            raise ValueError("Efficiency data must match seeds and checkpoints.")
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Efficiency data must be finite and nonnegative.")
        series.append(_series(method, result.evaluations, values.mean(dim=0)))
    return BenchmarkPlot("Benchmark efficiency", "Evaluations", metric, tuple(series))
