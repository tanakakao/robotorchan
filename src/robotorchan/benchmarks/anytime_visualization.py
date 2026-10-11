"""Backend-independent anytime performance data and optional plotting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from robotorchan.benchmarks.comparison_report import EmpiricalComparisonReport


@dataclass(frozen=True)
class AnytimeSeries:
    """One strategy's pointwise bootstrap confidence interval."""

    strategy: str
    evaluations: tuple[int, ...]
    mean: tuple[float, ...]
    ci_lower: tuple[float, ...]
    ci_upper: tuple[float, ...]


def prepare_anytime_series(
    report: EmpiricalComparisonReport,
) -> tuple[AnytimeSeries, ...]:
    """Extract validated, aligned curves without requiring a plotting backend."""
    from math import isfinite

    series = []
    common_axis: tuple[int, ...] | None = None
    for name, stats in sorted(report.strategies.items()):
        axis = tuple(stats["evaluations"])
        mean = tuple(float(value) for value in stats["mean"])
        lower = tuple(float(value) for value in stats["ci_lower"])
        upper = tuple(float(value) for value in stats["ci_upper"])
        if not axis or any(type(x) is not int or x < 0 for x in axis):
            raise ValueError("Evaluation axis must contain nonnegative integers.")
        if any(current <= previous for previous, current in pairwise(axis)):
            raise ValueError("Evaluation axis must be strictly increasing.")
        if common_axis is None:
            common_axis = axis
        elif axis != common_axis:
            raise ValueError("All strategies must share an evaluation axis.")
        if not (len(axis) == len(mean) == len(lower) == len(upper)):
            raise ValueError("Anytime curve lengths are inconsistent.")
        if any(not all(isfinite(value) for value in values) for values in (mean, lower, upper)):
            raise ValueError("Anytime statistics must be finite.")
        if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("Anytime confidence interval bounds are inconsistent.")
        series.append(
            AnytimeSeries(
                strategy=name,
                evaluations=axis,
                mean=mean,
                ci_lower=lower,
                ci_upper=upper,
            )
        )
    if not series:
        raise ValueError("At least one strategy is required.")
    return tuple(series)


def plot_anytime_performance(
    report: EmpiricalComparisonReport,
    *,
    ax=None,
):
    """Draw mean trajectories and pointwise bootstrap intervals with Matplotlib.

    Matplotlib is imported lazily so benchmark statistics remain usable without
    the optional plotting dependency. The caller owns figure lifecycle.
    """
    series = prepare_anytime_series(report)
    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install matplotlib to plot anytime performance.") from exc
        _, ax = plt.subplots()
    for item in series:
        (line,) = ax.plot(item.evaluations, item.mean, label=item.strategy)
        ax.fill_between(
            item.evaluations,
            item.ci_lower,
            item.ci_upper,
            color=line.get_color(),
            alpha=0.2,
        )
    ax.set_xlabel("Additional evaluations")
    ax.set_ylabel(report.metric.replace("_", " "))
    ax.legend()
    return ax


def save_anytime_plot(
    report: EmpiricalComparisonReport,
    path: str | Path,
    *,
    dpi: int = 150,
) -> None:
    """Save a standalone anytime comparison figure."""
    if type(dpi) is not int or dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    ax = plot_anytime_performance(report)
    try:
        ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt

        plt.close(ax.figure)
