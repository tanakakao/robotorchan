"""Align empirical optimization metric curves by additional evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from robotorchan.benchmarks.metrics import (
    cumulative_feasibility_rate,
    hypervolume_curve,
    simple_regret_curve,
)
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import BenchmarkTrajectory

MetricName = Literal["simple_regret", "hypervolume", "feasibility_rate"]


@dataclass(frozen=True)
class MetricCurves:
    """Per-seed metric histories on a common post-initial evaluation axis."""

    metric: MetricName
    seeds: tuple[int, ...]
    evaluations: tuple[int, ...]
    values: Tensor

    @property
    def mean(self) -> Tensor:
        """Mean across seeds at each evaluation count."""
        return self.values.mean(dim=0)

    @property
    def final_values(self) -> Tensor:
        """Metric attained after the final additional evaluation."""
        return self.values[:, -1]


def align_metric_curves(
    problem: BenchmarkProblem,
    trajectories: tuple[BenchmarkTrajectory, ...],
    metric: MetricName,
    *,
    include_initial: bool = True,
) -> MetricCurves:
    """Compute metric histories using truth, not noisy observed objectives.

    Evaluation zero represents the complete initial design when requested.
    Constrained simple regret may be infinite before any feasible observation.
    """
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    if len({run.seed for run in trajectories}) != len(trajectories):
        raise ValueError("Trajectory seeds must be unique.")
    if metric not in ("simple_regret", "hypervolume", "feasibility_rate"):
        raise ValueError(f"Unsupported metric: {metric!r}.")
    budget = trajectories[0].evaluation_count
    initial_points = trajectories[0].initial_points
    if budget < 1:
        raise ValueError("At least one additional evaluation is required.")
    curves = []
    for run in trajectories:
        if run.evaluation_count != budget or run.initial_points != initial_points:
            raise ValueError("All trajectories must share initial points and evaluation budget.")
        if metric == "simple_regret":
            curve = simple_regret_curve(problem, run)
        elif metric == "hypervolume":
            curve = hypervolume_curve(problem, run)
        else:
            curve = cumulative_feasibility_rate(run.constraints)
        if curve.ndim != 1 or curve.shape[0] != initial_points + budget:
            raise ValueError("Metric curve length does not match trajectory history.")
        curves.append(curve[initial_points - 1 :] if include_initial else curve[initial_points:])
    values = torch.stack(curves)
    evaluations = tuple(range(budget + 1)) if include_initial else tuple(range(1, budget + 1))
    return MetricCurves(
        metric=metric,
        seeds=tuple(run.seed for run in trajectories),
        evaluations=evaluations,
        values=values,
    )
