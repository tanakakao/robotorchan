"""Phase 16 paired constrained benchmark comparison and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_continuous_constraints import ContinuousConstraintResult
from robotorchan.benchmarks.comparative_strength_conductivity_pass import (
    StrengthConductivityPassResult,
)
from robotorchan.benchmarks.comparative_strength_pass import StrengthPassResult
from robotorchan.benchmarks.runner import BenchmarkTrajectory

ConstrainedResult = (
    ContinuousConstraintResult | StrengthPassResult | StrengthConductivityPassResult
)


@dataclass(frozen=True)
class ConstrainedComparison:
    """Paired constrained scores at identical completed-batch checkpoints."""

    seeds: tuple[int, ...]
    evaluations: Tensor
    metric: str
    scores_by_method: dict[str, Tensor]
    feasibility_by_method: dict[str, Tensor]
    mean_score_by_method: dict[str, Tensor]
    standard_error_by_method: dict[str, Tensor]
    mean_feasibility_by_method: dict[str, Tensor]


def _metric(result: ConstrainedResult) -> tuple[str, Tensor]:
    if isinstance(result, StrengthConductivityPassResult):
        return "feasible_hypervolume", result.feasible_hypervolume
    if isinstance(result, StrengthPassResult):
        return "feasible_regret", result.feasible_regret
    return "feasible_regret", result.regret


def compare_constrained(
    results: Mapping[str, ConstrainedResult],
) -> ConstrainedComparison:
    """Compare paired runs without mixing incompatible constrained metrics.

    Feasible regret is minimized; feasible hypervolume is maximized.
    Standard error reflects between-seed variation, not significance.
    """
    if not results:
        raise ValueError("At least one constrained result is required.")
    seeds: tuple[int, ...] | None = None
    checkpoints: Tensor | None = None
    metric_name: str | None = None
    initial: dict[int, Tensor] = {}
    reference_dtype: torch.dtype | None = None
    reference_device: torch.device | None = None
    scores: dict[str, Tensor] = {}
    feasibility: dict[str, Tensor] = {}
    for method, result in results.items():
        if not result.trajectories:
            raise ValueError("Every method requires trajectories.")
        current_seeds = tuple(run.seed for run in result.trajectories)
        if len(set(current_seeds)) != len(current_seeds):
            raise ValueError("Duplicate seeds are not allowed.")
        name, values = _metric(result)
        if metric_name is None:
            metric_name = name
        elif name != metric_name:
            raise ValueError("Cannot mix regret and hypervolume.")
        if seeds is None:
            seeds = current_seeds
            checkpoints = result.evaluations.clone()
            initial = {
                run.seed: run.X[: run.initial_points].clone()
                for run in result.trajectories
            }
            reference_dtype = result.trajectories[0].X.dtype
            reference_device = result.trajectories[0].X.device
        if current_seeds != seeds or not torch.equal(result.evaluations, checkpoints):
            raise ValueError("Seeds and evaluation checkpoints must match.")
        for run in result.trajectories:
            if (
                run.X.dtype != reference_dtype
                or run.X.device != reference_device
                or not torch.equal(run.X[: run.initial_points], initial[run.seed])
            ):
                raise ValueError("Initial designs, dtypes, and devices must match.")
        expected = (len(seeds), checkpoints.numel())
        if values.shape != expected or result.feasibility_rate.shape != expected:
            raise ValueError("Metric shapes must match seeds and checkpoints.")
        scores[method] = values
        feasibility[method] = result.feasibility_rate

    assert seeds is not None
    assert checkpoints is not None
    assert metric_name is not None
    mean = {method: values.mean(dim=0) for method, values in scores.items()}
    stderr = {
        method: (
            values.std(dim=0, unbiased=True) / len(seeds) ** 0.5
            if len(seeds) > 1
            else torch.zeros_like(values[0])
        )
        for method, values in scores.items()
    }
    return ConstrainedComparison(
        seeds=seeds,
        evaluations=checkpoints,
        metric=metric_name,
        scores_by_method=scores,
        feasibility_by_method=feasibility,
        mean_score_by_method=mean,
        standard_error_by_method=stderr,
        mean_feasibility_by_method={
            method: values.mean(dim=0) for method, values in feasibility.items()
        },
    )
