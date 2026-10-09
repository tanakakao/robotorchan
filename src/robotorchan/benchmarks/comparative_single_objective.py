"""Paired single-objective comparison metrics for Phase 8."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.metrics import simple_regret_curve
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class SingleObjectiveComparison:
    """Regret curves aligned by seed and candidate evaluation count."""

    seeds: tuple[int, ...]
    q: int
    evaluations: Tensor
    regret_by_method: dict[str, Tensor]
    mean_regret_by_method: dict[str, Tensor]
    standard_error_by_method: dict[str, Tensor]


def compare_single_objective(
    problem: BenchmarkProblem,
    trajectories: Mapping[str, tuple[BenchmarkTrajectory, ...]],
    *,
    q_by_method: Mapping[str, int],
) -> SingleObjectiveComparison:
    """Compare methods with identical seeds, initial design, and budgets.

    Curves exclude the initial design and include every candidate evaluation.
    The standard error is zero for a single seed; no statistical significance
    is inferred from this descriptive summary.
    """
    if not trajectories:
        raise ValueError("At least one method is required.")
    if set(q_by_method) != set(trajectories):
        raise ValueError("Batch-size metadata must cover every method.")
    if any(type(q) is not int or q < 1 for q in q_by_method.values()):
        raise ValueError("Batch sizes must be positive integers.")
    if len(set(q_by_method.values())) != 1:
        raise ValueError("Batch sizes must match across methods.")
    if problem.n_objectives != 1 or problem.optimal_value is None:
        raise ValueError("Comparison requires a known single-objective optimum.")
    if not set(trajectories).issubset({"random", "sobol", "qEI", "qNEI"}):
        raise ValueError("Unknown single-objective comparison method.")

    reference_seeds: tuple[int, ...] | None = None
    reference_initial: int | None = None
    reference_budget: int | None = None
    reference_X: dict[int, Tensor] = {}
    curves: dict[str, Tensor] = {}
    for method, runs in trajectories.items():
        if not runs:
            raise ValueError("Each method must contain at least one trajectory.")
        seeds = tuple(run.seed for run in runs)
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed in a method.")
        if reference_seeds is None:
            reference_seeds = seeds
            reference_initial = runs[0].initial_points
            reference_budget = runs[0].evaluation_count
            if reference_initial < 1 or reference_budget < 1:
                raise ValueError("Initial design and evaluation budget must be positive.")
            reference_X = {run.seed: run.X[:reference_initial].clone() for run in runs}
        if seeds != reference_seeds:
            raise ValueError("Seed ordering must match across methods.")
        method_curves = []
        for run in runs:
            if (
                run.initial_points != reference_initial
                or run.evaluation_count != reference_budget
                or not torch.equal(run.X[:reference_initial], reference_X[run.seed])
            ):
                raise ValueError("Initial designs and evaluation budgets must match.")
            regret = simple_regret_curve(problem, run)[reference_initial - 1 :]
            method_curves.append(regret)
        curves[method] = torch.stack(method_curves)

    assert reference_seeds is not None
    assert reference_budget is not None
    mean = {method: values.mean(dim=0) for method, values in curves.items()}
    standard_error = {
        method: (
            values.std(dim=0, unbiased=True) / len(reference_seeds) ** 0.5
            if len(reference_seeds) > 1
            else torch.zeros_like(values[0])
        )
        for method, values in curves.items()
    }
    return SingleObjectiveComparison(
        seeds=reference_seeds,
        q=next(iter(q_by_method.values())),
        evaluations=torch.arange(reference_budget + 1, dtype=torch.long),
        regret_by_method=curves,
        mean_regret_by_method=mean,
        standard_error_by_method=standard_error,
    )
