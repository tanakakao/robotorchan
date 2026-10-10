"""Paired multiobjective hypervolume comparison for Phase 12."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.metrics import hypervolume_curve
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class MultiobjectiveComparison:
    """Truth-based hypervolume statistics at completed batch checkpoints."""

    seeds: tuple[int, ...]
    q: int
    evaluations: Tensor
    hypervolume_by_method: dict[str, Tensor]
    mean_hypervolume_by_method: dict[str, Tensor]
    standard_error_by_method: dict[str, Tensor]


def compare_multiobjective(
    problem: BenchmarkProblem,
    trajectories: Mapping[str, tuple[BenchmarkTrajectory, ...]],
    *,
    q_by_method: Mapping[str, int],
) -> MultiobjectiveComparison:
    """Compare paired trajectories with identical initial designs and budgets.

    Evaluation zero is the hypervolume of the initial design. Intermediate
    points inside a jointly proposed batch are not treated as checkpoints.
    Standard errors describe seed variability, not statistical significance.
    """
    if problem.n_objectives != 2 or problem.reference_point is None:
        raise ValueError("Comparison requires a two-objective reference point.")
    if not trajectories:
        raise ValueError("At least one method is required.")
    if not set(trajectories).issubset({"random", "sobol", "qEHVI", "qNEHVI"}):
        raise ValueError("Unknown multiobjective comparison method.")
    if set(q_by_method) != set(trajectories):
        raise ValueError("Batch-size metadata must cover every method.")
    if any(type(q) is not int or q < 1 for q in q_by_method.values()):
        raise ValueError("Batch sizes must be positive integers.")
    if len(set(q_by_method.values())) != 1:
        raise ValueError("Batch sizes must match across methods.")

    q = next(iter(q_by_method.values()))
    reference_seeds: tuple[int, ...] | None = None
    reference_initial: int | None = None
    reference_budget: int | None = None
    reference_X: dict[int, Tensor] = {}
    reference_dtype: torch.dtype | None = None
    reference_device: torch.device | None = None
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
            reference_dtype = runs[0].X.dtype
            reference_device = runs[0].X.device
        if seeds != reference_seeds:
            raise ValueError("Seed ordering must match across methods.")
        method_curves = []
        for run in runs:
            if any(
                tensor.dtype != reference_dtype or tensor.device != reference_device
                for tensor in (run.X, run.Y_observed, run.Y_truth, run.constraints, run.costs)
            ):
                raise ValueError("Trajectory dtypes and devices must match across methods.")
            if (
                run.initial_points != reference_initial
                or run.evaluation_count != reference_budget
                or not torch.equal(run.X[:reference_initial], reference_X[run.seed])
            ):
                raise ValueError("Initial designs and evaluation budgets must match.")
            method_curves.append(hypervolume_curve(problem, run))
        curves[method] = torch.stack(method_curves)

    assert reference_seeds is not None
    assert reference_initial is not None
    assert reference_budget is not None
    checkpoints = [0, *range(q, reference_budget + 1, q)]
    if checkpoints[-1] != reference_budget:
        checkpoints.append(reference_budget)
    indices = torch.tensor([reference_initial + count - 1 for count in checkpoints])
    selected = {method: values.index_select(1, indices) for method, values in curves.items()}
    mean = {method: values.mean(dim=0) for method, values in selected.items()}
    standard_error = {
        method: (
            values.std(dim=0, unbiased=True) / len(reference_seeds) ** 0.5
            if len(reference_seeds) > 1
            else torch.zeros_like(values[0])
        )
        for method, values in selected.items()
    }
    return MultiobjectiveComparison(
        seeds=reference_seeds,
        q=q,
        evaluations=torch.tensor(checkpoints, dtype=torch.long),
        hypervolume_by_method=selected,
        mean_hypervolume_by_method=mean,
        standard_error_by_method=standard_error,
    )
