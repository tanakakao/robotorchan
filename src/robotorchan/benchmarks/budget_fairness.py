"""Validate evaluation budgets and batch schedules in empirical comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class BudgetFairnessResult:
    """Summary of a validated cross-strategy evaluation schedule."""

    seeds: tuple[int, ...]
    evaluation_budget: int
    initial_points: int
    q: int
    completed_batches: tuple[int, ...]
    cost_paired: bool


def expected_completed_batches(budget: int, q: int) -> tuple[int, ...]:
    """Return cumulative batch endpoints, including a short final batch."""
    if type(budget) is not int or budget < 1 or type(q) is not int or q < 1:
        raise ValueError("budget and q must be positive integers.")
    return tuple(range(q, budget + 1, q)) + ((budget,) if budget % q else ())


def validate_budget_fairness(
    experiments: Mapping[str, tuple[BenchmarkExperimentConfig, tuple[BenchmarkTrajectory, ...]]],
    *,
    require_cost_pairing: bool = False,
) -> BudgetFairnessResult:
    """Validate identical budgets, batch schedules and observed evaluation counts.

    Cost pairing is optional: heterogeneous candidate locations can legitimately
    incur different costs even under identical evaluation budgets.
    """
    if len(experiments) < 2:
        raise ValueError("At least two strategies are required.")
    reference_config = next(iter(experiments.values()))[0]
    seeds = tuple(sorted(reference_config.seeds))
    schedule = expected_completed_batches(reference_config.evaluation_budget, reference_config.q)
    reference_costs: dict[int, torch.Tensor] = {}
    cost_paired = True
    for index, (name, (config, runs)) in enumerate(experiments.items()):
        if (
            config.problem != reference_config.problem
            or config.initial_points != reference_config.initial_points
            or config.evaluation_budget != reference_config.evaluation_budget
            or config.q != reference_config.q
            or config.dtype != reference_config.dtype
            or config.device != reference_config.device
        ):
            raise ValueError(f"Strategy {name!r} has mismatched experiment settings.")
        if tuple(sorted(config.seeds)) != seeds:
            raise ValueError(f"Strategy {name!r} has mismatched seed sets.")
        if len(runs) != len(seeds) or {run.seed for run in runs} != set(seeds):
            raise ValueError(f"Strategy {name!r} has missing or duplicate trajectories.")
        for run in runs:
            if run.initial_points != config.initial_points:
                raise ValueError("Trajectory initial-point count mismatch.")
            if run.evaluation_count != config.evaluation_budget:
                raise ValueError("Trajectory evaluation budget mismatch.")
            if run.completed_batches != schedule:
                raise ValueError("Trajectory completed-batch schedule mismatch.")
            total = config.initial_points + config.evaluation_budget
            if any(
                tensor.shape[0] != total
                for tensor in (
                    run.X,
                    run.Y_observed,
                    run.Y_truth,
                    run.constraints,
                    run.costs,
                )
            ):
                raise ValueError("Trajectory history lengths are inconsistent.")
            if run.costs.shape[-1] != 1 or not torch.isfinite(run.costs).all():
                raise ValueError("Trajectory costs must be finite with width one.")
            if (run.costs < 0).any():
                raise ValueError("Trajectory costs must be nonnegative.")
            if len(run.candidate_seconds) != len(schedule):
                raise ValueError("Candidate timing count does not match batch schedule.")
            if len(run.evaluation_seconds) != len(schedule):
                raise ValueError("Evaluation timing count does not match batch schedule.")
            if index == 0:
                reference_costs[run.seed] = run.costs
            else:
                cost_paired &= torch.equal(reference_costs[run.seed], run.costs)
    if require_cost_pairing and not cost_paired:
        raise ValueError("Evaluation costs differ across strategies.")
    return BudgetFairnessResult(
        seeds=seeds,
        evaluation_budget=reference_config.evaluation_budget,
        initial_points=reference_config.initial_points,
        q=reference_config.q,
        completed_batches=schedule,
        cost_paired=cost_paired,
    )
