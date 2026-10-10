"""Phase 18 timing and evaluation-cost summaries for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class BenchmarkEfficiency:
    """Seed-wise wall time and cumulative evaluation cost at batch boundaries."""

    seeds: tuple[int, ...]
    evaluations: Tensor
    candidate_seconds: Tensor
    evaluation_seconds: Tensor
    total_seconds: Tensor
    evaluation_cost: Tensor


def summarize_efficiency(
    trajectories: tuple[BenchmarkTrajectory, ...],
    *,
    q: int,
) -> BenchmarkEfficiency:
    """Summarize instrumented runs without conflating wall time with oracle cost.

    Wall time is measured per completed batch; evaluation cost comes from the
    benchmark's explicit cost function. Initial evaluation time is included.
    """
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    if type(q) is not int or q < 1:
        raise ValueError("q must be a positive integer.")
    budget = trajectories[0].evaluation_count
    initial = trajectories[0].initial_points
    checkpoints = [0, *range(q, budget + 1, q)]
    if checkpoints[-1] != budget:
        checkpoints.append(budget)
    candidate_rows = []
    evaluation_rows = []
    cost_rows = []
    seeds = []
    for run in trajectories:
        if run.evaluation_count != budget or run.initial_points != initial:
            raise ValueError("Runs must have matching budgets and initial designs.")
        if run.seed in seeds:
            raise ValueError("Duplicate seeds are not allowed.")
        seeds.append(run.seed)
        if run.completed_batches != tuple(checkpoints[1:]):
            raise ValueError("Recorded batch boundaries must match q and budget.")
        batches = len(checkpoints) - 1
        if (
            len(run.candidate_seconds) != batches
            or len(run.evaluation_seconds) != batches
            or run.initial_evaluation_seconds is None
        ):
            raise ValueError("Run has missing or incompatible timing records.")
        times = torch.tensor(run.candidate_seconds, dtype=torch.float64)
        eval_times = torch.tensor(run.evaluation_seconds, dtype=torch.float64)
        initial_time = run.initial_evaluation_seconds
        if (
            not torch.isfinite(times).all()
            or not torch.isfinite(eval_times).all()
            or not torch.isfinite(torch.tensor(initial_time))
            or (times < 0).any()
            or (eval_times < 0).any()
            or initial_time < 0
        ):
            raise ValueError("Timing records must be finite and nonnegative.")
        candidate_rows.append(torch.cat((torch.zeros(1), times.cumsum(0))))
        evaluation_rows.append(
            torch.cat((torch.tensor([initial_time]), initial_time + eval_times.cumsum(0)))
        )
        indices = torch.tensor([initial + count - 1 for count in checkpoints])
        cost_rows.append(run.cumulative_cost.index_select(0, indices.to(run.X.device)))
    candidate = torch.stack(candidate_rows)
    evaluation = torch.stack(evaluation_rows)
    cost = torch.stack(cost_rows)
    if not torch.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Evaluation costs must be finite and nonnegative.")
    return BenchmarkEfficiency(
        seeds=tuple(seeds),
        evaluations=torch.tensor(checkpoints, dtype=torch.long),
        candidate_seconds=candidate,
        evaluation_seconds=evaluation,
        total_seconds=candidate + evaluation,
        evaluation_cost=cost,
    )
