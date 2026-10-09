"""Truth-based benchmark metrics for single- and multiobjective optimization."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import BenchmarkTrajectory


def feasibility_rate(constraints: Tensor) -> Tensor:
    """Fraction of feasible evaluated points; constraints use g(X) >= 0."""
    if constraints.ndim < 2:
        raise ValueError("constraints must have shape (..., n, c).")
    if constraints.shape[-2] == 0:
        raise ValueError("At least one evaluated point is required.")
    if constraints.shape[-1] == 0:
        return constraints.new_ones(constraints.shape[:-2])
    return (constraints >= 0).all(dim=-1).to(dtype=constraints.dtype).mean(dim=-1)


def cumulative_feasibility_rate(constraints: Tensor) -> Tensor:
    """Feasibility rate after each evaluation, including the initial design."""
    if constraints.ndim != 2 or constraints.shape[0] == 0:
        raise ValueError("constraints must have shape (n, c) with n > 0.")
    feasible = (constraints >= 0).all(dim=-1).to(dtype=constraints.dtype)
    return feasible.cumsum(dim=0) / torch.arange(
        1, constraints.shape[0] + 1, dtype=constraints.dtype, device=constraints.device
    )


def simple_regret_curve(problem: BenchmarkProblem, trajectory: BenchmarkTrajectory) -> Tensor:
    """Best feasible truth-based regret after every evaluation."""
    if problem.n_objectives != 1 or problem.optimal_value is None:
        raise ValueError("Simple regret requires a known single-objective optimum.")
    truth = problem.to_maximization(trajectory.Y_truth).squeeze(-1)
    if problem.n_constraints:
        feasible = (trajectory.constraints >= 0).all(dim=-1)
        truth = truth.masked_fill(~feasible, -torch.inf)
    best = truth.cummax(dim=0).values
    optimum = problem.to_maximization(
        problem.optimal_value.to(dtype=truth.dtype, device=truth.device)
    ).squeeze(-1)
    return (optimum - best).clamp_min(0)


def hypervolume_2d(values: Tensor, reference_point: Tensor) -> Tensor:
    """Exact dominated hypervolume for 2 objectives, both in maximization orientation.

    Points dominated by the reference point contribute zero. Duplicate and
    mutually dominated points are supported.
    """
    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError("values must have shape (n, 2).")
    if reference_point.shape != (2,):
        raise ValueError("reference_point must have shape (2,).")
    if not torch.isfinite(values).all() or not torch.isfinite(reference_point).all():
        raise ValueError("Hypervolume inputs must be finite.")
    if values.device != reference_point.device:
        raise ValueError("values and reference_point must be on the same device.")
    shifted = (values - reference_point).clamp_min(0)
    if shifted.shape[0] == 0:
        return values.new_zeros(())
    ordered = shifted[torch.argsort(shifted[:, 0], descending=True)]
    width = ordered[:, 0]
    height = ordered[:, 1].cummax(dim=0).values
    next_width = torch.cat((width[1:], width.new_zeros(1)))
    return ((width - next_width) * height).sum()


def hypervolume_curve(problem: BenchmarkProblem, trajectory: BenchmarkTrajectory) -> Tensor:
    """Exact feasible 2D hypervolume after each evaluation.

    Hypervolume is defined only for two objectives in this initial metrics phase.
    """
    if problem.n_objectives != 2 or problem.reference_point is None:
        raise ValueError("Hypervolume requires two objectives and a reference point.")
    values = problem.to_maximization(trajectory.Y_truth)
    reference = problem.to_maximization(
        problem.reference_point.to(dtype=values.dtype, device=values.device)
    )
    feasible = (trajectory.constraints >= 0).all(dim=-1)
    return torch.stack(
        [hypervolume_2d(values[:i][feasible[:i]], reference) for i in range(1, len(values) + 1)]
    )
