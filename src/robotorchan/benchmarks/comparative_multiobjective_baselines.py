"""Phase 9 multiobjective baseline evaluation with truth-based hypervolume."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.metrics import hypervolume_curve
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class MultiobjectiveBaselineResult:
    """Per-seed trajectories and hypervolume at evaluation counts 0 through budget."""

    trajectories: tuple[BenchmarkTrajectory, ...]
    evaluations: Tensor
    hypervolume: Tensor


def run_multiobjective_baseline(cell: ComparativeExperimentCell) -> MultiobjectiveBaselineResult:
    """Evaluate Random or Sobol on the validated Branin-Currin matrix."""
    if cell.config.problem != "branin_currin":
        raise ValueError("Multiobjective baseline requires branin_currin.")
    if cell.config.strategy not in ("random", "sobol"):
        raise ValueError("Multiobjective baseline requires random or sobol.")
    trajectories = run_comparative_baseline(cell)
    problem = branin_currin()
    initial = cell.config.initial_points
    budget = cell.config.evaluation_budget
    q = cell.config.q
    checkpoints = [0, *range(q, budget + 1, q)]
    if checkpoints[-1] != budget:
        checkpoints.append(budget)
    curves = [
        hypervolume_curve(problem, run)[
            torch.tensor([initial + count - 1 for count in checkpoints], dtype=torch.long)
        ]
        for run in trajectories
    ]
    return MultiobjectiveBaselineResult(
        trajectories=trajectories,
        evaluations=torch.tensor(checkpoints, dtype=torch.long),
        hypervolume=torch.stack(curves),
    )
