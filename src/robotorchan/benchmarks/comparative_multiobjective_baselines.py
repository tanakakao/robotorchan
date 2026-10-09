"""Phase 9 multiobjective baseline evaluation with truth-based hypervolume."""

from __future__ import annotations

from dataclasses import dataclass

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
    curves = [hypervolume_curve(problem, run)[initial - 1 :] for run in trajectories]
    import torch

    return MultiobjectiveBaselineResult(
        trajectories=trajectories,
        evaluations=torch.arange(cell.config.evaluation_budget + 1, dtype=torch.long),
        hypervolume=torch.stack(curves),
    )
