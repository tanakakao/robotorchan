"""Phase 10 comparative qEHVI runner for Branin-Currin."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from robotorchan.benchmarks.botorch_multiobjective_strategy import botorch_qehvi_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.multiobjective_problems import register_multiobjective_problems
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, run_benchmark


def make_comparative_qehvi_strategy(
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 128,
) -> Callable[[BenchmarkProblem, Tensor, Tensor, int, torch.Generator], Tensor]:
    """Construct a qEHVI callback using observed outcomes only."""
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")

    if type(mc_samples) is not int or mc_samples < 1:
        raise ValueError("mc_samples must be positive.")

    def propose(
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        return botorch_qehvi_candidates(
            problem,
            X,
            Y,
            q,
            generator,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            mc_samples=mc_samples,
        )

    return propose


def run_comparative_qehvi(
    cell: ComparativeExperimentCell,
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 128,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run qEHVI on the validated Branin-Currin comparison cell."""
    if cell.config.problem != "branin_currin" or cell.config.strategy != "qEHVI":
        raise ValueError("qEHVI comparison requires a branin_currin qEHVI cell.")
    registry = BenchmarkProblemRegistry()
    register_multiobjective_problems(registry)
    strategy = make_comparative_qehvi_strategy(
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return run_benchmark(cell.config, strategy, registry=registry)
