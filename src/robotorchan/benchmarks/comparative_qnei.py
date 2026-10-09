"""Phase 7 comparative qNEI execution using BoTorch-native acquisition."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from robotorchan.benchmarks.botorch_strategy import botorch_gp_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def make_comparative_qnei_strategy(
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
) -> Callable[[BenchmarkProblem, Tensor, Tensor, int, torch.Generator], Tensor]:
    """Build a qNEI strategy using observed history as X_baseline."""
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")

    def propose(
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        return botorch_gp_candidates(
            problem,
            X,
            Y,
            q,
            generator,
            acquisition="qNEI",
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return propose


def run_comparative_qnei(
    cell: ComparativeExperimentCell,
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run qNEI on validated unconstrained single-objective comparisons."""
    if cell.config.strategy != "qNEI":
        raise ValueError("This runner requires a qNEI comparison cell.")
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    strategy = make_comparative_qnei_strategy(
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return run_benchmark(cell.config, strategy, registry=registry)
