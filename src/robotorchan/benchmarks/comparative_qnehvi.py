"""Phase 11 comparative qNEHVI runner for Branin-Currin."""

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


def make_comparative_qnehvi_strategy(
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 128,
) -> Callable[[BenchmarkProblem, Tensor, Tensor, int, torch.Generator], Tensor]:
    """Use observed training data as the noisy HV improvement baseline."""
    for name, value in (
        ("num_restarts", num_restarts),
        ("raw_samples", raw_samples),
        ("mc_samples", mc_samples),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive.")

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
            acquisition="qNEHVI",
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            mc_samples=mc_samples,
        )

    return propose


def run_comparative_qnehvi(
    cell: ComparativeExperimentCell,
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 128,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run qNEHVI under the validated Branin-Currin comparison protocol."""
    if cell.config.problem != "branin_currin" or cell.config.strategy != "qNEHVI":
        raise ValueError("qNEHVI comparison requires a branin_currin qNEHVI cell.")
    registry = BenchmarkProblemRegistry()
    register_multiobjective_problems(registry)
    strategy = make_comparative_qnehvi_strategy(
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        mc_samples=mc_samples,
    )
    return run_benchmark(cell.config, strategy, registry=registry)
