"""Reproducible Random and Sobol candidates for comparative benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
)


def sobol_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    q: int,
    generator: torch.Generator,
    *,
    initial_points: int,
) -> Tensor:
    """Draw the next q scrambled Sobol points without inspecting outcomes."""
    del Y
    engine = torch.quasirandom.SobolEngine(
        dimension=problem.dimension,
        scramble=True,
        seed=generator.initial_seed() + 1,
    )
    engine.fast_forward(X.shape[0] - initial_points)
    unit = engine.draw(q).to(dtype=X.dtype, device=X.device)
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    candidates = bounds[0] + unit * (bounds[1] - bounds[0])
    for index, kind in enumerate(problem.variable_types):
        if kind != "continuous":
            candidates[:, index] = candidates[:, index].round()
    return candidates


def run_comparative_baseline(
    cell: ComparativeExperimentCell,
) -> tuple[BenchmarkTrajectory, ...]:
    """Execute a validated baseline cell with paired initial designs."""
    config = cell.config
    if config.strategy == "random":
        return run_benchmark(config, random_candidates)
    if config.strategy != "sobol":
        raise ValueError("Only Random and Sobol are supported by this runner.")

    def candidate_generator(
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        return sobol_candidates(
            problem,
            X,
            Y,
            q,
            generator,
            initial_points=config.initial_points,
        )

    return run_benchmark(config, candidate_generator)
