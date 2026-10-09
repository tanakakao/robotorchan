"""Reproducible reference strategies for benchmark comparisons."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from robotorchan.benchmarks.botorch_strategy import make_botorch_gp_strategy
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.runner import CandidateGenerator, random_candidates


def sobol_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    q: int,
    generator: torch.Generator,
) -> Tensor:
    """Draw a scrambled Sobol batch using a run-local random seed.

    Every callback draws a fresh independently scrambled block, not a
    persistent low-discrepancy stream across iterations.
    """
    del Y
    if q < 1 or X.ndim != 2 or X.shape[1] != problem.dimension:
        raise ValueError("Expected X=(n,d) and q >= 1.")
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    engine = torch.quasirandom.SobolEngine(problem.dimension, scramble=True, seed=seed)
    unit = engine.draw(q).to(device=X.device, dtype=X.dtype)
    bounds = problem.bounds.to(device=X.device, dtype=X.dtype)
    candidates = bounds[0] + unit * (bounds[1] - bounds[0])
    for index, kind in enumerate(problem.variable_types):
        if kind != "continuous":
            candidates[:, index] = candidates[:, index].round()
    return candidates


def make_baseline_strategy(
    name: str,
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
) -> CandidateGenerator:
    """Create a named baseline using the existing benchmark runner contract.

    GP baselines are restricted to unconstrained, continuous,
    single-objective problems by the BoTorch strategy implementation.
    """
    strategies: dict[str, Callable[[], CandidateGenerator]] = {
        "botorch_qei": lambda: make_botorch_gp_strategy(
            acquisition="qEI", num_restarts=num_restarts, raw_samples=raw_samples
        ),
        "botorch_qnei": lambda: make_botorch_gp_strategy(
            acquisition="qNEI", num_restarts=num_restarts, raw_samples=raw_samples
        ),
        "random": lambda: random_candidates,
        "sobol": lambda: sobol_candidates,
    }
    try:
        factory = strategies[name]
    except KeyError as error:
        raise ValueError(f"Unknown benchmark baseline: {name}") from error
    return factory()


def list_baseline_strategies() -> tuple[str, ...]:
    """List supported baseline identifiers in stable alphabetical order."""
    return ("botorch_qei", "botorch_qnei", "random", "sobol")
