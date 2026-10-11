"""BoTorch-native qLogEI benchmark execution with isolated RNG streams."""

from __future__ import annotations

from robotorchan.benchmarks.botorch_strategy import make_botorch_gp_strategy
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems


def run_comparative_qlogei(
    cell: ComparativeExperimentCell,
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run qLogEI under the same runner and optimizer budget as qEI."""
    if cell.config.strategy != "qLogEI":
        raise ValueError("This runner requires a qLogEI comparison cell.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return run_benchmark(
        cell.config,
        make_botorch_gp_strategy(
            acquisition="qLogEI",
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        ),
        registry=registry,
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
