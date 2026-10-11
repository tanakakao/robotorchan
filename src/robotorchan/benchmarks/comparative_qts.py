"""qTS comparative runner under the existing paired empirical protocol."""

from __future__ import annotations

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.thompson_strategy import thompson_candidates


def run_comparative_qts(
    cell: ComparativeExperimentCell,
    *,
    pool_size: int = 256,
) -> tuple[BenchmarkTrajectory, ...]:
    """Run qTS with paired designs, restricted truth and isolated strategy RNG."""
    if cell.config.strategy != "qTS":
        raise ValueError("This runner requires a qTS comparison cell.")
    if type(pool_size) is not int or pool_size < 2:
        raise ValueError("pool_size must be an integer >= 2.")

    def propose(problem, X, Y, q, generator):
        return thompson_candidates(problem, X, Y, q, generator, pool_size=pool_size)

    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    return run_benchmark(
        cell.config,
        propose,
        registry=registry,
        restrict_candidate_problem=True,
        independent_rng_streams=True,
        isolate_strategy_rng=True,
    )
