"""Reusable benchmark definitions and measurement utilities."""

from robotorchan.benchmarks.acquisition import (
    AcquisitionBenchmarkResult,
    benchmark_acquisition,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.optimization import (
    BenchmarkResult,
    CountingAcquisition,
    benchmark_optimizer,
    benchmark_optimizers,
    benchmark_result_record,
    benchmark_result_records,
)
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.profiling import ProfileResult, profile_callable
from robotorchan.benchmarks.registry import (
    BenchmarkProblemRegistry,
    get_problem,
    list_problems,
    register_problem,
)
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
    sobol_initial_design,
)
from robotorchan.benchmarks.standard_problems import (
    branin,
    hartmann6,
    register_standard_problems,
    sphere3,
)

__all__ = [
    "AcquisitionBenchmarkResult",
    "BenchmarkExperimentConfig",
    "BenchmarkProblem",
    "BenchmarkProblemRegistry",
    "BenchmarkResult",
    "BenchmarkTrajectory",
    "CountingAcquisition",
    "ProfileResult",
    "benchmark_acquisition",
    "benchmark_optimizer",
    "benchmark_optimizers",
    "benchmark_result_record",
    "benchmark_result_records",
    "branin",
    "get_problem",
    "hartmann6",
    "list_problems",
    "profile_callable",
    "random_candidates",
    "register_problem",
    "register_standard_problems",
    "run_benchmark",
    "sobol_initial_design",
    "sphere3",
]
