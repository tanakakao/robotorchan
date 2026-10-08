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

__all__ = [
    "AcquisitionBenchmarkResult",
    "BenchmarkExperimentConfig",
    "BenchmarkProblem",
    "BenchmarkProblemRegistry",
    "BenchmarkResult",
    "CountingAcquisition",
    "ProfileResult",
    "benchmark_acquisition",
    "benchmark_optimizer",
    "benchmark_optimizers",
    "benchmark_result_record",
    "benchmark_result_records",
    "get_problem",
    "list_problems",
    "profile_callable",
    "register_problem",
]
