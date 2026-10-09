"""Reusable benchmark definitions and measurement utilities."""

from robotorchan.benchmarks.acquisition import (
    AcquisitionBenchmarkResult,
    benchmark_acquisition,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.constrained_problems import (
    constrained_annulus,
    constrained_quadratic,
    register_constrained_problems,
)
from robotorchan.benchmarks.heterogeneous_problems import (
    register_heterogeneous_problems,
    strength_conductivity_pass,
    strength_conductivity_pass_labels,
)
from robotorchan.benchmarks.metrics import (
    cumulative_feasibility_rate,
    feasibility_rate,
    hypervolume_2d,
    hypervolume_curve,
    simple_regret_curve,
)
from robotorchan.benchmarks.mixed_problems import (
    categorical_switch,
    mixed_quadratic,
    register_mixed_problems,
)
from robotorchan.benchmarks.multiobjective_problems import (
    biobjective_linear,
    dtlz2,
    register_multiobjective_problems,
    zdt1,
)
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
    "biobjective_linear",
    "branin",
    "categorical_switch",
    "constrained_annulus",
    "constrained_quadratic",
    "cumulative_feasibility_rate",
    "dtlz2",
    "feasibility_rate",
    "get_problem",
    "hartmann6",
    "hypervolume_2d",
    "hypervolume_curve",
    "list_problems",
    "mixed_quadratic",
    "profile_callable",
    "random_candidates",
    "register_constrained_problems",
    "register_heterogeneous_problems",
    "register_mixed_problems",
    "register_multiobjective_problems",
    "register_problem",
    "register_standard_problems",
    "run_benchmark",
    "simple_regret_curve",
    "sobol_initial_design",
    "sphere3",
    "strength_conductivity_pass",
    "strength_conductivity_pass_labels",
    "zdt1",
]
