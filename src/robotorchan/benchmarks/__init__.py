"""Reusable benchmark definitions and measurement utilities."""

from robotorchan.benchmarks.acquisition import (
    AcquisitionBenchmarkResult,
    benchmark_acquisition,
)
from robotorchan.benchmarks.active_learning_problems import (
    active_learning_labels,
    boundary_mae,
    brier_score,
    circle_boundary,
    circle_margin,
    classification_accuracy,
    register_active_learning_problems,
    wave_boundary,
    wave_margin,
)
from robotorchan.benchmarks.async_runner import (
    AsyncBenchmarkResult,
    AsyncEvaluation,
    run_async_benchmark,
)
from robotorchan.benchmarks.botorch_strategy import (
    botorch_gp_candidates,
    make_botorch_gp_strategy,
)
from robotorchan.benchmarks.comparison import (
    CurveSummary,
    PairedCurveComparison,
    compare_paired_curves,
    summarize_curves,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.constrained_problems import (
    constrained_annulus,
    constrained_disconnected,
    constrained_narrow_band,
    constrained_quadratic,
    register_constrained_problems,
)
from robotorchan.benchmarks.engineering_problems import (
    cantilever_beam,
    register_engineering_problems,
    thermal_management,
)
from robotorchan.benchmarks.heterogeneous_problems import (
    register_heterogeneous_problems,
    strength_conductivity_pass,
    strength_conductivity_pass_labels,
)
from robotorchan.benchmarks.high_dimensional_problems import (
    interaction_chain_30,
    register_high_dimensional_problems,
    rotated_subspace_40,
    sparse_sphere_50,
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
    mixed_category_interaction,
    mixed_process_yield,
    mixed_quadratic,
    register_mixed_problems,
)
from robotorchan.benchmarks.multifidelity_problems import (
    multifidelity_oscillatory,
    multifidelity_quadratic,
    register_multifidelity_problems,
)
from robotorchan.benchmarks.multiobjective_heterogeneous_problems import (
    multiobjective_pass_labels,
    multiobjective_two_labels,
    register_multiobjective_heterogeneous_problems,
    strength_conductivity_pass_tradeoff,
    strength_conductivity_two_pass,
)
from robotorchan.benchmarks.multiobjective_problems import (
    biobjective_linear,
    branin_currin,
    dtlz2,
    register_multiobjective_problems,
    zdt1,
)
from robotorchan.benchmarks.multiple_learned_constraints import (
    binary_constraint_labels,
    register_multiple_learned_constraint_problems,
    yield_binary_continuous_constraints,
    yield_two_binary_constraints,
)
from robotorchan.benchmarks.noisy_problems import (
    heteroscedastic_quadratic,
    noisy_quadratic,
    register_noisy_problems,
)
from robotorchan.benchmarks.optimization import (
    BenchmarkResult,
    CountingAcquisition,
    benchmark_optimizer,
    benchmark_optimizers,
    benchmark_result_record,
    benchmark_result_records,
)
from robotorchan.benchmarks.parallel_comparison import (
    ParallelBenchmarkComparison,
    run_parallel_comparison,
)
from robotorchan.benchmarks.persistence import (
    load_trajectory,
    save_trajectory,
    trajectory_from_record,
    trajectory_to_record,
)
from robotorchan.benchmarks.probability_objective_problems import (
    pass_probability_objective,
    register_probability_objective_problems,
    sample_pass_labels,
    yield_probability_tradeoff,
)
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.profiling import ProfileResult, profile_callable
from robotorchan.benchmarks.registry import (
    BenchmarkProblemRegistry,
    get_problem,
    list_problems,
    register_problem,
)
from robotorchan.benchmarks.regression_binary_problems import (
    register_regression_binary_problems,
    strength_pass,
    strength_pass_labels,
)
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
    sobol_initial_design,
)
from robotorchan.benchmarks.standard_problems import (
    ackley2,
    branin,
    hartmann6,
    register_standard_problems,
    rosenbrock2,
    sphere3,
)

__all__ = [
    "AcquisitionBenchmarkResult",
    "AsyncBenchmarkResult",
    "AsyncEvaluation",
    "BenchmarkExperimentConfig",
    "BenchmarkProblem",
    "BenchmarkProblemRegistry",
    "BenchmarkResult",
    "BenchmarkTrajectory",
    "CountingAcquisition",
    "CurveSummary",
    "PairedCurveComparison",
    "ParallelBenchmarkComparison",
    "ProfileResult",
    "ackley2",
    "active_learning_labels",
    "benchmark_acquisition",
    "benchmark_optimizer",
    "benchmark_optimizers",
    "benchmark_result_record",
    "benchmark_result_records",
    "binary_constraint_labels",
    "biobjective_linear",
    "botorch_gp_candidates",
    "boundary_mae",
    "branin",
    "branin_currin",
    "brier_score",
    "cantilever_beam",
    "categorical_switch",
    "circle_boundary",
    "circle_margin",
    "classification_accuracy",
    "compare_paired_curves",
    "constrained_annulus",
    "constrained_disconnected",
    "constrained_narrow_band",
    "constrained_quadratic",
    "cumulative_feasibility_rate",
    "dtlz2",
    "feasibility_rate",
    "get_problem",
    "hartmann6",
    "heteroscedastic_quadratic",
    "hypervolume_2d",
    "hypervolume_curve",
    "interaction_chain_30",
    "list_problems",
    "load_trajectory",
    "make_botorch_gp_strategy",
    "mixed_category_interaction",
    "mixed_process_yield",
    "mixed_quadratic",
    "multifidelity_oscillatory",
    "multifidelity_quadratic",
    "multiobjective_pass_labels",
    "multiobjective_two_labels",
    "noisy_quadratic",
    "pass_probability_objective",
    "profile_callable",
    "random_candidates",
    "register_active_learning_problems",
    "register_constrained_problems",
    "register_engineering_problems",
    "register_heterogeneous_problems",
    "register_high_dimensional_problems",
    "register_mixed_problems",
    "register_multifidelity_problems",
    "register_multiobjective_heterogeneous_problems",
    "register_multiobjective_problems",
    "register_multiple_learned_constraint_problems",
    "register_noisy_problems",
    "register_probability_objective_problems",
    "register_problem",
    "register_regression_binary_problems",
    "register_standard_problems",
    "rosenbrock2",
    "rotated_subspace_40",
    "run_async_benchmark",
    "run_benchmark",
    "run_parallel_comparison",
    "sample_pass_labels",
    "save_trajectory",
    "simple_regret_curve",
    "sobol_initial_design",
    "sparse_sphere_50",
    "sphere3",
    "strength_conductivity_pass",
    "strength_conductivity_pass_labels",
    "strength_conductivity_pass_tradeoff",
    "strength_conductivity_two_pass",
    "strength_pass",
    "strength_pass_labels",
    "summarize_curves",
    "thermal_management",
    "trajectory_from_record",
    "trajectory_to_record",
    "wave_boundary",
    "wave_margin",
    "yield_binary_continuous_constraints",
    "yield_probability_tradeoff",
    "yield_two_binary_constraints",
    "zdt1",
]
