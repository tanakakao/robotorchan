"""Tests for capability-engine benchmarks."""

from robotorchan.benchmarks.capability import (
    run_capability_benchmark,
    run_capability_benchmarks,
)
from robotorchan.models.capabilities import InputType, TaskType
from robotorchan.problem import ProblemPurpose, ProblemSpec


def test_benchmark_counts_are_internally_consistent() -> None:
    result = run_capability_benchmark(
        "continuous-bo",
        ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION),
    )

    assert result.registered_models > 0
    assert 0 < result.compatible_models <= result.registered_models
    assert result.recommendations > 0


def test_benchmark_covers_specialized_problem_shapes() -> None:
    cases = {
        "mixed-bo": ProblemSpec(
            purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
            input_type=InputType.MIXED,
        ),
        "multitask-al": ProblemSpec(
            purpose=ProblemPurpose.ACTIVE_LEARNING,
            task_type=TaskType.MULTITASK,
        ),
        "high-dimensional-bo": ProblemSpec(
            purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
            high_dimensional=True,
        ),
        "robust-bo": ProblemSpec(
            purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
            robust=True,
        ),
    }

    results = run_capability_benchmarks(cases)

    assert tuple(result.name for result in results) == tuple(cases)
    assert all(result.compatible_models > 0 for result in results)


def test_benchmark_excludes_undeclared_special_structures() -> None:
    result = run_capability_benchmark(
        "ordinary-bo",
        ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION),
    )

    assert result.compatible_models < result.registered_models
