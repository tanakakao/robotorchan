"""Benchmarks for capability-engine coverage and recommendation consistency."""

from dataclasses import dataclass

from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.workflow.problem import ProblemSpec
from robotorchan.workflow.recommendation import recommend_compatible_workflows
from robotorchan.workflow.selector import evaluate_models


@dataclass(frozen=True, slots=True)
class CapabilityBenchmarkResult:
    """Summary of one declarative capability benchmark."""

    name: str
    registered_models: int
    compatible_models: int
    compatible_acquisitions: int
    recommendations: int


def run_capability_benchmark(
    name: str,
    spec: ProblemSpec,
) -> CapabilityBenchmarkResult:
    """Measure registry filtering and recommendation coverage for a problem."""
    evaluations = evaluate_models(spec)
    compatible = sum(result.compatible for result in evaluations)
    recommendations = recommend_compatible_workflows(spec)
    compatible_acquisitions = {
        item.acquisition_name for item in recommendations if item.acquisition_name is not None
    }
    return CapabilityBenchmarkResult(
        name=name,
        registered_models=len(MODEL_REGISTRY),
        compatible_models=compatible,
        compatible_acquisitions=len(compatible_acquisitions),
        recommendations=len(recommendations),
    )


def run_capability_benchmarks(
    cases: dict[str, ProblemSpec],
) -> tuple[CapabilityBenchmarkResult, ...]:
    """Run deterministic capability benchmarks in declared case order."""
    return tuple(run_capability_benchmark(name, spec) for name, spec in cases.items())
