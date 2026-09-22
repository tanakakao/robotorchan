"""Capability-based model filtering for a declared problem."""

from dataclasses import dataclass

from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    ModelRegistryEntry,
)
from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.problem import ProblemSpec


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Compatibility facts for one registered model."""

    model_name: str
    compatible: bool
    reasons: tuple[str, ...]


def _check_model(spec: ProblemSpec, entry: ModelRegistryEntry) -> ModelSelectionResult:
    capabilities = entry.capabilities
    reasons: list[str] = []

    if capabilities.input_type is not spec.input_type:
        reasons.append(f"requires {spec.input_type.value} inputs")
    if capabilities.task_type is not spec.task_type:
        reasons.append(f"requires {spec.task_type.value} task structure")
    if spec.multi_fidelity and not capabilities.multi_fidelity:
        reasons.append("requires multi-fidelity support")
    if spec.structured_output and not capabilities.structured_output:
        reasons.append("requires structured-output support")
    high_dimensional = capabilities.high_dimensional is not HighDimensionalStrategy.NONE
    if spec.high_dimensional and not high_dimensional:
        reasons.append("requires a high-dimensional strategy")
    if spec.robust and not capabilities.robustness:
        reasons.append("requires an explicit robustness strategy")

    return ModelSelectionResult(entry.model_name, not reasons, tuple(reasons))


def evaluate_models(
    spec: ProblemSpec,
    registry: dict[str, ModelRegistryEntry] | None = None,
) -> tuple[ModelSelectionResult, ...]:
    """Evaluate registered models without ranking compatible candidates."""
    source = MODEL_REGISTRY if registry is None else registry
    return tuple(_check_model(spec, entry) for entry in source.values())


def select_compatible_models(
    spec: ProblemSpec,
    registry: dict[str, ModelRegistryEntry] | None = None,
) -> tuple[str, ...]:
    """Return compatible model names in registry order, without ranking."""
    results = evaluate_models(spec, registry)
    return tuple(result.model_name for result in results if result.compatible)
