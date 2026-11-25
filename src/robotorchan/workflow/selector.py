"""Capability-based model filtering for a declared problem."""

from dataclasses import dataclass

from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    ModelCapabilities,
    ModelRegistryEntry,
)
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.workflow.problem import ObservationKind, OutputType, ProblemSpec


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Compatibility facts for one registered model."""

    model_name: str
    compatible: bool
    reasons: tuple[str, ...]


def _check_model(spec: ProblemSpec, entry: ModelRegistryEntry) -> ModelSelectionResult:
    reasons = model_selection_reasons(spec, entry.capabilities, ObservationKind.REGRESSION)
    return ModelSelectionResult(entry.model_name, not reasons, reasons)


def model_requirement_reasons(
    spec: ProblemSpec, capabilities: ModelCapabilities
) -> tuple[str, ...]:
    """Check shared model structure independently of observation kind."""
    reasons: list[str] = []

    if capabilities.input_type is not spec.input_type:
        reasons.append(f"requires {spec.input_type.value} inputs")
    if capabilities.task_type is not spec.task_type:
        reasons.append(f"requires {spec.task_type.value} task structure")
    if spec.output_type is OutputType.MULTI and not capabilities.supports_multi_output:
        reasons.append("requires multi-output model support")
    if capabilities.multi_fidelity is not spec.multi_fidelity:
        reasons.append("multi-fidelity structure does not match the problem")
    if spec.structured_output and not capabilities.structured_output:
        reasons.append("requires structured-output support")
    high_dimensional = capabilities.high_dimensional is not HighDimensionalStrategy.NONE
    if spec.high_dimensional and not high_dimensional:
        reasons.append("requires a high-dimensional strategy")
    if spec.robust and not capabilities.robustness:
        reasons.append("requires an explicit robustness strategy")
    if capabilities.preference is not spec.preference:
        reasons.append("preference-observation structure does not match the problem")

    return tuple(reasons)


def model_selection_reasons(
    spec: ProblemSpec, capabilities: ModelCapabilities, kind: ObservationKind
) -> tuple[str, ...]:
    """Check model structure and the declared observation kind."""
    reasons = model_requirement_reasons(spec, capabilities)
    if kind is not spec.observation_kind:
        return (*reasons, "observation kind does not match")
    return reasons


def evaluate_models(
    spec: ProblemSpec,
    registry: dict[str, ModelRegistryEntry] | None = None,
) -> tuple[ModelSelectionResult, ...]:
    """Evaluate registered models without ranking compatible candidates."""
    if registry is None and spec.observation_kind is not ObservationKind.REGRESSION:
        results = []
        for entry in CLASSIFICATION_MODEL_REGISTRY.values():
            kind = ObservationKind(entry.model_id.split(".", 1)[0])
            reasons = model_selection_reasons(spec, entry.capabilities, kind)
            results.append(ModelSelectionResult(entry.model_id, not reasons, reasons))
        return tuple(results)
    source = MODEL_REGISTRY if registry is None else registry
    return tuple(_check_model(spec, entry) for entry in source.values())


def select_compatible_models(
    spec: ProblemSpec,
    registry: dict[str, ModelRegistryEntry] | None = None,
) -> tuple[str, ...]:
    """Return compatible model names in registry order, without ranking."""
    results = evaluate_models(spec, registry)
    return tuple(result.model_name for result in results if result.compatible)
