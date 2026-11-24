"""Static component filtering using existing registries."""

from dataclasses import dataclass
from enum import StrEnum

from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.capabilities import InputType, ObservationType
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.optim.capabilities import (
    BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
    MIXED_CMAES_OPTIMIZER_CAPABILITIES,
    MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
    OPTIMIZER_CAPABILITIES,
)
from robotorchan.workflow.problem import ObjectiveType, ObservationKind, OutputType, ProblemSpec
from robotorchan.workflow.selector import model_requirement_reasons


class SelectionStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ComponentSelection:
    name: str
    status: SelectionStatus
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentSelections:
    models: tuple[ComponentSelection, ...]
    acquisitions: tuple[ComponentSelection, ...]
    optimizers: tuple[ComponentSelection, ...]
    search_strategies: tuple[ComponentSelection, ...]


def _result(name: str, reasons: list[str], conditions: tuple[str, ...] = ()) -> ComponentSelection:
    if reasons:
        return ComponentSelection(name, SelectionStatus.INCOMPATIBLE, tuple(reasons))
    if conditions:
        return ComponentSelection(name, SelectionStatus.CONDITIONAL, conditions)
    return ComponentSelection(name, SelectionStatus.COMPATIBLE)


def _unknown(name: str) -> ComponentSelection:
    return ComponentSelection(name, SelectionStatus.UNKNOWN, ("capability metadata unavailable",))


def _model(spec: ProblemSpec, name: str) -> ComponentSelection:
    entry = MODEL_REGISTRY.get(name)
    kind = ObservationKind.REGRESSION
    if entry is not None:
        caps = entry.capabilities
    else:
        matches = [
            item
            for item in CLASSIFICATION_MODEL_REGISTRY.values()
            if name in (item.model_id, item.model_class.__name__)
        ]
        if not matches:
            return _unknown(name)
        item = matches[0]
        caps = item.capabilities
        kind = ObservationKind(item.model_id.split(".", 1)[0])
    reasons = list(model_requirement_reasons(spec, caps))
    if kind is not spec.observation_kind:
        reasons.append("observation kind does not match")
    return _result(name, reasons)


def _acquisition(spec: ProblemSpec, name: str) -> ComponentSelection:
    entry = ACQUISITION_REGISTRY.get(name)
    if entry is None:
        return _unknown(name)
    caps = entry.capabilities
    reasons = []
    observation = (
        ObservationType.REGRESSION
        if spec.observation_kind is ObservationKind.REGRESSION
        else ObservationType.CLASSIFICATION
    )
    if observation not in caps.observation_types:
        reasons.append("observation type does not match")
    if caps.purpose.value != spec.purpose.value:
        reasons.append("purpose does not match")
    if caps.max_q is not None and spec.q > caps.max_q:
        reasons.append("requested q exceeds acquisition limit")
    if spec.objective_type is ObjectiveType.MULTI and not caps.supports_multi_objective:
        reasons.append("requires multi-objective support")
    if spec.objective_type is ObjectiveType.SINGLE and caps.supports_multi_objective:
        reasons.append("requires single-objective acquisition")
    if spec.output_type is OutputType.MULTI and (
        not caps.supports_multi_output or caps.requires_single_output
    ):
        reasons.append("requires multi-output support")
    if spec.constrained and not caps.supports_constraints:
        reasons.append("requires outcome constraint support")
    if caps.requires_multi_fidelity and not spec.multi_fidelity:
        reasons.append("requires multi-fidelity structure")
    return _result(name, reasons, ("model posterior and constructor arguments must be checked",))


def _optimizer(spec: ProblemSpec, name: str) -> ComponentSelection:
    caps = OPTIMIZER_CAPABILITIES.get(name)
    if caps is None:
        return _unknown(name)
    conditions = ["acquisition semantics and runtime arguments must be checked"]
    if spec.input_type is InputType.MIXED:
        if name == "botorch":
            caps = BOTORCH_MIXED_OPTIMIZER_CAPABILITIES
            conditions.append("requires fixed_features_list")
        elif name == "ga":
            caps = MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES
        elif name == "cmaes":
            caps = MIXED_CMAES_OPTIMIZER_CAPABILITIES
    reasons = []
    if spec.input_type is InputType.MIXED and not caps.mixed:
        reasons.append("requires mixed-domain support")
    if spec.q > 1 and not caps.q_batch:
        reasons.append("requires q-batch support")
    if spec.candidate_constrained:
        if not any(
            (
                caps.linear_inequality_constraints,
                caps.linear_equality_constraints,
                caps.nonlinear_inequality_constraints,
                caps.interpoint_nonlinear_constraints,
            )
        ):
            reasons.append("requires candidate constraint support")
        conditions.append("specific constraint types and enforcement must be checked")
    return _result(name, reasons, tuple(conditions))


def _strategy(spec: ProblemSpec, name: str) -> ComponentSelection:
    if name == "MixedSpaceStrategy":
        if spec.input_type is not InputType.MIXED:
            return _result(name, ["requires mixed assignments"])
        return _result(name, [], ("requires fixed_features_list and acquisition checks",))
    if name in ("RandomSearchStrategy", "SobolSearchStrategy"):
        if spec.input_type is InputType.MIXED or spec.candidate_constrained:
            return _result(name, ["strategy does not accept mixed or constraint arguments"])
        return _result(name, [], ("acquisition semantics must be checked",))
    return _unknown(name)


def evaluate_component_inventory(
    spec: ProblemSpec,
    *,
    models: tuple[str, ...],
    acquisitions: tuple[str, ...],
    optimizers: tuple[str, ...],
    search_strategies: tuple[str, ...],
) -> ComponentSelections:
    """Evaluate every supplied factory name independently, without ranking."""
    return ComponentSelections(
        models=tuple(_model(spec, name) for name in models),
        acquisitions=tuple(_acquisition(spec, name) for name in acquisitions),
        optimizers=tuple(_optimizer(spec, name) for name in optimizers),
        search_strategies=tuple(_strategy(spec, name) for name in search_strategies),
    )
