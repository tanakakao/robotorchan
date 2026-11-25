"""Explainable recommendations built from capability compatibility facts."""

from dataclasses import dataclass

from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.compatibility import (
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.capabilities import InputType
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.workflow.components import SelectionStatus, evaluate_component_inventory
from robotorchan.workflow.problem import ObjectiveType, OutputType, ProblemPurpose, ProblemSpec
from robotorchan.workflow.selector import select_compatible_models


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A compatible model/acquisition combination with traceable rationale."""

    model_name: str
    acquisition_name: str | None
    rationale: tuple[str, ...]


def recommend_compatible_workflows(spec: ProblemSpec) -> tuple[Recommendation, ...]:
    """Return compatible workflows without scoring or ranking candidates."""
    models = select_compatible_models(spec)
    recommendations: list[Recommendation] = []
    for model_name in models:
        for acquisition_name, entry in ACQUISITION_REGISTRY.items():
            capabilities = entry.capabilities
            expected_purpose = (
                AcquisitionPurpose.BAYESIAN_OPTIMIZATION
                if spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION
                else AcquisitionPurpose.ACTIVE_LEARNING
            )
            if capabilities.purpose is not expected_purpose:
                continue
            if (
                spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION
                and spec.objective_type is ObjectiveType.MULTI
                and not capabilities.supports_multi_objective
            ):
                continue
            if (
                spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION
                and spec.objective_type is ObjectiveType.SINGLE
                and capabilities.supports_multi_objective
            ):
                continue
            if spec.constrained and not capabilities.supports_constraints:
                continue
            if capabilities.requires_multi_fidelity and not spec.multi_fidelity:
                continue
            if capabilities.max_q is not None and spec.q > capabilities.max_q:
                continue
            if spec.output_type is OutputType.MULTI and not capabilities.supports_multi_output:
                continue
            if spec.output_type is OutputType.MULTI and capabilities.requires_single_output:
                continue
            result = check_model_acquisition_compatibility(model_name, acquisition_name)
            if not result.compatible:
                continue
            recommendations.append(
                Recommendation(
                    model_name=model_name,
                    acquisition_name=acquisition_name,
                    rationale=(
                        "model satisfies the declared problem capabilities",
                        "acquisition is statically compatible with the model",
                    ),
                )
            )
    return tuple(recommendations)


@dataclass(frozen=True, slots=True)
class WorkflowCompatibility:
    """Static decision for a named workflow, not a runtime validation."""

    status: SelectionStatus
    reasons: tuple[str, ...]
    conditions: tuple[str, ...]


def check_compatibility(
    spec: ProblemSpec,
    *,
    model: str,
    acquisition: str,
    optimizer: str | None = None,
    search_strategy: str | None = None,
) -> WorkflowCompatibility:
    """Compose existing capability rules without constructing or fitting objects."""
    inventory = evaluate_component_inventory(
        spec,
        models=(model,),
        acquisitions=(acquisition,),
        optimizers=(optimizer,) if optimizer is not None else (),
        search_strategies=(search_strategy,) if search_strategy is not None else (),
    )
    reasons: list[str] = []
    conditions: list[str] = []
    unknown = False
    for group in (
        inventory.models,
        inventory.acquisitions,
        inventory.optimizers,
        inventory.search_strategies,
    ):
        for item in group:
            messages = [f"{item.name}: {reason}" for reason in item.reasons]
            if item.status is SelectionStatus.INCOMPATIBLE:
                reasons.extend(messages)
            elif item.status is SelectionStatus.UNKNOWN:
                unknown = True
                conditions.extend(messages)
            else:
                conditions.extend(messages)
    model_id = model
    if model not in MODEL_REGISTRY and model not in CLASSIFICATION_MODEL_REGISTRY:
        model_id = next(
            (
                key
                for key, entry in CLASSIFICATION_MODEL_REGISTRY.items()
                if entry.model_class.__name__ == model
            ),
            model,
        )
    registered_model = model_id in MODEL_REGISTRY or model_id in CLASSIFICATION_MODEL_REGISTRY
    entry = ACQUISITION_REGISTRY.get(acquisition)
    if registered_model and entry is not None:
        pair = check_model_acquisition_compatibility(model_id, acquisition)
        reasons.extend(pair.reasons)
    else:
        conditions.append("model/acquisition posterior contract requires explicit validation")
    if optimizer is not None and search_strategy is not None:
        conditions.append("strategy must explicitly accept and use the specified optimizer")
    if optimizer in ("optimize_vector_nsga2", "optimize_vector_nsga2_mixed"):
        reasons.append("vector optimizer does not consume a scalar acquisition")
    if entry is not None:
        if optimizer == "optimize_mixed_one_shot_acqf":
            backend = CandidateOptimizer.MIXED_ONE_SHOT
        elif spec.input_type is InputType.MIXED:
            backend = CandidateOptimizer.MIXED
        else:
            backend = CandidateOptimizer.CONTINUOUS
        if optimizer is not None:
            pair_optimizer = check_acquisition_optimizer_compatibility(
                acquisition, backend, q=spec.q
            )
            reasons.extend(pair_optimizer.reasons)
        if entry.capabilities.one_shot and optimizer not in (
            None,
            "botorch",
            "optimize_acqf_botorch",
            "optimize_mixed_one_shot_acqf",
        ):
            reasons.append("optimizer has no verified one-shot augmented-batch path")
        if entry.capabilities.one_shot and search_strategy not in (
            None,
            "OriginalSpaceStrategy",
        ):
            reasons.append("strategy has no verified one-shot augmented-batch path")
    if reasons:
        status = SelectionStatus.INCOMPATIBLE
    elif unknown:
        status = SelectionStatus.UNKNOWN
    elif conditions:
        status = SelectionStatus.CONDITIONAL
    else:
        status = SelectionStatus.COMPATIBLE
    return WorkflowCompatibility(status, tuple(reasons), tuple(conditions))


@dataclass(frozen=True, slots=True)
class WorkflowRecommendation:
    model_name: str
    acquisition_name: str
    optimizer_name: str
    search_strategy_name: str | None
    compatibility: WorkflowCompatibility
    rationale: tuple[str, ...]


def recommend(
    spec: ProblemSpec,
    *,
    model: str | None = None,
    acquisition: str | None = None,
    optimizer: str = "botorch",
    search_strategy: str | None = None,
) -> tuple[WorkflowRecommendation, ...]:
    """Enumerate unranked registered pairs, preserving explicit component choices."""
    candidates = recommend_compatible_workflows(spec)
    if model is not None and acquisition is not None:
        pairs = ((model, acquisition),)
    else:
        pairs = tuple(
            (item.model_name, item.acquisition_name)
            for item in candidates
            if item.acquisition_name is not None
            and (model is None or item.model_name == model)
            and (acquisition is None or item.acquisition_name == acquisition)
        )
    results = []
    for model_name, acquisition_name in pairs:
        decision = check_compatibility(
            spec,
            model=model_name,
            acquisition=acquisition_name,
            optimizer=optimizer,
            search_strategy=search_strategy,
        )
        if decision.status not in (SelectionStatus.COMPATIBLE, SelectionStatus.CONDITIONAL):
            continue
        results.append(
            WorkflowRecommendation(
                model_name,
                acquisition_name,
                optimizer,
                search_strategy,
                decision,
                ("static capability match; no empirical performance ranking",),
            )
        )
    return tuple(results)
