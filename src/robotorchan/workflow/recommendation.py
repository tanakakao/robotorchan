"""Explainable recommendations built from capability compatibility facts."""

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature

from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.compatibility import (
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.factory import get_acquisition_class, list_acquisitions
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.capabilities import InputType
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES
from robotorchan.optim.factory import create_optimizer, list_optimizers
from robotorchan.optim.search_factory import list_search_strategies, resolve_search_strategy
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
            result = check_model_acquisition_compatibility(
                model_name, acquisition_name, multi_output=spec.output_type is OutputType.MULTI
            )
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


def _model_registry_id(model: str) -> str:
    if model in MODEL_REGISTRY or model in CLASSIFICATION_MODEL_REGISTRY:
        return model
    return next(
        (
            key
            for key, entry in CLASSIFICATION_MODEL_REGISTRY.items()
            if entry.model_class.__name__ == model
        ),
        model,
    )


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
    if spec.decoupled:
        masked = {
            "qHypervolumeKnowledgeGradient",
            "qMultiFidelityHypervolumeKnowledgeGradient",
            "qDecoupledKnowledgeGradient",
            "qDecoupledExpectedPredictiveInformationGain",
            "qDecoupledActiveLearning",
            "qDecoupledParticleKnowledgeGradient",
        }
        pending = {
            "qHypervolumeKnowledgeGradient",
            "qMultiFidelityHypervolumeKnowledgeGradient",
            "qDecoupledExpectedPredictiveInformationGain",
        }
        if acquisition not in masked:
            reasons.append("requires a verified masked acquisition")
        if spec.masked_pending and acquisition not in pending:
            reasons.append("requires masked pending evaluation support")
        if optimizer != "optimize_acqf_decoupled" or search_strategy is not None:
            reasons.append("requires the dedicated decoupled optimizer without a search strategy")
        conditions.append("validate output-specific training data and evaluation masks")
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
    model_id = _model_registry_id(model)
    registered_model = model_id in MODEL_REGISTRY or model_id in CLASSIFICATION_MODEL_REGISTRY
    entry = ACQUISITION_REGISTRY.get(acquisition)
    if registered_model and entry is not None:
        pair = check_model_acquisition_compatibility(
            model_id, acquisition, multi_output=spec.output_type is OutputType.MULTI
        )
        reasons.extend(pair.reasons)
    else:
        conditions.append("model/acquisition posterior contract requires explicit validation")
    if optimizer is not None and search_strategy is not None:
        if search_strategy == "TuRBOStrategy":
            if optimizer not in OPTIMIZER_CAPABILITIES:
                reasons.append("TuRBO requires a scalar dispatcher optimizer name")
            elif spec.input_type is InputType.MIXED:
                reasons.append("TuRBO mixed path owns its optimizer selection")
            else:
                conditions.append("pass optimizer to the TuRBO constructor")
        elif (
            search_strategy == "OriginalSpaceStrategy"
            and optimizer in ("botorch", "optimize_acqf_botorch")
        ) or (
            search_strategy == "MixedSpaceStrategy"
            and optimizer in ("botorch", "optimize_acqf_mixed_botorch")
        ):
            conditions.append("strategy owns the matching BoTorch optimization path")
        else:
            reasons.append("strategy does not expose the specified optimizer selection")
    if model in ("GPStructuralCausalModel", "StructuralCausalModel", "InterventionGPSurrogate"):
        reasons.append("use the dedicated causal workflow or extract its native BoTorch model")
    if (
        optimizer
        in ("optimize_acqf_mixed_botorch", "optimize_acqf_mixed_ga", "optimize_acqf_mixed_cmaes")
        and spec.input_type is not InputType.MIXED
    ):
        reasons.append("dedicated mixed optimizer requires a mixed domain")
    if optimizer == "optimize_acqf_botorch" and spec.input_type is InputType.MIXED:
        reasons.append("continuous BoTorch backend does not enumerate mixed assignments")
    if optimizer in ("optimize_vector_nsga2", "optimize_vector_nsga2_mixed"):
        reasons.append("vector optimizer does not consume a scalar acquisition")
    if entry is None and optimizer == "optimize_mixed_one_shot_acqf":
        conditions.append("wrapped acquisition must implement the one-shot contract")
    if entry is not None:
        if optimizer == "optimize_mixed_one_shot_acqf":
            backend = CandidateOptimizer.MIXED_ONE_SHOT
        elif optimizer in ("random", "sobol", "de", "ga", "cmaes", "pso"):
            backend = CandidateOptimizer.DERIVATIVE_FREE
        elif optimizer == "optimize_acqf_decoupled" and entry.capabilities.requires_derivative_free:
            backend = CandidateOptimizer.DERIVATIVE_FREE
            conditions.append("select a derivative-free backend in the decoupled optimizer")
        elif (
            optimizer == "optimize_acqf_decoupled"
            and spec.input_type is InputType.MIXED
            and entry.capabilities.one_shot
        ):
            backend = CandidateOptimizer.MIXED_ONE_SHOT
            conditions.append("requires supported mixed one-shot assignments")
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
            "optimize_acqf_decoupled",
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
    optimizer_name: str | None
    search_strategy_name: str | None
    compatibility: WorkflowCompatibility
    rationale: tuple[str, ...]

    def required_arguments(self) -> dict[str, tuple[str, ...] | None]:
        """Inspect original signatures; None means introspection is unavailable."""
        models = import_module("robotorchan.models")
        result = {
            "model": _required_parameters(models.get_model_class(self.model_name)),
            "acquisition": _required_parameters(get_acquisition_class(self.acquisition_name)),
        }
        if self.optimizer_name is not None:
            result["optimizer"] = _required_parameters(create_optimizer(self.optimizer_name))
        if self.search_strategy_name is not None:
            result["search_strategy"] = _required_parameters(
                resolve_search_strategy(self.search_strategy_name)
            )
        return result


def _required_parameters(constructor: object) -> tuple[str, ...] | None:
    try:
        parameters = signature(constructor).parameters.values()
    except (TypeError, ValueError):
        return None
    return tuple(
        parameter.name
        for parameter in parameters
        if parameter.default is Parameter.empty
        and parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    )


def recommend(
    spec: ProblemSpec,
    *,
    model: str | None = None,
    acquisition: str | None = None,
    optimizer: str | None = None,
    search_strategy: str | None = None,
) -> tuple[WorkflowRecommendation, ...]:
    """Enumerate unranked candidates; use auto to enumerate optimizer or strategy names."""
    if optimizer is None and search_strategy is None:
        optimizer = "optimize_acqf_decoupled" if spec.decoupled else "botorch"
    if model == "auto":
        models = import_module("robotorchan.models")
        model_names = models.list_models()
    else:
        model_names = (model,) if model is not None else select_compatible_models(spec)
    if acquisition == "auto":
        acquisition_names = list_acquisitions()
    elif acquisition is None:
        acquisition_names = tuple(ACQUISITION_REGISTRY)
    else:
        acquisition_names = (acquisition,)
    inventory = evaluate_component_inventory(
        spec,
        models=model_names,
        acquisitions=acquisition_names,
        optimizers=(),
        search_strategies=(),
    )
    model_names = tuple(
        item.name
        for item in inventory.models
        if item.status in (SelectionStatus.COMPATIBLE, SelectionStatus.CONDITIONAL)
    )
    acquisition_names = tuple(
        item.name
        for item in inventory.acquisitions
        if item.status in (SelectionStatus.COMPATIBLE, SelectionStatus.CONDITIONAL)
    )
    pairs = (
        (model_name, acquisition_name)
        for model_name in model_names
        for acquisition_name in acquisition_names
    )
    optimizer_names = list_optimizers() if optimizer == "auto" else (optimizer,)
    strategy_names = list_search_strategies() if search_strategy == "auto" else (search_strategy,)
    results = []
    for model_name, acquisition_name in pairs:
        for optimizer_name in optimizer_names:
            for strategy_name in strategy_names:
                decision = check_compatibility(
                    spec,
                    model=model_name,
                    acquisition=acquisition_name,
                    optimizer=optimizer_name,
                    search_strategy=strategy_name,
                )
                if decision.status in (SelectionStatus.INCOMPATIBLE, SelectionStatus.UNKNOWN):
                    continue
                results.append(
                    WorkflowRecommendation(
                        model_name,
                        acquisition_name,
                        optimizer_name,
                        strategy_name,
                        decision,
                        ("static capability match; no empirical performance ranking",),
                    )
                )
    return tuple(results)
