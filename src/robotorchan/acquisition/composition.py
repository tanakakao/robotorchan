"""Resolved acquisition-composition plans for heterogeneous problem semantics."""

from __future__ import annotations

from dataclasses import dataclass

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model as BoTorchModel

from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.feasibility import FeasibilityRepresentation, ProbabilityOfFeasibility
from robotorchan.semantics.objectives import RegressionObjective, SemanticObjective
from robotorchan.semantics.probability import ClassificationProbabilityOfFeasibility
from robotorchan.semantics.problem import ProblemSemantics


@dataclass(frozen=True, slots=True)
class ObjectiveBinding:
    """Bind one semantic objective to its owning heterogeneous model entry."""

    objective: SemanticObjective
    output_index: int
    entry_index: int
    local_output_index: int


@dataclass(frozen=True, slots=True)
class FeasibilityBinding:
    """Bind one feasibility representation to its owning model entry."""

    representation: FeasibilityRepresentation
    output_index: int
    entry_index: int
    local_output_index: int


@dataclass(frozen=True, slots=True)
class AcquisitionCompositionPlan:
    """Resolved semantic inputs for later BoTorch acquisition construction.

    The plan records ownership without constructing a shared posterior, sampler,
    objective, constraint, or acquisition function. Later composition policies
    decide how compatible bindings are mapped to native BoTorch interfaces.
    """

    objectives: tuple[ObjectiveBinding, ...]
    feasibility: tuple[FeasibilityBinding, ...]


def resolve_acquisition_composition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
) -> AcquisitionCompositionPlan:
    """Resolve heterogeneous problem semantics into acquisition-layer bindings."""
    semantics.validate(model)

    objective_bindings = []
    for objective, output_index in zip(
        semantics.objectives,
        semantics.resolve_objective_outputs(model),
        strict=True,
    ):
        entry_index, local_output_index = model.output_owner(output_index)
        objective_bindings.append(
            ObjectiveBinding(
                objective=objective,
                output_index=output_index,
                entry_index=entry_index,
                local_output_index=local_output_index,
            )
        )

    feasibility_bindings = []
    for representation, output_index in zip(
        semantics.feasibility_representations(model),
        semantics.resolve_constraint_outputs(model),
        strict=True,
    ):
        entry_index, local_output_index = model.output_owner(output_index)
        feasibility_bindings.append(
            FeasibilityBinding(
                representation=representation,
                output_index=output_index,
                entry_index=entry_index,
                local_output_index=local_output_index,
            )
        )

    return AcquisitionCompositionPlan(
        objectives=tuple(objective_bindings),
        feasibility=tuple(feasibility_bindings),
    )


@dataclass(frozen=True, slots=True)
class BoTorchObjectiveBridge:
    """Native BoTorch model and MC objective for one regression binding."""

    model: BoTorchModel
    objective: MCAcquisitionObjective


def make_botorch_objective_bridge(
    model: HeterogeneousModel,
    binding: ObjectiveBinding,
) -> BoTorchObjectiveBridge:
    """Adapt one regression objective binding to native BoTorch interfaces."""
    objective = binding.objective
    if not isinstance(objective, RegressionObjective):
        raise TypeError("BoTorch objective bridge currently supports RegressionObjective only.")

    output_index = objective.resolve_output(model)
    entry_index, local_output_index = model.output_owner(output_index)
    if (
        output_index != binding.output_index
        or entry_index != binding.entry_index
        or local_output_index != binding.local_output_index
    ):
        raise ValueError("ObjectiveBinding does not match the current heterogeneous model.")

    entry_model = model[entry_index]
    if not isinstance(entry_model, BoTorchModel):
        raise TypeError("Regression objective owner must be a BoTorch Model.")

    return BoTorchObjectiveBridge(
        model=entry_model,
        objective=objective.to_botorch(model),
    )


def make_deterministic_pof_acquisition(
    objective_acquisition: AcquisitionFunction,
    binding: FeasibilityBinding,
    *,
    q_reduction: str = "product",
) -> FeasibilityWeightedAcquisition:
    """Weight an acquisition by one deterministic posterior-predictive PoF."""
    representation = binding.representation
    if not isinstance(representation, ProbabilityOfFeasibility):
        raise TypeError(
            "Deterministic PoF weighting requires a ProbabilityOfFeasibility representation."
        )
    return FeasibilityWeightedAcquisition(
        objective_acquisition,
        representation.probability,
        q_reduction=q_reduction,
    )


def make_classification_feasibility_bridge(
    model: HeterogeneousModel,
    binding: FeasibilityBinding,
) -> ClassificationProbabilityOfFeasibility:
    """Expose one classification feasibility binding as posterior-predictive PoF."""
    representation = binding.representation
    if not isinstance(representation, ProbabilityOfFeasibility):
        raise TypeError(
            "Classification feasibility bridge requires a ProbabilityOfFeasibility representation."
        )

    probability = representation.probability
    entry_model = model[binding.entry_index]
    if probability.model is not entry_model:
        raise ValueError("FeasibilityBinding does not match the current heterogeneous model.")

    output_index = model.resolve_output(binding.output_index)
    entry_index, local_output_index = model.output_owner(output_index)
    if (
        entry_index != binding.entry_index
        or local_output_index != binding.local_output_index
    ):
        raise ValueError("FeasibilityBinding does not match the current heterogeneous model.")

    return probability
