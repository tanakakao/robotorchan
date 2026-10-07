"""Resolved acquisition-composition plans for heterogeneous problem semantics."""

from __future__ import annotations

from dataclasses import dataclass

from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.feasibility import FeasibilityRepresentation
from robotorchan.semantics.objectives import SemanticObjective
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
