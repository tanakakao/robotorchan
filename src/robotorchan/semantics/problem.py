"""Composition contract for heterogeneous optimization semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.constraints import (
    ClassificationConstraint,
    ContinuousConstraint,
)
from robotorchan.semantics.feasibility import FeasibilityRepresentation
from robotorchan.semantics.objectives import ObjectiveCollection, SemanticObjective

SemanticConstraint: TypeAlias = ContinuousConstraint | ClassificationConstraint


@dataclass(frozen=True, slots=True)
class ProblemSemantics:
    """Declare objectives and constraints for one heterogeneous problem."""

    objectives: ObjectiveCollection
    constraints: tuple[SemanticConstraint, ...]

    def __init__(
        self,
        objectives: ObjectiveCollection | tuple[SemanticObjective, ...],
        constraints: tuple[SemanticConstraint, ...] = (),
    ) -> None:
        """Initialize an immutable semantic problem definition."""
        if isinstance(objectives, ObjectiveCollection):
            objective_collection = objectives
        elif isinstance(objectives, tuple):
            objective_collection = ObjectiveCollection(*objectives)
        else:
            raise TypeError("objectives must be an ObjectiveCollection or tuple of objectives.")

        constraint_types = (ContinuousConstraint, ClassificationConstraint)
        if not isinstance(constraints, tuple):
            raise TypeError("constraints must be a tuple of semantic constraints.")
        if not all(isinstance(constraint, constraint_types) for constraint in constraints):
            raise TypeError(
                "constraints accepts ContinuousConstraint or ClassificationConstraint instances."
            )

        object.__setattr__(self, "objectives", objective_collection)
        object.__setattr__(self, "constraints", constraints)

    def validate(self, model: HeterogeneousModel) -> None:
        """Validate every semantic reference against the heterogeneous model."""
        self.objectives.validate(model)
        for constraint in self.constraints:
            constraint.resolve_output(model)

    def resolve_objective_outputs(self, model: HeterogeneousModel) -> tuple[int, ...]:
        """Resolve objective outputs in declaration order."""
        return self.objectives.resolve_outputs(model)

    def resolve_constraint_outputs(self, model: HeterogeneousModel) -> tuple[int, ...]:
        """Resolve constraint outputs in declaration order."""
        return tuple(constraint.resolve_output(model) for constraint in self.constraints)

    def resolve_constraint_owners(
        self,
        model: HeterogeneousModel,
    ) -> tuple[tuple[int, int], ...]:
        """Resolve constraints to entry and local output indices."""
        return tuple(constraint.resolve_owner(model) for constraint in self.constraints)

    def feasibility_representations(
        self,
        model: HeterogeneousModel,
    ) -> tuple[FeasibilityRepresentation, ...]:
        """Build runtime feasibility representations in declaration order."""
        return tuple(
            constraint.to_feasibility_representation(model)
            for constraint in self.constraints
        )
