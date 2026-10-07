"""Composition contract for heterogeneous optimization semantics."""

from __future__ import annotations

from dataclasses import dataclass

from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.constraints import (
    ClassificationConstraint,
    ConstraintCollection,
    ContinuousConstraint,
    SemanticConstraint,
)
from robotorchan.semantics.feasibility import FeasibilityRepresentation
from robotorchan.semantics.objectives import ObjectiveCollection, SemanticObjective


@dataclass(frozen=True, slots=True)
class ProblemSemantics:
    """Declare objectives and constraints for one heterogeneous problem."""

    objectives: ObjectiveCollection
    constraints: ConstraintCollection

    def __init__(
        self,
        objectives: ObjectiveCollection | tuple[SemanticObjective, ...],
        constraints: ConstraintCollection | tuple[SemanticConstraint, ...] = (),
    ) -> None:
        """Initialize an immutable semantic problem definition."""
        if isinstance(objectives, ObjectiveCollection):
            objective_collection = objectives
        elif isinstance(objectives, tuple):
            objective_collection = ObjectiveCollection(*objectives)
        else:
            raise TypeError("objectives must be an ObjectiveCollection or tuple of objectives.")

        if isinstance(constraints, ConstraintCollection):
            constraint_collection = constraints
        elif isinstance(constraints, tuple):
            constraint_types = (ContinuousConstraint, ClassificationConstraint)
            if not all(isinstance(constraint, constraint_types) for constraint in constraints):
                raise TypeError(
                    "constraints accepts ContinuousConstraint or "
                    "ClassificationConstraint instances."
                )
            constraint_collection = ConstraintCollection(*constraints)
        else:
            raise TypeError("constraints must be a ConstraintCollection or tuple of constraints.")

        object.__setattr__(self, "objectives", objective_collection)
        object.__setattr__(self, "constraints", constraint_collection)

    def validate(self, model: HeterogeneousModel) -> None:
        """Validate every semantic reference against the heterogeneous model."""
        self.objectives.validate(model)
        self.constraints.validate(model)

    def resolve_objective_outputs(self, model: HeterogeneousModel) -> tuple[int, ...]:
        """Resolve objective outputs in declaration order."""
        return self.objectives.resolve_outputs(model)

    def resolve_constraint_outputs(self, model: HeterogeneousModel) -> tuple[int, ...]:
        """Resolve constraint outputs in declaration order."""
        return self.constraints.resolve_outputs(model)

    def resolve_constraint_owners(
        self,
        model: HeterogeneousModel,
    ) -> tuple[tuple[int, int], ...]:
        """Resolve constraints to entry and local output indices."""
        return self.constraints.resolve_owners(model)

    def feasibility_representations(
        self,
        model: HeterogeneousModel,
    ) -> tuple[FeasibilityRepresentation, ...]:
        """Build runtime feasibility representations in declaration order."""
        return self.constraints.feasibility_representations(model)
