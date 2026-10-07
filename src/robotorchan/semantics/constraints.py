"""Semantic specifications for outcome constraints."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

from torch import Tensor

from robotorchan.semantics.probability import ClassificationProbabilityOfFeasibility
from robotorchan.models.capabilities import ObservationType
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.classes import resolve_class
from robotorchan.semantics.direction import ConstraintDirection


@dataclass(frozen=True, slots=True)
class ContinuousConstraint:
    """Declare a one-sided constraint on one regression output."""

    output: int | str
    threshold: float
    direction: ConstraintDirection = ConstraintDirection.LESS_THAN_OR_EQUAL

    def __post_init__(self) -> None:
        """Validate scalar threshold and direction semantics."""
        if not isinstance(self.threshold, Real) or isinstance(self.threshold, bool):
            raise TypeError("threshold must be a real scalar.")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite.")
        if not isinstance(self.direction, ConstraintDirection):
            raise TypeError("direction must be a ConstraintDirection.")

    def resolve_output(self, model: HeterogeneousModel) -> int:
        """Resolve and validate the referenced regression output."""
        output_index = model.resolve_output(self.output)
        if model.output_observation_type(output_index) is not ObservationType.REGRESSION:
            raise TypeError("ContinuousConstraint requires a regression output.")
        return output_index

    def resolve_owner(self, model: HeterogeneousModel) -> tuple[int, int]:
        """Resolve the referenced output to its entry and local output index."""
        return model.output_owner(self.resolve_output(model))

    def to_botorch(self, model: HeterogeneousModel):
        """Create a BoTorch outcome-constraint callable for entry samples."""
        _, local_output_index = self.resolve_owner(model)

        def constraint(samples: Tensor) -> Tensor:
            values = samples[..., local_output_index]
            return self.direction.residual(values, self.threshold)

        return constraint

    def to_feasibility_representation(self, model: HeterogeneousModel):
        """Represent this constraint as a sample-space residual."""
        from robotorchan.semantics.feasibility import SampleResidualFeasibility

        return SampleResidualFeasibility(self.to_botorch(model))


@dataclass(frozen=True, slots=True)
class ClassificationConstraint:
    """Declare classifier membership in one class as feasibility."""

    output: int | str
    feasible_class: int | object = 1
    probability_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate optional posterior-predictive probability threshold."""
        if self.probability_threshold is None:
            return
        threshold = self.probability_threshold
        if not isinstance(threshold, Real) or isinstance(threshold, bool):
            raise TypeError("probability_threshold must be a real scalar.")
        if not math.isfinite(float(threshold)):
            raise ValueError("probability_threshold must be finite.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("probability_threshold must be between 0 and 1.")

    def resolve_output(self, model: HeterogeneousModel) -> int:
        """Resolve and validate the referenced classification output."""
        output_index = model.resolve_output(self.output)
        if model.output_observation_type(output_index) is not ObservationType.CLASSIFICATION:
            raise TypeError("ClassificationConstraint requires a classification output.")
        metadata = model.output_classification_metadata(output_index)
        if metadata is None:
            raise TypeError("ClassificationConstraint requires classification metadata.")
        self.resolve_feasible_class(model)
        return output_index

    def resolve_feasible_class(self, model: HeterogeneousModel) -> int:
        """Resolve a class index or label to the canonical probability index."""
        output_index = model.resolve_output(self.output)
        metadata = model.output_classification_metadata(output_index)
        if metadata is None:
            raise TypeError("ClassificationConstraint requires classification metadata.")
        return resolve_class(
            self.feasible_class,
            metadata,
            argument_name="feasible_class",
        )

    def resolve_owner(self, model: HeterogeneousModel) -> tuple[int, int]:
        """Resolve the referenced output to its entry and local output index."""
        return model.output_owner(self.resolve_output(model))

    def to_probability_of_feasibility(
        self,
        model: HeterogeneousModel,
    ) -> ClassificationProbabilityOfFeasibility:
        """Create the existing classifier-backed probability-of-feasibility adapter."""
        entry_index, _ = self.resolve_owner(model)
        return ClassificationProbabilityOfFeasibility(
            model[entry_index],
            feasible_class=self.resolve_feasible_class(model),
        )

    def to_probability_constraint(
        self,
        model: HeterogeneousModel,
    ) -> ClassificationProbabilityConstraint:
        """Create a thresholded posterior-predictive probability constraint."""
        if self.probability_threshold is None:
            raise ValueError("probability_threshold is required for a probability constraint.")
        return ClassificationProbabilityConstraint(
            self.to_probability_of_feasibility(model),
            self.probability_threshold,
        )

    def to_feasibility_representation(self, model: HeterogeneousModel):
        """Represent class feasibility without erasing its probability semantics."""
        from robotorchan.semantics.feasibility import (
            ProbabilityOfFeasibility,
            ProbabilityResidualFeasibility,
        )

        if self.probability_threshold is None:
            return ProbabilityOfFeasibility(self.to_probability_of_feasibility(model))
        return ProbabilityResidualFeasibility(self.to_probability_constraint(model))


class ClassificationProbabilityConstraint:
    """Evaluate a thresholded classifier probability with <= 0 feasibility."""

    def __init__(
        self,
        probability_of_feasibility: ClassificationProbabilityOfFeasibility,
        threshold: float,
    ) -> None:
        self.probability_of_feasibility = probability_of_feasibility
        self.threshold = threshold

    def __call__(self, X: Tensor) -> Tensor:
        """Return threshold minus P(feasible), where values <= 0 are feasible."""
        return self.threshold - self.probability_of_feasibility(X)


SemanticConstraint: TypeAlias = ContinuousConstraint | ClassificationConstraint


@dataclass(frozen=True, slots=True)
class ConstraintCollection(Sequence[SemanticConstraint]):
    """Ordered mixed semantic constraints for one optimization problem."""

    constraints: tuple[SemanticConstraint, ...]

    def __init__(self, *constraints: SemanticConstraint) -> None:
        """Initialize an immutable ordered constraint collection."""
        constraint_types = (ContinuousConstraint, ClassificationConstraint)
        if not all(isinstance(constraint, constraint_types) for constraint in constraints):
            raise TypeError(
                "ConstraintCollection accepts ContinuousConstraint or "
                "ClassificationConstraint instances."
            )
        object.__setattr__(self, "constraints", tuple(constraints))

    def __len__(self) -> int:
        """Return the number of semantic constraints."""
        return len(self.constraints)

    def __getitem__(
        self,
        index: int | slice,
    ) -> SemanticConstraint | tuple[SemanticConstraint, ...]:
        """Return constraints in their declared order."""
        return self.constraints[index]

    def __iter__(self) -> Iterator[SemanticConstraint]:
        """Iterate over constraints in their declared order."""
        return iter(self.constraints)

    def resolve_outputs(self, model: HeterogeneousModel) -> tuple[int, ...]:
        """Resolve constraints to canonical heterogeneous output indices."""
        return tuple(constraint.resolve_output(model) for constraint in self.constraints)

    def resolve_owners(
        self,
        model: HeterogeneousModel,
    ) -> tuple[tuple[int, int], ...]:
        """Resolve constraints to entry and local output indices."""
        return tuple(constraint.resolve_owner(model) for constraint in self.constraints)

    def group_by_entry(
        self,
        model: HeterogeneousModel,
    ) -> dict[int, tuple[SemanticConstraint, ...]]:
        """Group mixed constraints by model entry while preserving declared order."""
        grouped: dict[int, list[SemanticConstraint]] = {}
        for constraint, (entry_index, _) in zip(
            self.constraints,
            self.resolve_owners(model),
            strict=True,
        ):
            grouped.setdefault(entry_index, []).append(constraint)
        return {entry: tuple(constraints) for entry, constraints in grouped.items()}

    def classification_constraints(self) -> tuple[ClassificationConstraint, ...]:
        """Return classification constraints in their declared order."""
        return tuple(
            constraint
            for constraint in self.constraints
            if isinstance(constraint, ClassificationConstraint)
        )

    def group_classification_by_output(
        self,
        model: HeterogeneousModel,
    ) -> dict[int, tuple[ClassificationConstraint, ...]]:
        """Group classification constraints by canonical output index."""
        grouped: dict[int, list[ClassificationConstraint]] = {}
        for constraint in self.classification_constraints():
            output_index = constraint.resolve_output(model)
            grouped.setdefault(output_index, []).append(constraint)
        return {output: tuple(constraints) for output, constraints in grouped.items()}

    def group_classification_by_entry(
        self,
        model: HeterogeneousModel,
    ) -> dict[int, tuple[ClassificationConstraint, ...]]:
        """Group classification constraints by owning model entry."""
        grouped: dict[int, list[ClassificationConstraint]] = {}
        for constraint in self.classification_constraints():
            entry_index, _ = constraint.resolve_owner(model)
            grouped.setdefault(entry_index, []).append(constraint)
        return {entry: tuple(constraints) for entry, constraints in grouped.items()}

    def feasibility_representations(self, model: HeterogeneousModel):
        """Build heterogeneous feasibility representations in declaration order."""
        return tuple(
            constraint.to_feasibility_representation(model) for constraint in self.constraints
        )

    def validate(self, model: HeterogeneousModel) -> None:
        """Validate every constraint against the heterogeneous model contract."""
        self.resolve_outputs(model)
