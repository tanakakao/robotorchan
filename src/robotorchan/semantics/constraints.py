"""Semantic specifications for outcome constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from torch import Tensor

from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
)
from robotorchan.models.capabilities import ObservationType
from robotorchan.models.heterogeneous import HeterogeneousModel
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


@dataclass(frozen=True, slots=True)
class ClassificationConstraint:
    """Declare classifier membership in one class as feasibility."""

    output: int | str
    feasible_class: int = 1

    def resolve_output(self, model: HeterogeneousModel) -> int:
        """Resolve and validate the referenced classification output."""
        output_index = model.resolve_output(self.output)
        if model.output_observation_type(output_index) is not ObservationType.CLASSIFICATION:
            raise TypeError("ClassificationConstraint requires a classification output.")
        metadata = model.output_classification_metadata(output_index)
        if metadata is None:
            raise TypeError("ClassificationConstraint requires classification metadata.")
        if not isinstance(self.feasible_class, int) or isinstance(self.feasible_class, bool):
            raise TypeError("feasible_class must be an integer.")
        if not 0 <= self.feasible_class < metadata.num_classes:
            raise ValueError("feasible_class is outside the classifier class range.")
        return output_index

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
            feasible_class=self.feasible_class,
        )
