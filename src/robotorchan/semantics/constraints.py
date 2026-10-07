"""Semantic specifications for outcome constraints."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from robotorchan.models.capabilities import ObservationType
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.direction import ConstraintDirection


@dataclass(frozen=True, slots=True)
class ContinuousConstraint:
    """Declare a one-sided constraint on one regression output."""

    output: int | str
    threshold: float
    direction: ConstraintDirection = ConstraintDirection.LESS_THAN_OR_EQUAL

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
