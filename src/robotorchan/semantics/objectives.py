"""Semantic specifications for optimization objectives."""

from __future__ import annotations

from dataclasses import dataclass

from botorch.acquisition.objective import GenericMCObjective
from torch import Tensor

from robotorchan.models.capabilities import ObservationType
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.direction import ObjectiveDirection


@dataclass(frozen=True, slots=True)
class RegressionObjective:
    """Declare one regression output as a scalar optimization objective."""

    output: int | str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    def resolve_output(self, model: HeterogeneousModel) -> int:
        """Resolve and validate the referenced regression output."""
        output_index = model.resolve_output(self.output)
        if model.output_observation_type(output_index) is not ObservationType.REGRESSION:
            raise TypeError("RegressionObjective requires a regression output.")
        return output_index

    def to_botorch(self, model: HeterogeneousModel) -> GenericMCObjective:
        """Create a BoTorch MC objective for heterogeneous posterior samples."""
        output_index = self.resolve_output(model)
        _, local_output_index = model.output_owner(output_index)

        def objective(samples: Tensor, X: Tensor | None = None) -> Tensor:
            del X
            return self.direction.apply(samples[..., local_output_index])

        return GenericMCObjective(objective)
