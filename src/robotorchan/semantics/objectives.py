"""Semantic specifications for optimization objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
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



@dataclass(frozen=True, slots=True)
class ProbabilityObjective:
    """Declare one classification probability as an optimization objective."""

    output: int | str
    class_index: int
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    def resolve_output(self, model: HeterogeneousModel) -> int:
        """Resolve and validate the referenced classification output."""
        output_index = model.resolve_output(self.output)
        if model.output_observation_type(output_index) is not ObservationType.CLASSIFICATION:
            raise TypeError("ProbabilityObjective requires a classification output.")
        metadata = model.output_classification_metadata(output_index)
        if metadata is None:
            raise TypeError("ProbabilityObjective requires classification metadata.")
        if not isinstance(self.class_index, int) or isinstance(self.class_index, bool):
            raise TypeError("class_index must be an integer.")
        if not 0 <= self.class_index < metadata.num_classes:
            raise ValueError("class_index is outside the classifier class range.")
        return output_index

    def evaluate(self, model: HeterogeneousModel, X: Tensor) -> Tensor:
        """Return the directed posterior-predictive class probability."""
        output_index = self.resolve_output(model)
        entry_index, _ = model.output_owner(output_index)
        probabilities = model.entry_predict_proba(entry_index, X)
        return self.direction.apply(probabilities[..., self.class_index])

    def sample(
        self,
        model: HeterogeneousModel,
        X: Tensor,
        *,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        """Return directed class-probability samples induced by the latent posterior."""
        output_index = self.resolve_output(model)
        entry_index, _ = model.output_owner(output_index)
        classifier = model[entry_index]
        probabilities = classifier.sample_class_probabilities(X, sample_shape=sample_shape)
        return self.direction.apply(probabilities[..., self.class_index])
