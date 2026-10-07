"""Semantic specifications for optimization objectives."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TypeAlias

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

    def resolve_owner(self, model: HeterogeneousModel) -> tuple[int, int]:
        """Resolve the referenced output to its entry and local output index."""
        return model.output_owner(self.resolve_output(model))

    def to_botorch(self, model: HeterogeneousModel) -> GenericMCObjective:
        """Create a BoTorch MC objective for the referenced entry samples."""
        _, local_output_index = self.resolve_owner(model)

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

    def evaluate(
        self,
        model: HeterogeneousModel,
        X: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Return the directed posterior-predictive class probability."""
        output_index = self.resolve_output(model)
        entry_index, _ = model.output_owner(output_index)
        probabilities = model.entry_predict_proba(entry_index, X, **kwargs)
        return self.direction.apply(probabilities[..., self.class_index])

    def sample(
        self,
        model: HeterogeneousModel,
        X: Tensor,
        *,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Return directed class-probability samples induced by the latent posterior."""
        output_index = self.resolve_output(model)
        entry_index, _ = model.output_owner(output_index)
        classifier = model[entry_index]
        probabilities = classifier.sample_class_probabilities(
            X,
            sample_shape=sample_shape,
            **kwargs,
        )
        return self.direction.apply(probabilities[..., self.class_index])


SemanticObjective: TypeAlias = RegressionObjective | ProbabilityObjective


@dataclass(frozen=True, slots=True)
class ObjectiveCollection(Sequence[SemanticObjective]):
    """Ordered semantic objectives for one optimization problem."""

    objectives: tuple[SemanticObjective, ...]

    def __init__(self, *objectives: SemanticObjective) -> None:
        """Initialize an immutable ordered objective collection."""
        if not objectives:
            raise ValueError("ObjectiveCollection requires at least one objective.")
        objective_types = (RegressionObjective, ProbabilityObjective)
        if not all(isinstance(objective, objective_types) for objective in objectives):
            raise TypeError(
                "ObjectiveCollection accepts RegressionObjective or ProbabilityObjective instances."
            )
        object.__setattr__(self, "objectives", tuple(objectives))

    def __len__(self) -> int:
        """Return the number of semantic objectives."""
        return len(self.objectives)

    def __getitem__(self, index: int | slice) -> SemanticObjective | tuple[SemanticObjective, ...]:
        """Return objectives in their declared order."""
        return self.objectives[index]

    def __iter__(self) -> Iterator[SemanticObjective]:
        """Iterate over objectives in their declared order."""
        return iter(self.objectives)

    def resolve_outputs(self, model: HeterogeneousModel) -> tuple[int, ...]:
        """Resolve every objective to its canonical heterogeneous output index."""
        return tuple(objective.resolve_output(model) for objective in self.objectives)

    def resolve_owners(
        self,
        model: HeterogeneousModel,
    ) -> tuple[tuple[int, int], ...]:
        """Resolve objective outputs to entry and local output indices."""
        owners = []
        for objective in self.objectives:
            output_index = objective.resolve_output(model)
            owners.append(model.output_owner(output_index))
        return tuple(owners)

    def group_by_entry(
        self,
        model: HeterogeneousModel,
    ) -> dict[int, tuple[SemanticObjective, ...]]:
        """Group objectives by owning model entry while preserving declaration order."""
        grouped: dict[int, list[SemanticObjective]] = {}
        for objective, (entry_index, _) in zip(
            self.objectives,
            self.resolve_owners(model),
            strict=True,
        ):
            grouped.setdefault(entry_index, []).append(objective)
        return {entry: tuple(objectives) for entry, objectives in grouped.items()}

    def validate(self, model: HeterogeneousModel) -> None:
        """Validate every objective against the heterogeneous model contract."""
        self.resolve_outputs(model)
