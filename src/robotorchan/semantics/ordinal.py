"""Ordinal and utility semantics for classification probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.models.capabilities import ObservationType
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.classes import resolve_class
from robotorchan.semantics.direction import ObjectiveDirection


def _classification_entry(
    model: HeterogeneousModel,
    output: int | str,
) -> tuple[int, object]:
    output_index = model.resolve_output(output)
    if model.output_observation_type(output_index) is not ObservationType.CLASSIFICATION:
        raise TypeError("Ordinal and utility semantics require a classification output.")
    metadata = model.output_classification_metadata(output_index)
    if metadata is None:
        raise TypeError("Ordinal and utility semantics require classification metadata.")
    entry_index, _ = model.output_owner(output_index)
    return entry_index, metadata


def _resolve_order(
    class_order: tuple[object, ...],
    metadata: object,
) -> tuple[int, ...]:
    labels = metadata.class_labels
    if len(class_order) != metadata.num_classes or len(set(class_order)) != len(class_order):
        raise ValueError("class_order must contain every classifier class exactly once.")
    try:
        indices = tuple(labels.index(label) for label in class_order)
    except ValueError as error:
        raise ValueError("class_order contains a label not present in classifier class_labels.") from error
    if len(set(indices)) != metadata.num_classes:
        raise ValueError("class_order must contain every classifier class exactly once.")
    return indices


@dataclass(frozen=True, slots=True)
class OrdinalProbabilityObjective:
    """Optimize probability mass above or below an explicit ordinal threshold."""

    output: int | str
    threshold_class: int | str
    class_order: tuple[object, ...]
    at_or_above: bool = True
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    def evaluate(self, model: HeterogeneousModel, X: Tensor, **kwargs: object) -> Tensor:
        """Return directed posterior-predictive probability of the ordinal event."""
        entry_index, metadata = _classification_entry(model, self.output)
        order = _resolve_order(self.class_order, metadata)
        threshold = resolve_class(
            self.threshold_class,
            metadata,
            argument_name="threshold_class",
        )
        threshold_position = order.index(threshold)
        selected = order[threshold_position:] if self.at_or_above else order[: threshold_position + 1]
        probabilities = model.entry_predict_proba(entry_index, X, **kwargs)
        value = probabilities[..., list(selected)].sum(dim=-1)
        return self.direction.apply(value)

    def sample(
        self,
        model: HeterogeneousModel,
        X: Tensor,
        *,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Return posterior samples of ordinal-event probability."""
        entry_index, metadata = _classification_entry(model, self.output)
        order = _resolve_order(self.class_order, metadata)
        threshold = resolve_class(
            self.threshold_class,
            metadata,
            argument_name="threshold_class",
        )
        threshold_position = order.index(threshold)
        selected = order[threshold_position:] if self.at_or_above else order[: threshold_position + 1]
        probabilities = model[entry_index].sample_class_probabilities(
            X,
            sample_shape=sample_shape,
            **kwargs,
        )
        value = probabilities[..., list(selected)].sum(dim=-1)
        return self.direction.apply(value)


@dataclass(frozen=True, slots=True)
class ExpectedClassUtilityObjective:
    """Optimize expected utility from an explicit class-to-utility mapping."""

    output: int | str
    utilities: tuple[tuple[object, float], ...]
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    def _utility_vector(self, metadata: object, reference: Tensor) -> Tensor:
        mapping = dict(self.utilities)
        if len(mapping) != len(self.utilities):
            raise ValueError("utilities must not contain duplicate class labels.")
        labels = metadata.class_labels
        if set(mapping) != set(labels):
            raise ValueError("utilities must define exactly one value for every classifier class.")
        return reference.new_tensor([mapping[label] for label in labels])

    def evaluate(self, model: HeterogeneousModel, X: Tensor, **kwargs: object) -> Tensor:
        """Return directed posterior-predictive expected class utility."""
        entry_index, metadata = _classification_entry(model, self.output)
        probabilities = model.entry_predict_proba(entry_index, X, **kwargs)
        utilities = self._utility_vector(metadata, probabilities)
        return self.direction.apply((probabilities * utilities).sum(dim=-1))

    def sample(
        self,
        model: HeterogeneousModel,
        X: Tensor,
        *,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Return posterior samples of expected class utility."""
        entry_index, metadata = _classification_entry(model, self.output)
        probabilities = model[entry_index].sample_class_probabilities(
            X,
            sample_shape=sample_shape,
            **kwargs,
        )
        utilities = self._utility_vector(metadata, probabilities)
        return self.direction.apply((probabilities * utilities).sum(dim=-1))
