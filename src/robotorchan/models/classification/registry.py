"""Public registry facade for classification surrogate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torch import nn

from robotorchan.models.capabilities import ModelCapabilities
from robotorchan.models.classification.binary.registry import (
    BINARY_CLASSIFICATION_MODEL_SPECS,
)

ClassificationModelType: TypeAlias = type[nn.Module]


@dataclass(frozen=True, slots=True)
class ClassificationModelRegistryEntry:
    """Stable public identifier and constructor for a classification model."""

    model_id: str
    model_class: ClassificationModelType
    num_classes: int | None
    family: str
    capabilities: ModelCapabilities


ClassificationModelSpec: TypeAlias = tuple[
    str,
    ClassificationModelType,
    str,
    ModelCapabilities,
]


def _register_fragment(
    registry: dict[str, ClassificationModelRegistryEntry],
    specs: tuple[ClassificationModelSpec, ...],
    *,
    num_classes: int | None,
) -> None:
    """Register one task-specific model fragment without task assumptions."""
    for model_id, model_class, family, capabilities in specs:
        if model_id in registry:
            raise RuntimeError(f"Duplicate classification model ID: {model_id!r}")
        registry[model_id] = ClassificationModelRegistryEntry(
            model_id=model_id,
            model_class=model_class,
            num_classes=num_classes,
            family=family,
            capabilities=capabilities,
        )


def _make_registry() -> dict[str, ClassificationModelRegistryEntry]:
    """Compose task-specific classification registry fragments."""
    registry: dict[str, ClassificationModelRegistryEntry] = {}
    _register_fragment(
        registry,
        BINARY_CLASSIFICATION_MODEL_SPECS,
        num_classes=2,
    )
    return registry


CLASSIFICATION_MODEL_REGISTRY = _make_registry()


def get_classification_model_entry(model_id: str) -> ClassificationModelRegistryEntry:
    """Return metadata and constructor for a stable classification model ID."""
    try:
        return CLASSIFICATION_MODEL_REGISTRY[model_id]
    except KeyError as error:
        available = ", ".join(CLASSIFICATION_MODEL_REGISTRY)
        message = f"Unknown classification model {model_id!r}. Available: {available}"
        raise KeyError(message) from error


def get_classification_model_class(model_id: str) -> ClassificationModelType:
    """Return the constructor class associated with a stable model ID."""
    return get_classification_model_entry(model_id).model_class


def list_classification_models(*, prefix: str | None = None) -> tuple[str, ...]:
    """List stable public classification model IDs, optionally filtered by prefix."""
    names = tuple(CLASSIFICATION_MODEL_REGISTRY)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))
