"""Public registry for classification surrogate-model constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torch import nn

from robotorchan.models.classification.high_dimensional import (
    ALEBOBinarySingleTaskGPClassifier,
    BinarySingleTaskDeepGPClassifier,
    JointEncoderBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.standard.binary import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)

ClassificationModelType: TypeAlias = type[nn.Module]


@dataclass(frozen=True, slots=True)
class ClassificationModelRegistryEntry:
    """Stable public identifier and constructor for a classification model."""

    model_id: str
    model_class: ClassificationModelType
    num_classes: int | None
    family: str


_ENTRIES = (
    ("binary.standard", BinarySingleTaskGPClassifier, "standard"),
    ("binary.mixed", MixedBinarySingleTaskGPClassifier, "mixed"),
    ("binary.multitask", MultiTaskBinaryGPClassifier, "multitask"),
    ("binary.kronecker_multitask", KroneckerMultiTaskBinaryGPClassifier, "multitask"),
    ("binary.map_saas", MapSaasBinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.saas", SaasBinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.reduced", ReducedBinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.pca", PCABinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.pls", PLSBinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.random_projection", RandomProjectionBinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.alebo", ALEBOBinarySingleTaskGPClassifier, "high_dimensional"),
    ("binary.joint_encoder", JointEncoderBinaryGPClassifier, "high_dimensional"),
    ("binary.deep_gp", BinarySingleTaskDeepGPClassifier, "high_dimensional"),
)

CLASSIFICATION_MODEL_REGISTRY = {
    model_id: ClassificationModelRegistryEntry(
        model_id=model_id, model_class=model_class, num_classes=2, family=family
    )
    for model_id, model_class, family in _ENTRIES
}


def get_classification_model_entry(model_id: str) -> ClassificationModelRegistryEntry:
    """Return metadata and constructor for a stable classification model ID."""
    try:
        return CLASSIFICATION_MODEL_REGISTRY[model_id]
    except KeyError as error:
        available = ", ".join(CLASSIFICATION_MODEL_REGISTRY)
        raise KeyError(f"Unknown classification model {model_id!r}. Available: {available}") from error


def get_classification_model_class(model_id: str) -> ClassificationModelType:
    """Return the constructor class associated with a stable model ID."""
    return get_classification_model_entry(model_id).model_class


def list_classification_models(*, prefix: str | None = None) -> tuple[str, ...]:
    """List stable public classification model IDs, optionally filtered by prefix."""
    names = tuple(CLASSIFICATION_MODEL_REGISTRY)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))
