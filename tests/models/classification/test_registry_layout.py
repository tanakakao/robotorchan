"""Ownership contracts for task-specific classification registries."""

from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.registry import (
    BINARY_CLASSIFICATION_MODEL_SPECS,
)


def test_binary_registry_specs_are_owned_by_binary_package() -> None:
    model_ids = tuple(spec[0] for spec in BINARY_CLASSIFICATION_MODEL_SPECS)

    assert model_ids
    assert all(model_id.startswith("binary.") for model_id in model_ids)


def test_shared_registry_composes_binary_specs() -> None:
    model_ids = tuple(spec[0] for spec in BINARY_CLASSIFICATION_MODEL_SPECS)

    assert tuple(CLASSIFICATION_MODEL_REGISTRY) == model_ids
    assert CLASSIFICATION_MODEL_REGISTRY["binary.standard"].model_class is (
        BinarySingleTaskGPClassifier
    )


def test_binary_specs_do_not_construct_public_registry_entries() -> None:
    assert all(len(spec) == 4 for spec in BINARY_CLASSIFICATION_MODEL_SPECS)
