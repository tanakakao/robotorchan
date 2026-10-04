"""Tests for the public classification model registry."""

import pytest

from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    BinarySingleTaskGPClassifier,
    get_classification_model_class,
    get_classification_model_entry,
    list_classification_models,
)


def test_binary_standard_resolves_to_public_classifier() -> None:
    entry = get_classification_model_entry("binary.standard")
    assert entry.model_class is BinarySingleTaskGPClassifier
    assert entry.num_classes == 2
    assert get_classification_model_class("binary.standard") is BinarySingleTaskGPClassifier


def test_registry_uses_stable_unique_ids() -> None:
    names = list_classification_models()
    assert len(names) == len(set(names)) == len(CLASSIFICATION_MODEL_REGISTRY)
    assert all(name.startswith("binary.") for name in names)


def test_prefix_filter_is_multiclass_ready() -> None:
    assert list_classification_models(prefix="binary.") == list_classification_models()
    assert list_classification_models(prefix="multiclass.") == ()


def test_unknown_model_reports_available_ids() -> None:
    with pytest.raises(KeyError, match=r"Available: binary\.standard"):
        get_classification_model_entry("binary.unknown")
