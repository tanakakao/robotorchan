"""Ownership contracts for classification task-specific helpers."""

from robotorchan.models.classification import (
    BinaryClassificationMixin as PublicBinaryClassificationMixin,
    base as common_base,
    validate_binary_labels as public_validate_binary_labels,
    validation as common_validation,
)
from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.binary.validation import validate_binary_labels


def test_binary_mixin_has_binary_canonical_owner() -> None:
    assert BinaryClassificationMixin is PublicBinaryClassificationMixin
    assert not hasattr(common_base, "BinaryClassificationMixin")


def test_binary_validation_has_binary_canonical_owner() -> None:
    assert validate_binary_labels is public_validate_binary_labels
    assert not hasattr(common_validation, "validate_binary_labels")
