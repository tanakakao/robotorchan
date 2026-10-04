"""Classification surrogate-model contracts."""

from robotorchan.models.classification.binary import BinarySingleTaskGPClassifier

from robotorchan.models.classification.base import (
    BinaryClassificationMixin,
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.validation import validate_binary_labels

__all__ = [
    "BinarySingleTaskGPClassifier",
    "BinaryClassificationMixin",
    "ClassificationLikelihoodFamily",
    "ClassificationMetadata",
    "ClassificationModelMixin",
    "LatentOutputStructure",
    "validate_binary_labels",
]
