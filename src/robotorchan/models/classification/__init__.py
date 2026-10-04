"""Classification surrogate-model contracts."""

from robotorchan.models.classification.base import (
    BinaryClassificationMixin,
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.binary import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.validation import validate_binary_labels

__all__ = [
    "BinaryClassificationMixin",
    "BinarySingleTaskGPClassifier",
    "ClassificationLikelihoodFamily",
    "ClassificationMetadata",
    "ClassificationModelMixin",
    "KroneckerMultiTaskBinaryGPClassifier",
    "LatentOutputStructure",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "validate_binary_labels",
]
