"""Standard classification surrogate models."""

from robotorchan.models.classification.standard.binary import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.standard.model_list import ClassificationModelList

__all__ = [
    "BinarySingleTaskGPClassifier",
    "ClassificationModelList",
    "KroneckerMultiTaskBinaryGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
]
