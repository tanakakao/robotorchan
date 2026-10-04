"""Standard binary classification surrogate models."""

from robotorchan.models.classification.binary.standard.model_list import ClassificationModelList
from robotorchan.models.classification.binary.standard.multitask import (
    KroneckerMultiTaskBinaryGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
)

__all__ = [
    "BinarySingleTaskGPClassifier",
    "ClassificationModelList",
    "KroneckerMultiTaskBinaryGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
]
