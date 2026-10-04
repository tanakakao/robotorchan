"""Multi-task binary classification surrogate models."""

from robotorchan.models.classification.standard.binary import (
    KroneckerMultiTaskBinaryGPClassifier,
    MultiTaskBinaryGPClassifier,
)

__all__ = [
    "KroneckerMultiTaskBinaryGPClassifier",
    "MultiTaskBinaryGPClassifier",
]
