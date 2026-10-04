"""Binary classification surrogate models."""

from robotorchan.models.classification.binary.high_dimensional import (
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
from robotorchan.models.classification.binary.standard import (
    BinarySingleTaskGPClassifier,
    ClassificationModelList,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)

__all__ = [
    "ALEBOBinarySingleTaskGPClassifier",
    "BinarySingleTaskDeepGPClassifier",
    "BinarySingleTaskGPClassifier",
    "ClassificationModelList",
    "JointEncoderBinaryGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "MapSaasBinarySingleTaskGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "SaasBinarySingleTaskGPClassifier",
]
