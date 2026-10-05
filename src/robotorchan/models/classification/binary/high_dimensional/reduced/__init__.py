"""Input-reduced classification surrogate models."""

from robotorchan.models.classification.binary.high_dimensional.reduced.base import (
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
)

from robotorchan.models.classification.binary.high_dimensional.reduced.joint_neural import (
    JointEncoderBinaryGPClassifier,
)

__all__ = [
    "JointEncoderBinaryGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
]
