"""High-dimensional binary classification surrogate models."""

from robotorchan.models.classification.binary.high_dimensional.alebo import (
    ALEBOBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.high_dimensional.deep_gp import (
    BinarySingleTaskDeepGPClassifier,
)
from robotorchan.models.classification.binary.high_dimensional.joint_neural import (
    JointEncoderBinaryGPClassifier,
)
from robotorchan.models.classification.binary.high_dimensional.map_saas import (
    MapSaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.high_dimensional.reduced import (
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.high_dimensional.saas import (
    SaasBinarySingleTaskGPClassifier,
)

__all__ = [
    "ALEBOBinarySingleTaskGPClassifier",
    "BinarySingleTaskDeepGPClassifier",
    "JointEncoderBinaryGPClassifier",
    "MapSaasBinarySingleTaskGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "SaasBinarySingleTaskGPClassifier",
]
