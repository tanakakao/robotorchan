"""High-dimensional classification surrogate models."""

from robotorchan.models.classification.high_dimensional.alebo import (
    ALEBOBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.high_dimensional.joint_neural import (
    JointEncoderBinaryGPClassifier,
)
from robotorchan.models.classification.high_dimensional.map_saas import (
    MapSaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.high_dimensional.reduced import (
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.high_dimensional.saas import (
    SaasBinarySingleTaskGPClassifier,
)

__all__ = [
    "ALEBOBinarySingleTaskGPClassifier",
    "JointEncoderBinaryGPClassifier",
    "MapSaasBinarySingleTaskGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "SaasBinarySingleTaskGPClassifier",
]
