"""Binary classification surrogate models."""

from robotorchan.models.classification.binary.expressive import BinarySingleTaskDeepGPClassifier
from robotorchan.models.classification.binary.high_dimensional import (
    ALEBOBinarySingleTaskGPClassifier,
    JointEncoderBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust import (
    ClassificationRobustnessType,
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
    RobustBinaryClassificationMixin,
    RobustClassificationMetadata,
)
from robotorchan.models.classification.binary.standard import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)

__all__ = [

]""Binary classification surrogate models."""

from robotorchan.models.classification.binary.expressive import BinarySingleTaskDeepGPClassifier
from robotorchan.models.classification.binary.high_dimensional import (
    ALEBOBinarySingleTaskGPClassifier,
    JointEncoderBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust import (
    ClassificationRobustnessType,
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
    RobustBinaryClassificationMixin,
    RobustClassificationMetadata,
)
from robotorchan.models.classification.binary.standard import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)

__all__ = [
    "ALEBOBinarySingleTaskGPClassifier",
    "BinarySingleTaskDeepGPClassifier",
    "BinarySingleTaskGPClassifier",
    "ClassificationRobustnessType",
    "JointEncoderBinaryGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "LabelNoiseBernoulliLikelihood",
    "LabelNoiseBinarySingleTaskGPClassifier",
    "MapSaasBinarySingleTaskGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "RobustBinaryClassificationMixin",
    "RobustClassificationMetadata",
    "SaasBinarySingleTaskGPClassifier",
]