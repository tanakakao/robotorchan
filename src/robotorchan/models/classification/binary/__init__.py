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
    ContaminatedBernoulliLikelihood,
    ContaminatedBinarySingleTaskGPClassifier,
    InputDependentLabelNoiseBinarySingleTaskGPClassifier,
    InputDependentLabelNoiseLikelihood,
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
    MixedNonstationaryBinarySingleTaskGPClassifier,
    NonstationaryBinarySingleTaskGPClassifier,
    NonstationaryMultiTaskBinaryGPClassifier,
    ReplicateLabelBinarySingleTaskGPClassifier,
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
    "ContaminatedBernoulliLikelihood",
    "ContaminatedBinarySingleTaskGPClassifier",
    "InputDependentLabelNoiseBinarySingleTaskGPClassifier",
    "InputDependentLabelNoiseLikelihood",
    "JointEncoderBinaryGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "LabelNoiseBernoulliLikelihood",
    "LabelNoiseBinarySingleTaskGPClassifier",
    "MapSaasBinarySingleTaskGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "MixedNonstationaryBinarySingleTaskGPClassifier",
    "NonstationaryBinarySingleTaskGPClassifier",
    "NonstationaryMultiTaskBinaryGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "ReplicateLabelBinarySingleTaskGPClassifier",
    "RobustBinaryClassificationMixin",
    "RobustClassificationMetadata",
    "SaasBinarySingleTaskGPClassifier",
]
