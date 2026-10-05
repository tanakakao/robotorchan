"""Robust binary classification contracts and models."""

from robotorchan.models.classification.binary.robust.base import (
    ClassificationRobustnessType,
    RobustBinaryClassificationMixin,
    RobustClassificationMetadata,
)
from robotorchan.models.classification.binary.robust.contaminated import (
    ContaminatedBernoulliLikelihood,
    ContaminatedBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust.input_dependent_noise import (
    InputDependentLabelNoiseBinarySingleTaskGPClassifier,
    InputDependentLabelNoiseLikelihood,
)
from robotorchan.models.classification.binary.robust.label_noise import (
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust.nonstationary import (
    NonstationaryBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust.replicate import (
    ReplicateLabelBinarySingleTaskGPClassifier,
)

__all__ = [
    "ClassificationRobustnessType",
    "ContaminatedBernoulliLikelihood",
    "ContaminatedBinarySingleTaskGPClassifier",
    "InputDependentLabelNoiseBinarySingleTaskGPClassifier",
    "InputDependentLabelNoiseLikelihood",
    "LabelNoiseBernoulliLikelihood",
    "NonstationaryBinarySingleTaskGPClassifier",
    "LabelNoiseBinarySingleTaskGPClassifier",
    "ReplicateLabelBinarySingleTaskGPClassifier",
    "RobustBinaryClassificationMixin",
    "RobustClassificationMetadata",
]
