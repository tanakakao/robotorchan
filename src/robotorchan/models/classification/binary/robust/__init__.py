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
from robotorchan.models.classification.binary.robust.replicate import (
    ReplicateLabelBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust.label_noise import (
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
)

__all__ = [
    "ClassificationRobustnessType",
    "ContaminatedBernoulliLikelihood",
    "ContaminatedBinarySingleTaskGPClassifier",
    "LabelNoiseBernoulliLikelihood",
    "LabelNoiseBinarySingleTaskGPClassifier",
    "ReplicateLabelBinarySingleTaskGPClassifier",
    "RobustBinaryClassificationMixin",
    "RobustClassificationMetadata",
]
