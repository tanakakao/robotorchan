"""Classification surrogate-model contracts."""

from robotorchan.models.classification.base import (
    BinaryClassificationMixin,
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.high_dimensional import (
    ALEBOBinarySingleTaskGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.standard.binary import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.standard.model_list import ClassificationModelList
from robotorchan.models.classification.validation import validate_binary_labels

__all__ = [
    "ALEBOBinarySingleTaskGPClassifier",
    "BinaryClassificationMixin",
    "BinarySingleTaskGPClassifier",
    "ClassificationLikelihoodFamily",
    "ClassificationMetadata",
    "ClassificationModelList",
    "ClassificationModelMixin",
    "KroneckerMultiTaskBinaryGPClassifier",
    "LatentOutputStructure",
    "MapSaasBinarySingleTaskGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "SaasBinarySingleTaskGPClassifier",
    "validate_binary_labels",
]
