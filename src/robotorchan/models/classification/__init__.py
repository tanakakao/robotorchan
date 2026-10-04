"""Classification surrogate-model contracts."""

from robotorchan.models.classification.base import (
    BinaryClassificationMixin,
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.binary import (
    ALEBOBinarySingleTaskGPClassifier,
    BinarySingleTaskDeepGPClassifier,
    BinarySingleTaskGPClassifier,
    JointEncoderBinaryGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.model_list import ClassificationModelList
from robotorchan.models.classification.registry import (
    CLASSIFICATION_MODEL_REGISTRY,
    ClassificationModelRegistryEntry,
    get_classification_model_class,
    get_classification_model_entry,
    list_classification_models,
)
from robotorchan.models.classification.validation import validate_binary_labels

__all__ = [
    "CLASSIFICATION_MODEL_REGISTRY",
    "ALEBOBinarySingleTaskGPClassifier",
    "BinaryClassificationMixin",
    "BinarySingleTaskDeepGPClassifier",
    "BinarySingleTaskGPClassifier",
    "ClassificationLikelihoodFamily",
    "ClassificationMetadata",
    "ClassificationModelList",
    "ClassificationModelMixin",
    "ClassificationModelRegistryEntry",
    "JointEncoderBinaryGPClassifier",
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
    "get_classification_model_class",
    "get_classification_model_entry",
    "list_classification_models",
    "validate_binary_labels",
]
