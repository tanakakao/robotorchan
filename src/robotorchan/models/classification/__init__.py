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
    BinarySingleTaskDeepGPClassifier,
    JointEncoderBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.registry import (
    CLASSIFICATION_MODEL_REGISTRY,
    ClassificationModelRegistryEntry,
    get_classification_model_class,
    get_classification_model_entry,
    list_classification_models,
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
    "BinarySingleTaskDeepGPClassifier",
    "BinarySingleTaskGPClassifier",
    "CLASSIFICATION_MODEL_REGISTRY",
    "ClassificationLikelihoodFamily",
    "ClassificationMetadata",
    "ClassificationModelList",
    "ClassificationModelMixin",
    "ClassificationModelRegistryEntry",
    "get_classification_model_class",
    "get_classification_model_entry",
    "JointEncoderBinaryGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "LatentOutputStructure",
    "list_classification_models",
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
