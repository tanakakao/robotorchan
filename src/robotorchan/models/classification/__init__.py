"""Classification surrogate-model contracts."""

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.binary import (
    ALEBOBinarySingleTaskGPClassifier,
    BinarySingleTaskDeepGPClassifier,
    BinarySingleTaskGPClassifier,
    ClassificationRobustnessType,
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
    JointEncoderBinaryGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    RobustBinaryClassificationMixin,
    RobustClassificationMetadata,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.binary.validation import validate_binary_labels
from robotorchan.models.classification.model_list import ClassificationModelList
from robotorchan.models.classification.registry import (
    CLASSIFICATION_MODEL_REGISTRY,
    ClassificationModelRegistryEntry,
    get_classification_model_class,
    get_classification_model_entry,
    list_classification_models,
)

__all__ = [

]""Classification surrogate-model contracts."""

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.binary import (
    ALEBOBinarySingleTaskGPClassifier,
    BinarySingleTaskDeepGPClassifier,
    BinarySingleTaskGPClassifier,
    ClassificationRobustnessType,
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
    JointEncoderBinaryGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
    MapSaasBinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
    ReducedBinarySingleTaskGPClassifier,
    RobustBinaryClassificationMixin,
    RobustClassificationMetadata,
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.binary.validation import validate_binary_labels
from robotorchan.models.classification.model_list import ClassificationModelList
from robotorchan.models.classification.registry import (
    CLASSIFICATION_MODEL_REGISTRY,
    ClassificationModelRegistryEntry,
    get_classification_model_class,
    get_classification_model_entry,
    list_classification_models,
)

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
    "ClassificationRobustnessType",
    "JointEncoderBinaryGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "LabelNoiseBernoulliLikelihood",
    "LabelNoiseBinarySingleTaskGPClassifier",
    "LatentOutputStructure",
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
    "get_classification_model_class",
    "get_classification_model_entry",
    "list_classification_models",
    "validate_binary_labels",
]