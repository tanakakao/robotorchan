"""Public registry for classification surrogate-model constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torch import nn

from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    InferenceType,
    InputType,
    ModelCapabilities,
    ObservationType,
    PosteriorSamplingType,
    TaskType,
)
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
from robotorchan.models.classification.binary.standard.multitask import (
    KroneckerMultiTaskBinaryGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
)

ClassificationModelType: TypeAlias = type[nn.Module]


@dataclass(frozen=True, slots=True)
class ClassificationModelRegistryEntry:
    """Stable public identifier and constructor for a classification model."""

    model_id: str
    model_class: ClassificationModelType
    num_classes: int | None
    family: str
    capabilities: ModelCapabilities


def _binary_capabilities(
    *,
    input_type: InputType = InputType.CONTINUOUS,
    task_type: TaskType = TaskType.SINGLE,
    high_dimensional: HighDimensionalStrategy = HighDimensionalStrategy.NONE,
    posterior_sampling_type: PosteriorSamplingType = PosteriorSamplingType.GAUSSIAN,
    supports_multi_output: bool = False,
) -> ModelCapabilities:
    """Build reviewed capabilities shared by binary variational classifiers."""
    return ModelCapabilities(
        observation_type=ObservationType.CLASSIFICATION,
        input_type=input_type,
        task_type=task_type,
        inference=InferenceType.VARIATIONAL,
        high_dimensional=high_dimensional,
        supports_multi_output=supports_multi_output,
        supports_posterior_samples=True,
        posterior_sampling_type=posterior_sampling_type,
        supports_fantasize=False,
    )


_ENTRIES = (
    ("binary.standard", BinarySingleTaskGPClassifier, "standard", _binary_capabilities()),
    (
        "binary.mixed",
        MixedBinarySingleTaskGPClassifier,
        "mixed",
        _binary_capabilities(input_type=InputType.MIXED),
    ),
    (
        "binary.multitask",
        MultiTaskBinaryGPClassifier,
        "multitask",
        _binary_capabilities(task_type=TaskType.MULTITASK, supports_multi_output=True),
    ),
    (
        "binary.kronecker_multitask",
        KroneckerMultiTaskBinaryGPClassifier,
        "multitask",
        _binary_capabilities(task_type=TaskType.MULTITASK, supports_multi_output=True),
    ),
    (
        "binary.map_saas",
        MapSaasBinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.MAP_SAAS),
    ),
    (
        "binary.saas",
        SaasBinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.SAAS),
    ),
    (
        "binary.reduced",
        ReducedBinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.REDUCTION),
    ),
    (
        "binary.pca",
        PCABinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.REDUCTION),
    ),
    (
        "binary.pls",
        PLSBinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.REDUCTION),
    ),
    (
        "binary.random_projection",
        RandomProjectionBinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.REDUCTION),
    ),
    (
        "binary.alebo",
        ALEBOBinarySingleTaskGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.RANDOM_EMBEDDING),
    ),
    (
        "binary.joint_encoder",
        JointEncoderBinaryGPClassifier,
        "high_dimensional",
        _binary_capabilities(high_dimensional=HighDimensionalStrategy.NEURAL_REDUCTION),
    ),
    (
        "binary.deep_gp",
        BinarySingleTaskDeepGPClassifier,
        "high_dimensional",
        _binary_capabilities(
            high_dimensional=HighDimensionalStrategy.DEEP,
            posterior_sampling_type=PosteriorSamplingType.STOCHASTIC,
        ),
    ),
)

CLASSIFICATION_MODEL_REGISTRY = {
    model_id: ClassificationModelRegistryEntry(
        model_id=model_id,
        model_class=model_class,
        num_classes=2,
        family=family,
        capabilities=capabilities,
    )
    for model_id, model_class, family, capabilities in _ENTRIES
}


def get_classification_model_entry(model_id: str) -> ClassificationModelRegistryEntry:
    """Return metadata and constructor for a stable classification model ID."""
    try:
        return CLASSIFICATION_MODEL_REGISTRY[model_id]
    except KeyError as error:
        available = ", ".join(CLASSIFICATION_MODEL_REGISTRY)
        message = f"Unknown classification model {model_id!r}. Available: {available}"
        raise KeyError(message) from error


def get_classification_model_class(model_id: str) -> ClassificationModelType:
    """Return the constructor class associated with a stable model ID."""
    return get_classification_model_entry(model_id).model_class


def list_classification_models(*, prefix: str | None = None) -> tuple[str, ...]:
    """List stable public classification model IDs, optionally filtered by prefix."""
    names = tuple(CLASSIFICATION_MODEL_REGISTRY)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))
