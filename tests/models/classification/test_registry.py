"""Tests for the public classification model registry."""

import pytest

from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    InferenceType,
    InputType,
    ObservationType,
    PosteriorSamplingType,
    TaskType,
)
from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    BinarySingleTaskGPClassifier,
    get_classification_model_class,
    get_classification_model_entry,
    list_classification_models,
)


def test_binary_standard_resolves_to_public_classifier() -> None:
    entry = get_classification_model_entry("binary.standard")
    assert entry.model_class is BinarySingleTaskGPClassifier
    assert entry.num_classes == 2
    assert get_classification_model_class("binary.standard") is BinarySingleTaskGPClassifier


def test_registry_uses_stable_unique_ids() -> None:
    names = list_classification_models()
    assert len(names) == len(set(names)) == len(CLASSIFICATION_MODEL_REGISTRY)
    assert all(name.startswith("binary.") for name in names)


def test_prefix_filter_is_multiclass_ready() -> None:
    assert list_classification_models(prefix="binary.") == list_classification_models()
    assert list_classification_models(prefix="multiclass.") == ()


def test_unknown_model_reports_available_ids() -> None:
    with pytest.raises(KeyError, match=r"Available: .*binary\.standard"):
        get_classification_model_entry("binary.unknown")


def test_binary_registry_exposes_classification_capabilities() -> None:
    standard = get_classification_model_entry("binary.standard").capabilities
    assert standard.observation_type is ObservationType.CLASSIFICATION
    assert standard.input_type is InputType.CONTINUOUS
    assert standard.task_type is TaskType.SINGLE
    assert standard.supports_posterior_samples
    assert not standard.supports_fantasize


def test_structural_classifier_capabilities_are_explicit() -> None:
    mixed = get_classification_model_entry("binary.mixed").capabilities
    multitask = get_classification_model_entry("binary.multitask").capabilities
    assert mixed.input_type is InputType.MIXED
    assert multitask.task_type is TaskType.MULTITASK
    assert multitask.supports_multi_output


def test_high_dimensional_and_deep_capabilities_are_explicit() -> None:
    pca = get_classification_model_entry("binary.pca").capabilities
    deep_entry = get_classification_model_entry("binary.deep_gp")
    deep = deep_entry.capabilities
    assert pca.high_dimensional is HighDimensionalStrategy.REDUCTION
    assert deep_entry.family == "expressive"
    assert deep.high_dimensional is HighDimensionalStrategy.DEEP
    assert deep.posterior_sampling_type is PosteriorSamplingType.STOCHASTIC


def test_binary_saas_is_variational_not_fully_bayesian() -> None:
    """Binary SAAS currently means variational SAAS-style shrinkage."""
    entry = get_classification_model_entry("binary.saas")
    assert entry.family == "high_dimensional"
    assert entry.capabilities.high_dimensional is HighDimensionalStrategy.SAAS
    assert entry.capabilities.inference is InferenceType.VARIATIONAL


def test_registry_family_tracks_filesystem_family_not_capability_axes() -> None:
    registry = CLASSIFICATION_MODEL_REGISTRY

    for model_id in (
        "binary.standard",
        "binary.mixed",
        "binary.multitask",
        "binary.kronecker_multitask",
    ):
        assert registry[model_id].family == "standard"

    for model_id in (
        "binary.map_saas",
        "binary.saas",
        "binary.reduced",
        "binary.pca",
        "binary.pls",
        "binary.random_projection",
        "binary.alebo",
        "binary.joint_encoder",
    ):
        assert registry[model_id].family == "high_dimensional"

    assert registry["binary.deep_gp"].family == "expressive"


def test_registry_fragment_registration_is_task_agnostic() -> None:
    from robotorchan.models.classification.registry import _register_fragment

    registry = {}
    capabilities = get_classification_model_entry("binary.standard").capabilities
    specs = (
        (
            "multiclass.standard",
            BinarySingleTaskGPClassifier,
            "standard",
            capabilities,
        ),
    )

    _register_fragment(registry, specs, num_classes=None)

    entry = registry["multiclass.standard"]
    assert entry.model_id == "multiclass.standard"
    assert entry.num_classes is None
    assert entry.family == "standard"
