"""Regression-protection tests for classification metadata additions."""

from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.capabilities import ModelCapabilities, ObservationType
from robotorchan.models.classification import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY


def test_model_capabilities_default_to_regression_observations() -> None:
    capabilities = ModelCapabilities()

    assert capabilities.observation_type is ObservationType.REGRESSION


def test_every_regression_registry_entry_remains_regression() -> None:
    assert MODEL_REGISTRY
    assert all(
        entry.capabilities.observation_type is ObservationType.REGRESSION
        for entry in MODEL_REGISTRY.values()
    )


def test_classification_registry_remains_separate_from_regression_registry() -> None:
    assert CLASSIFICATION_MODEL_REGISTRY
    assert set(MODEL_REGISTRY).isdisjoint(CLASSIFICATION_MODEL_REGISTRY)


def test_existing_active_learning_acquisitions_remain_regression_compatible() -> None:
    regression_active_learning = (
        name
        for name, entry in ACQUISITION_REGISTRY.items()
        if entry.capabilities.purpose is AcquisitionPurpose.ACTIVE_LEARNING
        and ObservationType.REGRESSION in entry.capabilities.observation_types
    )

    for acquisition_name in regression_active_learning:
        result = check_model_acquisition_compatibility("SingleTaskGP", acquisition_name)
        assert result.status is CompatibilityStatus.COMPATIBLE, acquisition_name


def test_existing_botorch_bo_acquisition_remains_regression_compatible() -> None:
    result = check_model_acquisition_compatibility(
        "SingleTaskGP",
        "qLogExpectedImprovement",
    )

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.reasons == ()


def test_regression_acquisitions_do_not_silently_become_classification_only() -> None:
    for name, entry in ACQUISITION_REGISTRY.items():
        if name in {
            "BALD",
            "LatentStraddle",
            "MarginUncertainty",
            "PredictiveEntropy",
            "ProbabilityVariance",
        }:
            continue
        assert ObservationType.REGRESSION in entry.capabilities.observation_types, name
