"""Tests for acquisition capability metadata and model compatibility."""

import robotorchan.acquisition as acquisition
from robotorchan.acquisition.capabilities import (
    AcquisitionPurpose,
    PosteriorRequirement,
)
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY


def test_registry_covers_public_acquisition_classes() -> None:
    helpers = {
        "AcquisitionCapabilities",
        "AcquisitionPurpose",
        "AcquisitionRegistryEntry",
        "PosteriorRequirement",
        "get_acquisition_registry_entry",
        "make_non_gp_acquisition",
        "select_thompson_candidates",
        "validate_non_gp_acquisition",
    }
    expected = set(acquisition.__all__) - helpers
    public_extensions = {
        name
        for name, entry in ACQUISITION_REGISTRY.items()
        if entry.capabilities.purpose is AcquisitionPurpose.ACTIVE_LEARNING
    }
    assert public_extensions == expected


def test_single_task_gp_supports_scalar_active_learning() -> None:
    result = check_model_acquisition_compatibility("SingleTaskGP", "Straddle")
    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.compatible


def test_structured_output_requires_scalarization() -> None:
    result = check_model_acquisition_compatibility("HigherOrderGP", "PosteriorVariance")
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "structured-output posteriors require scalarization" in result.reasons


def test_epig_rejects_multitask_model() -> None:
    result = check_model_acquisition_compatibility(
        "KroneckerMultiTaskGP",
        "ExpectedPredictiveInformationGain",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires a single-output posterior" in result.reasons


def test_random_forest_rejects_variance_active_learning() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "PosteriorVariance",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support ensemble posteriors" in result.reasons


def test_registry_keys_match_acquisition_names() -> None:
    entries = ACQUISITION_REGISTRY.values()
    registered_names = {entry.acquisition_name for entry in entries}
    assert registered_names == set(ACQUISITION_REGISTRY)


def test_non_gp_model_accepts_registered_monte_carlo_bo_acquisition() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qLogExpectedImprovement",
    )
    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.compatible


def test_qkg_rejects_multi_output_model() -> None:
    result = check_model_acquisition_compatibility(
        "KroneckerMultiTaskGP",
        "qKnowledgeGradient",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support multi-output posteriors" in result.reasons


def test_epig_rejects_model_list_multi_output_model() -> None:
    result = check_model_acquisition_compatibility(
        "ModelListGP",
        "ExpectedPredictiveInformationGain",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires a single-output posterior" in result.reasons


def test_ngboost_supports_marginal_variance_active_learning() -> None:
    result = check_model_acquisition_compatibility("NGBoostSurrogate", "PosteriorVariance")
    assert result.status is CompatibilityStatus.COMPATIBLE


def test_ngboost_rejects_joint_gaussian_information_gain() -> None:
    result = check_model_acquisition_compatibility(
        "NGBoostSurrogate",
        "ExpectedPredictiveInformationGain",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires a joint Gaussian posterior" in result.reasons


def test_registry_capabilities_are_internally_consistent() -> None:
    for entry in ACQUISITION_REGISTRY.values():
        capabilities = entry.capabilities

        assert capabilities.max_q is None or capabilities.max_q >= 1
        if capabilities.requires_single_output:
            assert not capabilities.supports_multi_output
        if capabilities.supports_multi_objective:
            assert capabilities.supports_multi_output
        if capabilities.one_shot:
            assert capabilities.requires_fantasize
        if capabilities.monte_carlo:
            assert capabilities.posterior_requirement is PosteriorRequirement.POSTERIOR_SAMPLES


def test_static_compatibility_does_not_imply_runtime_validation() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "qLogExpectedImprovement",
    )

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.reasons == ()
