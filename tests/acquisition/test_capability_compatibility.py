"""Tests for acquisition capability metadata and model compatibility."""

import robotorchan.acquisition as acquisition
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY


def test_registry_covers_public_acquisition_classes() -> None:
    helpers = {
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


def test_non_gp_model_requires_mc_acquisition_path() -> None:
    result = check_model_acquisition_compatibility(
        "RandomForestSurrogate",
        "PosteriorVariance",
    )
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "non-GP empirical ensembles require BoTorch Monte Carlo acquisitions" in result.reasons


def test_registry_keys_match_acquisition_names() -> None:
    entries = ACQUISITION_REGISTRY.values()
    registered_names = {entry.acquisition_name for entry in entries}
    assert registered_names == set(ACQUISITION_REGISTRY)
