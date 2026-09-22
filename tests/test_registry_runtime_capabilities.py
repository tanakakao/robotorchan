"""Regression tests for explicit runtime capability metadata."""

from robotorchan.models.capabilities import InputType, PosteriorSamplingType
from robotorchan.models.registry import MODEL_REGISTRY


def test_runtime_capabilities_are_not_blanket_enabled() -> None:
    random_forest = MODEL_REGISTRY["RandomForestSurrogate"].capabilities
    variational = MODEL_REGISTRY["MixedSingleTaskVariationalGP"].capabilities

    assert random_forest.supports_posterior_samples
    assert not random_forest.supports_fantasize
    assert variational.supports_posterior_samples
    assert not variational.supports_fantasize


def test_runtime_validated_exact_models_keep_required_capabilities() -> None:
    for name in (
        "SingleTaskGP",
        "MixedSingleTaskGP",
        "KroneckerMultiTaskGP",
        "SingleTaskMultiFidelityGP",
        "MixedSingleTaskMultiFidelityGP",
    ):
        capabilities = MODEL_REGISTRY[name].capabilities
        assert capabilities.supports_posterior_samples
        assert capabilities.supports_fantasize


def test_mixed_reduced_models_keep_mixed_input_metadata() -> None:
    for name in (
        "MixedReducedGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
    ):
        assert MODEL_REGISTRY[name].capabilities.input_type is InputType.MIXED


def test_runtime_validated_reduced_models_support_fantasize() -> None:
    for name in (
        "ReducedGP",
        "PCAGP",
        "PLSGP",
        "RandomProjectionGP",
        "MixedReducedGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
    ):
        assert MODEL_REGISTRY[name].capabilities.supports_fantasize


def test_posterior_sampling_type_distinguishes_gaussian_and_ensemble_models() -> None:
    assert (
        MODEL_REGISTRY["SingleTaskGP"].capabilities.posterior_sampling_type
        is PosteriorSamplingType.GAUSSIAN
    )
    assert (
        MODEL_REGISTRY["RandomForestSurrogate"].capabilities.posterior_sampling_type
        is PosteriorSamplingType.ENSEMBLE
    )


def test_sampling_type_is_none_when_posterior_sampling_is_disabled() -> None:
    for entry in MODEL_REGISTRY.values():
        capabilities = entry.capabilities
        if not capabilities.supports_posterior_samples:
            assert capabilities.posterior_sampling_type is PosteriorSamplingType.NONE
