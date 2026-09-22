"""Regression tests for explicit runtime capability metadata."""

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
        assert MODEL_REGISTRY[name].capabilities.input_type.value == "mixed"


def test_unaudited_reduced_fantasy_support_stays_disabled() -> None:
    for name in (
        "ReducedGP",
        "PLSGP",
        "RandomProjectionGP",
        "MixedReducedGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
    ):
        assert not MODEL_REGISTRY[name].capabilities.supports_fantasize

    assert MODEL_REGISTRY["PCAGP"].capabilities.supports_fantasize
    assert MODEL_REGISTRY["MixedPCAGP"].capabilities.supports_fantasize
