"""Regression tests for explicit runtime capability metadata."""

from robotorchan.models.registry import MODEL_REGISTRY


def test_runtime_capabilities_are_not_blanket_enabled() -> None:
    random_forest = MODEL_REGISTRY["RandomForestSurrogate"].capabilities
    variational = MODEL_REGISTRY["MixedSingleTaskVariationalGP"].capabilities

    assert not random_forest.supports_posterior_samples
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
