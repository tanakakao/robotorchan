"""Cross-combination inventory checks for intentionally supported model axes."""

import importlib  # noqa: I001

MODELS = importlib.import_module("robotorchan.models")

EXPRESSIVE_COVERAGE = {
    "deep_gp": {
        "single": "SingleTaskDeepGP",
        "mixed": "MixedSingleTaskDeepGP",
        "multitask": "MultiTaskDeepGP",
    },
    "infinite_width_bnn": {
        "single": "InfiniteWidthBNNGP",
        "mixed": "MixedInfiniteWidthBNNGP",
        "multitask": "InfiniteWidthBNNMultiTaskGP",
    },
    "spectral_mixture": {
        "single": "SpectralMixtureGP",
        "mixed": "MixedSpectralMixtureGP",
        "multitask": "SpectralMixtureMultiTaskGP",
    },
}

INTENTIONALLY_DEFERRED_EXPRESSIVE = {
    "MixedMultiTaskDeepGP",
    "MixedInfiniteWidthBNNMultiTaskGP",
    "MixedSpectralMixtureMultiTaskGP",
}


def test_expressive_single_mixed_and_multitask_axes_are_explicit() -> None:
    exported = set(MODELS.__all__)
    for family in EXPRESSIVE_COVERAGE.values():
        assert set(family.values()) <= exported


def test_expressive_mixed_multitask_cross_products_are_not_accidental_exports() -> None:
    exported = set(MODELS.__all__)
    assert exported.isdisjoint(INTENTIONALLY_DEFERRED_EXPRESSIVE)
