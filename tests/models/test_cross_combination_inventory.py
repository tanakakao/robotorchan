"""Cross-combination inventory checks for intentionally supported model axes."""

import robotorchan.models as MODELS

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

IMPLEMENTED_EXPRESSIVE_MIXED_MULTITASK = {
    "MixedMultiTaskDeepGP",
    "MixedInfiniteWidthBNNMultiTaskGP",
    "MixedSpectralMixtureMultiTaskGP",
}


def test_expressive_single_mixed_and_multitask_axes_are_explicit() -> None:
    exported = set(MODELS.__all__)
    for family in EXPRESSIVE_COVERAGE.values():
        assert set(family.values()) <= exported


def test_expressive_mixed_multitask_cross_products_are_explicit_exports() -> None:
    exported = set(MODELS.__all__)
    assert IMPLEMENTED_EXPRESSIVE_MIXED_MULTITASK <= exported
