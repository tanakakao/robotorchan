"""Final public-contract audit for expressive surrogate models."""

import robotorchan.models as models

EXPRESSIVE_PUBLIC_MODELS = {
    "InfiniteWidthBNNGP",
    "JointEncoderGP",
    "SingleTaskDeepGP",
    "SpectralMixtureGP",
}


def test_expressive_models_are_public_exports() -> None:
    assert set(models.__all__) >= EXPRESSIVE_PUBLIC_MODELS
    for name in EXPRESSIVE_PUBLIC_MODELS:
        assert getattr(models, name).__name__ == name


def test_expressive_models_do_not_expose_compatibility_aliases() -> None:
    forbidden = {
        "DeepKernelGP",
        "DeepGaussianProcess",
        "InfiniteWidthGP",
        "SpectralGP",
    }
    assert forbidden.isdisjoint(models.__all__)
