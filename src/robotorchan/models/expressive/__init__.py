"""Expressive surrogate models with learned or flexible representations."""

from robotorchan.models.expressive.deep_gp import (
    MixedMultiTaskDeepGP, MixedSingleTaskDeepGP, MultiTaskDeepGP, SingleTaskDeepGP,
)
from robotorchan.models.expressive.infinite_width_bnn import (
    InfiniteWidthBNNGP, InfiniteWidthBNNMultiTaskGP,
    MixedInfiniteWidthBNNGP, MixedInfiniteWidthBNNMultiTaskGP,
)
from robotorchan.models.expressive.spectral_mixture import (
    MixedSpectralMixtureGP, MixedSpectralMixtureMultiTaskGP,
    SpectralMixtureGP, SpectralMixtureMultiTaskGP,
)

__all__ = [
    "InfiniteWidthBNNGP", "InfiniteWidthBNNMultiTaskGP",
    "MixedInfiniteWidthBNNGP", "MixedInfiniteWidthBNNMultiTaskGP",
    "MixedMultiTaskDeepGP", "MixedSingleTaskDeepGP",
    "MixedSpectralMixtureGP", "MixedSpectralMixtureMultiTaskGP",
    "MultiTaskDeepGP", "SingleTaskDeepGP",
    "SpectralMixtureGP", "SpectralMixtureMultiTaskGP",
]
