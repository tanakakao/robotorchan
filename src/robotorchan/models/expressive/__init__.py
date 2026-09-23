"""Expressive surrogate models with learned or flexible representations."""

from robotorchan.models.expressive.deep_gp import (
    MixedMultiTaskDeepGP,
    MixedSingleTaskDeepGP,
    MultiTaskDeepGP,
    SingleTaskDeepGP,
)
from robotorchan.models.expressive.infinite_width_bnn import (
    InfiniteWidthBNNGP,
    InfiniteWidthBNNKroneckerMultiTaskGP,
    InfiniteWidthBNNMultiTaskGP,
    MixedInfiniteWidthBNNGP,
    MixedInfiniteWidthBNNKroneckerMultiTaskGP,
    MixedInfiniteWidthBNNMultiTaskGP,
)
from robotorchan.models.expressive.spectral_mixture import (
    MixedSpectralMixtureGP,
    MixedSpectralMixtureKroneckerMultiTaskGP,
    MixedSpectralMixtureMultiTaskGP,
    SpectralMixtureGP,
    SpectralMixtureKroneckerMultiTaskGP,
    SpectralMixtureMultiTaskGP,
)

__all__ = [
    "InfiniteWidthBNNGP",
    "InfiniteWidthBNNKroneckerMultiTaskGP",
    "InfiniteWidthBNNMultiTaskGP",
    "MixedInfiniteWidthBNNGP",
    "MixedInfiniteWidthBNNKroneckerMultiTaskGP",
    "MixedInfiniteWidthBNNMultiTaskGP",
    "MixedMultiTaskDeepGP",
    "MixedSingleTaskDeepGP",
    "MixedSpectralMixtureGP",
    "MixedSpectralMixtureKroneckerMultiTaskGP",
    "MixedSpectralMixtureMultiTaskGP",
    "MultiTaskDeepGP",
    "SingleTaskDeepGP",
    "SpectralMixtureGP",
    "SpectralMixtureKroneckerMultiTaskGP",
    "SpectralMixtureMultiTaskGP",
]
