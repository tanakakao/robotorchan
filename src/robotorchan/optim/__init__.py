"""Acquisition-function optimization and search-strategy interfaces."""

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.embedding import REMBOStrategy
from robotorchan.optim.latent import (
    LatentReconstruction,
    LatentSpaceStrategy,
    PCAReconstruction,
    RandomProjectionReconstruction,
)
from robotorchan.optim.original import OriginalSpaceStrategy
from robotorchan.optim.random import RandomSearchStrategy

__all__ = [
    "LatentReconstruction",
    "LatentSpaceStrategy",
    "OriginalSpaceStrategy",
    "PCAReconstruction",
    "REMBOStrategy",
    "RandomProjectionReconstruction",
    "RandomSearchStrategy",
    "SearchResult",
    "SearchStrategy",
]
