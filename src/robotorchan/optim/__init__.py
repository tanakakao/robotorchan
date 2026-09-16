"""Acquisition-function optimization and search-strategy interfaces."""

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.original import OriginalSpaceStrategy
from robotorchan.optim.random import RandomSearchStrategy

__all__ = [
    "OriginalSpaceStrategy",
    "RandomSearchStrategy",
    "SearchResult",
    "SearchStrategy",
]
