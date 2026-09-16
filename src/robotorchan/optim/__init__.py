"""Acquisition-function optimization and search-strategy interfaces."""

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.original import OriginalSpaceStrategy

__all__ = ["OriginalSpaceStrategy", "SearchResult", "SearchStrategy"]
