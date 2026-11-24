"""Construct existing search strategies without wrapping their state or behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

from robotorchan.optim.base import SearchStrategy
from robotorchan.optim.embedding.alebo import ALEBOStrategy
from robotorchan.optim.embedding.baxus import BAxUSStrategy
from robotorchan.optim.embedding.baxus_ts import BAxUSThompsonSamplingStrategy
from robotorchan.optim.embedding.hesbo import HeSBOStrategy
from robotorchan.optim.embedding.rembo import REMBOStrategy
from robotorchan.optim.latent.strategy import LatentSpaceStrategy
from robotorchan.optim.strategies.combinatorial import HammingLocalSearchStrategy
from robotorchan.optim.strategies.mixed import MixedSpaceStrategy
from robotorchan.optim.strategies.original import OriginalSpaceStrategy
from robotorchan.optim.strategies.random import RandomSearchStrategy
from robotorchan.optim.strategies.sobol import SobolSearchStrategy
from robotorchan.optim.strategies.tree import TreeEnsembleSearchStrategy
from robotorchan.optim.trust_region.discrete import DiscreteTrustRegionStrategy
from robotorchan.optim.trust_region.turbo import TuRBOStrategy

_STRATEGIES: dict[str, type[SearchStrategy]] = {
    cls.__name__: cls
    for cls in (
        ALEBOStrategy,
        BAxUSStrategy,
        BAxUSThompsonSamplingStrategy,
        HeSBOStrategy,
        REMBOStrategy,
        LatentSpaceStrategy,
        HammingLocalSearchStrategy,
        MixedSpaceStrategy,
        OriginalSpaceStrategy,
        RandomSearchStrategy,
        SobolSearchStrategy,
        TreeEnsembleSearchStrategy,
        DiscreteTrustRegionStrategy,
        TuRBOStrategy,
    )
}
T = TypeVar("T", bound=SearchStrategy)


def list_search_strategies() -> tuple[str, ...]:
    """List concrete search strategies by their canonical class names."""
    return tuple(sorted(_STRATEGIES))


@overload
def create_search_strategy(strategy: Callable[..., T], /, *args: Any, **kwargs: Any) -> T: ...


@overload
def create_search_strategy(strategy: str, /, *args: Any, **kwargs: Any) -> SearchStrategy: ...


def create_search_strategy(
    strategy: str | Callable[..., T], /, *args: Any, **kwargs: Any
) -> SearchStrategy:
    """Construct a strategy with its original arguments and state contract."""
    if isinstance(strategy, str):
        try:
            constructor = _STRATEGIES[strategy]
        except KeyError as error:
            raise KeyError(
                f"Unknown search strategy {strategy!r}. Use list_search_strategies()."
            ) from error
    elif callable(strategy):
        constructor = strategy
    else:
        raise TypeError("strategy must be a name or callable")
    result = constructor(*args, **kwargs)
    if not isinstance(result, SearchStrategy):
        raise TypeError("strategy constructor must return a SearchStrategy")
    return result
