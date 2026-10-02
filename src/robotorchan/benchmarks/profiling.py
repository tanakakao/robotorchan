"""Lightweight performance and CPU memory profiling utilities."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ProfileResult:
    """Measured wall time and Python CPU peak memory for one callable."""

    wall_time_seconds: float
    peak_memory_bytes: int


def profile_callable(function: Callable[[], T]) -> tuple[T, ProfileResult]:
    """Run a callable and measure wall time plus Python CPU peak memory."""
    if tracemalloc.is_tracing():
        raise RuntimeError("profile_callable requires tracemalloc to be inactive.")
    tracemalloc.start()
    start = perf_counter()
    try:
        value = function()
        elapsed = perf_counter() - start
        _, peak_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, ProfileResult(
        wall_time_seconds=elapsed,
        peak_memory_bytes=peak_memory,
    )
