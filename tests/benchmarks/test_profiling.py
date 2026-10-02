"""Tests for lightweight performance profiling utilities."""

import tracemalloc

import pytest

from robotorchan.benchmarks import profile_callable


def test_profile_callable_returns_value_and_nonnegative_metrics() -> None:
    value, result = profile_callable(lambda: [index * index for index in range(128)])

    assert value[3] == 9
    assert result.wall_time_seconds >= 0.0
    assert result.peak_memory_bytes > 0


def test_profile_callable_restores_tracemalloc_state() -> None:
    assert not tracemalloc.is_tracing()

    profile_callable(lambda: bytearray(1024))

    assert not tracemalloc.is_tracing()


def test_profile_callable_rejects_nested_tracemalloc_session() -> None:
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="tracemalloc"):
            profile_callable(lambda: None)
    finally:
        tracemalloc.stop()
