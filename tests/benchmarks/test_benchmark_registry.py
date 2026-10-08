"""Benchmark problem registry contract tests."""

import pytest
import torch

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import (
    BenchmarkProblemRegistry,
    get_problem,
    list_problems,
)


def _factory(name: str = "quadratic") -> BenchmarkProblem:
    return BenchmarkProblem(
        name=name,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=lambda x: -(x - 0.5).square(),
        directions=("maximize",),
        variable_types=("continuous",),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


def test_registration_lookup_and_deterministic_names() -> None:
    registry = BenchmarkProblemRegistry()
    registry.register("zeta", lambda: _factory("zeta"))
    registry.register("alpha", lambda: _factory("alpha"))
    assert registry.names() == ("alpha", "zeta")
    assert "alpha" in registry
    first = registry.create("alpha")
    second = registry.create("alpha")
    assert first is not second
    assert first.name == "alpha"


def test_duplicate_unknown_and_invalid_factory() -> None:
    registry = BenchmarkProblemRegistry()
    registry.register("quadratic", _factory)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("quadratic", _factory)
    with pytest.raises(KeyError, match="Unknown benchmark"):
        registry.create("missing")
    with pytest.raises(TypeError, match="factory"):
        registry.register("bad", None)
    with pytest.raises(ValueError, match="trimmed"):
        registry.register(" bad ", _factory)


def test_factory_result_validation() -> None:
    registry = BenchmarkProblemRegistry()
    registry.register("wrong", _factory)
    with pytest.raises(ValueError, match="expected"):
        registry.create("wrong")
    registry.register("invalid", lambda: None)
    with pytest.raises(TypeError, match="BenchmarkProblem"):
        registry.create("invalid")


def test_default_registry_api_is_available() -> None:
    assert isinstance(list_problems(), tuple)
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_problem("unregistered-phase-3-example")
