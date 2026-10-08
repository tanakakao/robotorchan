"""Explicit registry of benchmark problem factories."""

from __future__ import annotations

from collections.abc import Callable

from robotorchan.benchmarks.problem import BenchmarkProblem

ProblemFactory = Callable[[], BenchmarkProblem]


class BenchmarkProblemRegistry:
    """Register and instantiate benchmark problems without shared mutable instances."""

    def __init__(self) -> None:
        self._factories: dict[str, ProblemFactory] = {}

    def register(self, name: str, factory: ProblemFactory) -> None:
        """Register a zero-argument factory under a unique name."""
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("Problem name must be a nonempty, trimmed string.")
        if not callable(factory):
            raise TypeError("Problem factory must be callable.")
        if name in self._factories:
            raise ValueError(f"Benchmark problem already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> BenchmarkProblem:
        """Create a fresh problem from its registered factory."""
        try:
            factory = self._factories[name]
        except KeyError as error:
            raise KeyError(f"Unknown benchmark problem: {name}") from error
        problem = factory()
        if not isinstance(problem, BenchmarkProblem):
            raise TypeError(f"Factory for {name} did not return BenchmarkProblem.")
        if problem.name != name:
            raise ValueError(f"Factory returned problem {problem.name!r}, expected {name!r}.")
        return problem

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic alphabetical order."""
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        """Check whether a problem name is registered."""
        return isinstance(name, str) and name in self._factories


default_problem_registry = BenchmarkProblemRegistry()


def register_problem(name: str, factory: ProblemFactory) -> None:
    """Register a problem factory in the default registry."""
    default_problem_registry.register(name, factory)


def get_problem(name: str) -> BenchmarkProblem:
    """Instantiate a registered problem by name."""
    return default_problem_registry.create(name)


def list_problems() -> tuple[str, ...]:
    """List the names in the default problem registry."""
    return default_problem_registry.names()
