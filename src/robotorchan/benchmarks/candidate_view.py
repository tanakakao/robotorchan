"""Restricted public metadata for benchmark candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem, Direction, VariableKind


@dataclass(frozen=True)
class CandidateProblemView:
    """Expose public search-space metadata without any ground-truth evaluators."""

    name: str
    bounds: Tensor
    directions: tuple[Direction, ...]
    variable_types: tuple[VariableKind, ...]
    n_constraints: int

    @classmethod
    def from_problem(cls, problem: BenchmarkProblem) -> CandidateProblemView:
        """Build a defensive snapshot of the declared search-space contract."""
        return cls(
            name=problem.name,
            bounds=problem.bounds.detach().clone(),
            directions=problem.directions,
            variable_types=problem.variable_types,
            n_constraints=problem.n_constraints,
        )

    @property
    def dimension(self) -> int:
        """Return the number of search dimensions."""
        return self.bounds.shape[1]

    @property
    def n_objectives(self) -> int:
        """Return the number of objectives."""
        return len(self.directions)
