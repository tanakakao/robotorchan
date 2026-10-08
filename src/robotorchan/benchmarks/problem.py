"""Tensor-native problem contracts for reproducible optimization benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Direction = Literal["maximize", "minimize"]
VariableKind = Literal["continuous", "integer", "categorical"]
Evaluator = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class BenchmarkProblem:
    """Describe ground truth separately from noisy experimental observations.

    Constraints use g(X) >= 0. Evaluators accept (..., q, d) and return
    (..., q, m) for objectives or (..., q, c) for constraints.
    """

    name: str
    bounds: Tensor
    objective: Evaluator
    directions: tuple[Direction, ...]
    variable_types: tuple[VariableKind, ...]
    constraints: Evaluator | None = None
    n_constraints: int = 0
    observe: Evaluator | None = None
    cost: Evaluator | None = None
    optimal_value: Tensor | None = None
    reference_point: Tensor | None = None
    reference_front: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty.")
        bounds = self.bounds
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
            raise ValueError("bounds must have shape (2, d).")
        if not bounds.is_floating_point() or not torch.isfinite(bounds).all():
            raise ValueError("bounds must contain finite floating-point values.")
        if not torch.all(bounds[0] < bounds[1]):
            raise ValueError("Each lower bound must be smaller than its upper bound.")
        if not self.directions or any(d not in ("maximize", "minimize") for d in self.directions):
            raise ValueError("directions must contain maximize/minimize values.")
        if len(self.variable_types) != bounds.shape[1]:
            raise ValueError("variable_types must have one entry per input dimension.")
        if any(t not in ("continuous", "integer", "categorical") for t in self.variable_types):
            raise ValueError("Unknown variable type.")
        if self.n_constraints < 0 or (self.constraints is None) != (self.n_constraints == 0):
            raise ValueError("constraints and n_constraints must agree.")
        m = len(self.directions)
        for name, value, width in (
            ("optimal_value", self.optimal_value, 1),
            ("reference_point", self.reference_point, m),
        ):
            if value is not None and (
                value.ndim != 1
                or value.numel() != width
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be a finite vector of length {width}.")
        if self.optimal_value is not None and m != 1:
            raise ValueError("optimal_value is defined only for single-objective problems.")
        if self.reference_front is not None and (
            self.reference_front.ndim != 2
            or self.reference_front.shape[1] != m
            or not torch.isfinite(self.reference_front).all()
        ):
            raise ValueError("reference_front must be finite with shape (n, m).")

    @property
    def dimension(self) -> int:
        """Number of decision variables."""
        return self.bounds.shape[1]

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.directions)

    def _validate_X(self, X: Tensor) -> None:
        if X.ndim < 2 or X.shape[-1] != self.dimension:
            raise ValueError("X must have shape (..., q, d).")
        if not X.is_floating_point() or not torch.isfinite(X).all():
            raise ValueError("X must contain finite floating-point values.")
        bounds = self.bounds.to(device=X.device, dtype=X.dtype)
        if ((bounds[0] > X) | (bounds[1] < X)).any():
            raise ValueError("X must lie within bounds.")
        for index, kind in enumerate(self.variable_types):
            if kind != "continuous" and not torch.all(X[..., index] == X[..., index].round()):
                raise ValueError(f"Discrete variable {index} must be integral.")

    def _evaluate(self, fn: Evaluator, X: Tensor, width: int, name: str) -> Tensor:
        self._validate_X(X)
        result = fn(X)
        if (
            not isinstance(result, Tensor)
            or result.shape != (*X.shape[:-1], width)
            or not result.is_floating_point()
            or not torch.isfinite(result).all()
        ):
            raise ValueError(f"{name} must return finite floating tensor (..., q, {width}).")
        return result

    def evaluate_truth(self, X: Tensor) -> Tensor:
        """Evaluate deterministic objective truth in native directions."""
        return self._evaluate(self.objective, X, self.n_objectives, "objective")

    def evaluate_observation(self, X: Tensor) -> Tensor:
        """Evaluate observed objectives; use truth when no observation model exists."""
        return self._evaluate(
            self.observe or self.objective, X, self.n_objectives, "observation"
        )

    def evaluate_constraints(self, X: Tensor) -> Tensor:
        """Return constraint residuals g(X) >= 0."""
        if self.constraints is None:
            self._validate_X(X)
            return X.new_empty((*X.shape[:-1], 0))
        return self._evaluate(self.constraints, X, self.n_constraints, "constraints")

    def evaluate_cost(self, X: Tensor) -> Tensor:
        """Return nonnegative evaluation cost with shape (..., q, 1)."""
        if self.cost is None:
            self._validate_X(X)
            return X.new_ones((*X.shape[:-1], 1))
        values = self._evaluate(self.cost, X, 1, "cost")
        if (values < 0).any():
            raise ValueError("Evaluation cost must be nonnegative.")
        return values

    def to_maximization(self, values: Tensor) -> Tensor:
        """Orient objective values for regret and hypervolume calculations."""
        if values.shape[-1] != self.n_objectives:
            raise ValueError("Objective width mismatch.")
        signs = values.new_tensor([1 if d == "maximize" else -1 for d in self.directions])
        return values * signs

    def simple_regret(self, X: Tensor) -> Tensor:
        """Return truth-based simple regret for a single-objective problem.

        The known optimum is required. For constrained problems only feasible
        evaluated points are eligible; regret is +inf when none is feasible.
        """
        if self.n_objectives != 1 or self.optimal_value is None:
            raise ValueError("Simple regret requires a known single-objective optimum.")
        truth = self.to_maximization(self.evaluate_truth(X)).squeeze(-1)
        if self.n_constraints:
            feasible = (self.evaluate_constraints(X) >= 0).all(dim=-1)
            truth = truth.masked_fill(~feasible, -torch.inf)
        best = truth.max(dim=-1).values
        optimum = self.to_maximization(
            self.optimal_value.to(device=X.device, dtype=X.dtype)
        ).squeeze(-1)
        return (optimum - best).clamp_min(0)
