"""Acquisition-guided multistart local search on finite Hamming spaces."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.constraints.evaluation import candidate_is_feasible
from robotorchan.optim.domains.combinatorial import (
    _validate_points,
    hamming_distance,
    hamming_neighbors,
)
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class HammingLocalSearchStrategy(SearchStrategy):
    """Best-improvement radius-one search from user-supplied restart points.

    Maximizes a BoTorch acquisition function with q=1. All initial points
    must be valid discrete candidates. Infeasible or excluded starts are
    skipped; the best feasible local optimum is returned.
    """

    def __init__(
        self,
        space: MixedVariableSpace,
        initial_points: Tensor,
        *,
        constraints: CandidateConstraints | None = None,
        excluded: Iterable[tuple[float, ...]] = (),
        max_steps: int = 100,
        trust_region_center: Tensor | None = None,
        trust_region_radius: int | None = None,
    ) -> None:
        super().__init__(space.bounds)
        if space.continuous_dims:
            raise ValueError("Hamming search requires a fully discrete MixedVariableSpace.")
        if initial_points.ndim != 2 or initial_points.shape[1] != space.input_dim:
            raise ValueError("initial_points must have shape [num_restarts, d].")
        if initial_points.shape[0] == 0:
            raise ValueError("initial_points must contain at least one restart.")
        if (
            initial_points.dtype != space.bounds.dtype
            or initial_points.device != space.bounds.device
        ):
            raise ValueError("initial_points must match the space bounds dtype and device.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        _validate_points(initial_points, space)
        self.space = space
        self.initial_points = initial_points.detach().clone()
        self.constraints = constraints
        self.excluded = set(excluded)
        self.max_steps = max_steps
        if (trust_region_center is None) != (trust_region_radius is None):
            raise ValueError("Trust-region center and radius must be provided together.")
        if trust_region_center is not None:
            if trust_region_center.shape != (space.input_dim,):
                raise ValueError("trust_region_center must have shape [d].")
            if (
                trust_region_center.dtype != space.bounds.dtype
                or trust_region_center.device != space.bounds.device
            ):
                raise ValueError("Trust-region center must match bounds dtype and device.")
            _validate_points(trust_region_center, space)
            if not 1 <= trust_region_radius <= space.input_dim:
                raise ValueError("Trust-region radius must be between 1 and d.")
            self.trust_region_center = trust_region_center.detach().clone()
        else:
            self.trust_region_center = None
        self.trust_region_radius = trust_region_radius

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Return the best feasible q=1 candidate across local restarts."""
        if q != 1:
            raise NotImplementedError("Hamming local search currently supports q=1 only.")

        def evaluate(points: Tensor) -> Tensor:
            with torch.no_grad():
                values = acq_function(points.unsqueeze(-2))
            if values.numel() != points.shape[0]:
                raise ValueError("Acquisition must return one scalar per q=1 candidate.")
            return values.reshape(-1)

        best_point: Tensor | None = None
        best_value: Tensor | None = None
        evaluations = 0
        steps = 0
        starts = self.initial_points
        if self.constraints is not None and self.constraints.has_constraints:
            starts = starts[candidate_is_feasible(starts.unsqueeze(-2), self.constraints)]
        if self.trust_region_center is not None:
            starts = starts[
                hamming_distance(starts, self.trust_region_center, self.space)
                <= self.trust_region_radius
            ]
        for start in starts:
            if tuple(start.tolist()) in self.excluded:
                continue
            current = start.clone()
            current_value = evaluate(current.unsqueeze(0))[0]
            evaluations += 1
            for _ in range(self.max_steps):
                neighbors = hamming_neighbors(
                    current, self.space, constraints=self.constraints, excluded=self.excluded
                )
                if self.trust_region_center is not None:
                    neighbors = neighbors[
                        hamming_distance(neighbors, self.trust_region_center, self.space)
                        <= self.trust_region_radius
                    ]
                if neighbors.shape[0] == 0:
                    break
                values = evaluate(neighbors)
                evaluations += neighbors.shape[0]
                index = int(torch.argmax(values).item())
                if not bool(values[index] > current_value):
                    break
                current = neighbors[index]
                current_value = values[index]
                steps += 1
            if best_value is None or bool(current_value > best_value):
                best_point = current
                best_value = current_value
        if best_point is None or best_value is None:
            raise ValueError("No feasible, non-excluded initial point was provided.")
        return SearchResult(
            candidates=best_point.unsqueeze(0),
            acquisition_value=best_value.reshape(()),
            metadata={"evaluations": evaluations, "steps": steps},
        )
