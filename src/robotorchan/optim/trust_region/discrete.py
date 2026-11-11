"""Discrete Hamming trust-region state and acquisition search."""

from __future__ import annotations

from dataclasses import dataclass, replace

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.base import SearchResult
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.combinatorial import _validate_points
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.optim.strategies.combinatorial import HammingLocalSearchStrategy


@dataclass(frozen=True)
class DiscreteTrustRegionState:
    """Immutable Hamming-radius adaptation state for sequential optimization."""

    radius: int = 1
    min_radius: int = 1
    max_radius: int = 1
    success_tolerance: int = 3
    failure_tolerance: int = 3
    successes: int = 0
    failures: int = 0
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.min_radius <= self.radius <= self.max_radius:
            raise ValueError("Require 1 <= min_radius <= radius <= max_radius.")
        if self.success_tolerance < 1 or self.failure_tolerance < 1:
            raise ValueError("Success and failure tolerances must be positive.")


def update_discrete_trust_region(
    state: DiscreteTrustRegionState, *, improved: bool
) -> DiscreteTrustRegionState:
    """Expand after repeated improvement; shrink or request restart on failures."""
    if improved:
        successes = state.successes + 1
        if successes >= state.success_tolerance:
            return replace(
                state,
                radius=min(state.max_radius, state.radius + 1),
                successes=0,
                failures=0,
                restart_triggered=False,
            )
        return replace(state, successes=successes, failures=0, restart_triggered=False)
    failures = state.failures + 1
    if failures >= state.failure_tolerance:
        if state.radius == state.min_radius:
            return replace(
                state, successes=0, failures=0, restart_triggered=True
            )
        return replace(
            state, radius=max(state.min_radius, state.radius - 1),
            successes=0, failures=0, restart_triggered=False
        )
    return replace(state, successes=0, failures=failures, restart_triggered=False)


class DiscreteTrustRegionStrategy(HammingLocalSearchStrategy):
    """Optimize q=1 acquisition within a Hamming ball around a fixed center.

    The caller owns the center and state updates between sequential BO steps.
    """

    def __init__(
        self,
        space: MixedVariableSpace,
        center: Tensor,
        state: DiscreteTrustRegionState,
        *,
        initial_points: Tensor | None = None,
        constraints: CandidateConstraints | None = None,
        excluded: tuple[tuple[float, ...], ...] = (),
        max_steps: int = 100,
    ) -> None:
        if center.shape != (space.input_dim,):
            raise ValueError("center must have shape [d].")
        _validate_points(center, space)
        if state.max_radius > space.input_dim:
            raise ValueError("Trust-region max_radius cannot exceed input dimension.")
        if initial_points is None:
            initial_points = center.unsqueeze(0)
        super().__init__(
            space,
            initial_points,
            constraints=constraints,
            excluded=excluded,
            max_steps=max_steps,
            trust_region_center=center,
            trust_region_radius=state.radius,
        )
        self.state = state

    def optimize(
        self, acq_function: AcquisitionFunction, *, q: int = 1
    ) -> SearchResult:
        """Search the current discrete trust region."""
        result = super().optimize(acq_function, q=q)
        return SearchResult(
            candidates=result.candidates,
            acquisition_value=result.acquisition_value,
            metadata={**result.metadata, "trust_region_radius": self.state.radius},
        )
