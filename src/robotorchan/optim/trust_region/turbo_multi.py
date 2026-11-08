"""Independent region storage for multi-trust-region TuRBO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from robotorchan.optim.trust_region.turbo import TuRBOState, turbo_trust_region_bounds


@dataclass
class TuRBORegion:
    """One active region with independent observations and surrogate reference."""

    center: Tensor
    state: TuRBOState
    train_X: Tensor
    train_Y: Tensor
    model: Any | None = None
    dimension_weights: Tensor | None = None
    history_X: Tensor = field(init=False)
    history_Y: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.center = self.center.detach().clone()
        self.train_X = self.train_X.detach().clone()
        self.train_Y = self.train_Y.detach().clone()
        self.history_X = self.train_X.clone()
        self.history_Y = self.train_Y.clone()


class TuRBOMultiRegionState:
    """Manage region-local data without implementing cross-region selection.

    Candidate generation, observation attribution and restart policy are
    introduced in subsequent TuRBO-m phases.
    """

    def __init__(
        self,
        bounds: Tensor,
        centers: Tensor,
        *,
        batch_size: int = 1,
        states: list[TuRBOState] | None = None,
    ) -> None:
        if bounds.ndim != 2 or bounds.shape[0] != 2 or not torch.all(bounds[0] < bounds[1]):
            raise ValueError("bounds must have shape [2, d] with lower < upper.")
        if centers.ndim != 2 or centers.shape[-1] != bounds.shape[-1]:
            raise ValueError("centers must have shape [m, d].")
        if centers.shape[0] < 1 or not torch.isfinite(centers).all():
            raise ValueError("centers must contain at least one finite center.")
        if torch.any(centers < bounds[0]) or torch.any(centers > bounds[1]):
            raise ValueError("centers must lie inside bounds.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if states is not None and len(states) != centers.shape[0]:
            raise ValueError("states must contain one state per center.")
        self.bounds = bounds.detach().clone()
        self.regions: list[TuRBORegion] = []
        for index, center in enumerate(centers):
            state = (
                TuRBOState(dim=bounds.shape[-1], batch_size=batch_size)
                if states is None
                else states[index]
            )
            if state.dim != bounds.shape[-1]:
                raise ValueError("each state.dim must match the input dimension.")
            empty_X = bounds.new_empty((0, bounds.shape[-1]))
            empty_Y = bounds.new_empty((0, 1))
            self.regions.append(TuRBORegion(center, state, empty_X, empty_Y))

    @property
    def n_trust_regions(self) -> int:
        return len(self.regions)

    def region(self, region_id: int) -> TuRBORegion:
        return self.regions[region_id]

    def trust_region_bounds(self, region_id: int) -> Tensor:
        region = self.region(region_id)
        return turbo_trust_region_bounds(
            region.center,
            self.bounds,
            length=region.state.length,
            dimension_weights=region.dimension_weights,
        )

    def set_model(self, region_id: int, model: Any) -> None:
        self.region(region_id).model = model

    def set_region_data(self, region_id: int, X: Tensor, Y: Tensor) -> None:
        """Replace active local training data; append-only history is separate."""
        if X.ndim != 2 or X.shape[-1] != self.bounds.shape[-1]:
            raise ValueError("X must have shape [n, d].")
        if Y.shape != (X.shape[0], 1):
            raise ValueError("Y must have shape [n, 1].")
        if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
            raise ValueError("training data must be finite.")
        if torch.any(self.bounds[0] > X) or torch.any(self.bounds[1] < X):
            raise ValueError("training inputs must lie inside bounds.")
        region = self.region(region_id)
        region.train_X = X.to(self.bounds).detach().clone()
        region.train_Y = Y.to(self.bounds).detach().clone()

    def append_observations(self, region_id: int, X: Tensor, Y: Tensor) -> None:
        """Append attributed observations without updating TuRBO counters."""
        region = self.region(region_id)
        previous_X, previous_Y = region.train_X, region.train_Y
        self.set_region_data(region_id, X, Y)
        new_X, new_Y = region.train_X, region.train_Y
        region.train_X = torch.cat((previous_X, new_X))
        region.train_Y = torch.cat((previous_Y, new_Y))
        region.history_X = torch.cat((region.history_X, new_X))
        region.history_Y = torch.cat((region.history_Y, new_Y))
