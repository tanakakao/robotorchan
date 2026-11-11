"""Focused tests for discrete Hamming trust-region search."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import (
    DiscreteTrustRegionState,
    DiscreteTrustRegionStrategy,
    MixedVariableSpace,
    update_discrete_trust_region,
)


class TargetAcquisition(AcquisitionFunction):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__(model=None)
        self.target = target

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -(X.squeeze(-2) - self.target).square().sum(dim=-1)


def test_hamming_ball_limits_search_and_reports_radius():
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(0, 1, 2))
    center = torch.zeros(3, dtype=torch.double)
    acquisition = TargetAcquisition(torch.ones(3, dtype=torch.double))
    state = DiscreteTrustRegionState(radius=1, max_radius=3)
    result = DiscreteTrustRegionStrategy(space, center, state).optimize(acquisition)
    assert result.candidates.sum().item() == 1.0
    assert result.metadata["trust_region_radius"] == 1
    with pytest.raises(NotImplementedError, match="q=1"):
        DiscreteTrustRegionStrategy(space, center, state).optimize(acquisition, q=2)


def test_adaptation_and_restart_signal():
    state = DiscreteTrustRegionState(
        radius=2, min_radius=1, max_radius=3, success_tolerance=2, failure_tolerance=2
    )
    state = update_discrete_trust_region(state, improved=True)
    assert state.radius == 2
    state = update_discrete_trust_region(state, improved=True)
    assert state.radius == 3
    state = update_discrete_trust_region(state, improved=False)
    state = update_discrete_trust_region(state, improved=False)
    assert state.radius == 2
    state = update_discrete_trust_region(state, improved=False)
    state = update_discrete_trust_region(state, improved=False)
    assert state.radius == 1
    state = update_discrete_trust_region(state, improved=False)
    state = update_discrete_trust_region(state, improved=False)
    assert state.restart_triggered


def test_reject_invalid_radius():
    space = MixedVariableSpace(torch.tensor([[0.0], [1.0]]), integer_dims=(0,))
    with pytest.raises(ValueError, match="max_radius"):
        DiscreteTrustRegionStrategy(
            space, torch.tensor([0.0]), DiscreteTrustRegionState(max_radius=2)
        )
