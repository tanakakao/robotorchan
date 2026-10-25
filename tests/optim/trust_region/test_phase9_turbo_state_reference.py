"""Phase 9 reference checks for TuRBO trust-region state transitions."""

import math

import torch

from robotorchan.optim.trust_region.turbo import (
    TuRBOState,
    restart_turbo_state,
    update_turbo_state,
)


def test_turbo_success_threshold_and_capped_expansion() -> None:
    """A third meaningful improvement doubles length, capped at length_max."""
    state = TuRBOState(
        dim=2,
        batch_size=1,
        length=0.9,
        length_max=1.6,
        success_tolerance=3,
        best_value=0.0,
    )
    for step, value in enumerate([0.1, 0.2, 0.3], start=1):
        state = update_turbo_state(state, torch.tensor([value], dtype=torch.float64))
        assert math.isclose(state.length, 0.9 if step < 3 else 1.6)
        assert state.success_counter == (step if step < 3 else 0)
        assert state.failure_counter == 0
        assert math.isclose(state.best_value, value)


def test_turbo_failure_halving_and_restart_preserves_best() -> None:
    """Two consecutive failures halve length and crossing minimum triggers restart."""
    state = TuRBOState(
        dim=2,
        batch_size=1,
        length=0.15,
        length_min=0.1,
        failure_tolerance=2,
        best_value=2.0,
        observed_best_value=2.0,
    )
    state = update_turbo_state(state, torch.tensor([1.0], dtype=torch.float64))
    assert state.failure_counter == 1
    assert math.isclose(state.length, 0.15)
    state = update_turbo_state(state, torch.tensor([1.5], dtype=torch.float64))
    assert state.failure_counter == 0
    assert math.isclose(state.length, 0.075)
    assert state.restart_triggered
    assert state.best_value == 2.0

    restarted = restart_turbo_state(state, length=0.8)
    assert restarted.restart_count == 1
    assert not restarted.restart_triggered
    assert math.isclose(restarted.length, 0.8)
    assert restarted.best_value == 2.0
    assert restarted.observed_best_value == 2.0


def test_turbo_decision_values_are_distinct_from_observed_best() -> None:
    """Denoised decision utilities control success while raw observations are tracked."""
    state = TuRBOState(dim=1, batch_size=1, best_value=1.0, observed_best_value=1.0)
    updated = update_turbo_state(
        state,
        torch.tensor([3.0], dtype=torch.float64),
        state_values=torch.tensor([0.5], dtype=torch.float64),
    )
    assert updated.failure_counter == 1
    assert updated.success_counter == 0
    assert updated.best_value == 1.0
    assert updated.observed_best_value == 3.0
