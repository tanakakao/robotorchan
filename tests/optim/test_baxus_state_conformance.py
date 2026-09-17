"""Conformance tests for the BoTorch BAxUS state equations."""

import math

import pytest

from robotorchan.optim import BAxUSState


def _reference_state_values(
    dim: int,
    eval_budget: int,
    *,
    new_bins_on_split: int = 3,
    target_dim: int | None = None,
) -> tuple[int, int, int, int]:
    n_splits = 0 if dim == 1 else round(math.log(dim, new_bins_on_split + 1))
    scale = (new_bins_on_split + 1) ** n_splits
    d_init = min(
        range(1, new_bins_on_split + 1),
        key=lambda value: abs(value * scale - dim),
    )
    current_dim = d_init if target_dim is None else target_dim
    split_budget = round(
        -(new_bins_on_split * eval_budget * current_dim)
        / (d_init * (1 - (new_bins_on_split + 1) ** (n_splits + 1)))
    )
    if current_dim == dim:
        failure_tolerance = current_dim
    else:
        shrink_steps = math.floor(math.log((0.5**7) / 0.8, 0.5))
        failure_tolerance = min(current_dim, max(1, math.floor(split_budget / shrink_steps)))
    return n_splits, d_init, split_budget, failure_tolerance


@pytest.mark.parametrize(
    ("dim", "eval_budget"),
    [(1, 1), (20, 100), (50, 100), (100, 200), (500, 500)],
)
def test_baxus_state_matches_botorch_reference_equations(dim: int, eval_budget: int) -> None:
    state = BAxUSState(dim=dim, eval_budget=eval_budget)
    n_splits, d_init, split_budget, failure_tolerance = _reference_state_values(
        dim, eval_budget
    )

    assert state.n_splits == n_splits
    assert state.initial_target_dim == d_init
    assert state.target_dim == d_init
    assert state.split_budget == split_budget
    assert state.failure_tolerance == failure_tolerance


def test_split_budget_is_not_artificially_clamped() -> None:
    state = BAxUSState(dim=500, eval_budget=1)

    assert state.initial_target_dim == 2
    assert state.n_splits == 4
    assert state.split_budget == 0
    assert state.failure_tolerance == 1


def test_full_dimension_failure_tolerance_matches_target_dimension() -> None:
    state = BAxUSState(dim=20, eval_budget=100, target_dim=20)

    assert state.failure_tolerance == 20
