"""Regression tests against the current BoTorch BAxUS reference schedule."""

from dataclasses import replace

import pytest
import torch

from robotorchan.optim import BAxUSState, BAxUSStrategy


def _bounds(dim: int) -> torch.Tensor:
    return torch.stack([torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)])


def test_500d_reference_state_matches_botorch_tutorial() -> None:
    state = BAxUSState(dim=500, eval_budget=500)

    assert state.n_splits == 4
    assert state.initial_target_dim == 2
    assert state.target_dim == 2
    assert state.split_budget == 1
    assert state.failure_tolerance == 1


@pytest.mark.parametrize(
    ("target_dim", "expected_split_budget", "expected_failure_tolerance"),
    [
        (2, 1, 1),
        (8, 6, 1),
        (32, 23, 3),
        (128, 94, 15),
        (500, 367, 500),
    ],
)
def test_500d_reference_budget_schedule(
    target_dim: int,
    expected_split_budget: int,
    expected_failure_tolerance: int,
) -> None:
    state = BAxUSState(dim=500, eval_budget=500, target_dim=target_dim)

    assert state.split_budget == expected_split_budget
    assert state.failure_tolerance == expected_failure_tolerance


def test_nested_expansion_reaches_reference_dimensions() -> None:
    strategy = BAxUSStrategy(_bounds(500), state=BAxUSState(dim=500, eval_budget=500), seed=7)
    dimensions = [strategy.target_dim]

    while strategy.target_dim < strategy.input_dim:
        strategy.state = replace(
            strategy.state,
            length=strategy.state.length_min / 2,
            restart_triggered=True,
        )
        assert strategy.expand_subspace()
        dimensions.append(strategy.target_dim)

    assert dimensions == [2, 8, 32, 128, 500]
