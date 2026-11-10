"""Focused diversity checks for independent mixed preference pairs."""

import torch

from robotorchan.optim.preference_loop import _pair_is_distinct, _select_diverse_pair


def test_pair_duplicates_are_order_invariant() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    pair = torch.tensor([[0.2, 0.0], [0.8, 1.0]], dtype=torch.double)
    assert not _pair_is_distinct(pair.flip(0), [pair], bounds, [1], 0.0)
    different = torch.tensor([[0.2, 1.0], [0.8, 0.0]], dtype=torch.double)
    assert _pair_is_distinct(different, [pair], bounds, [1], 0.0)


def test_minimum_pair_distance() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    pair = torch.tensor([[0.2, 0.0], [0.8, 1.0]], dtype=torch.double)
    nearby = torch.tensor([[0.21, 0.0], [0.81, 1.0]], dtype=torch.double)
    assert not _pair_is_distinct(nearby, [pair], bounds, [1], 0.05)
    assert _pair_is_distinct(nearby, [pair], bounds, [1], 0.005)


def test_diverse_fallback_raises_for_exhausted_discrete_space() -> None:
    bounds = torch.tensor([[0.0], [0.0]], dtype=torch.double)
    pair = torch.zeros(2, 1, dtype=torch.double)
    try:
        _select_diverse_pair(None, bounds, [{0: 0.0}], [0], [pair], 0.0, 8)
    except RuntimeError as error:
        assert "No distinct preference pair" in str(error)
    else:
        raise AssertionError("Expected exhausted discrete space to raise")
