"""Focused tests for transfer warm-start initial point selection."""

import pytest
import torch

from robotorchan.optim.transfer.warm_start import select_warm_start_points


def test_source_ranking_and_no_target_value_transfer():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    source_x = torch.tensor([[0.2], [0.7], [0.2], [1.5]], dtype=torch.double)
    source_y = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.double)
    result = select_warm_start_points(bounds, 3, source_x=source_x, source_y=source_y, seed=7)
    assert result.shape == (3, 1)
    assert torch.equal(result[:2], torch.tensor([[0.2], [0.7]], dtype=torch.double))
    assert bool(((result >= 0) & (result <= 1)).all())
    assert len(torch.unique(result, dim=0)) == 3


def test_minimization():
    bounds = torch.tensor([[0.0], [1.0]])
    result = select_warm_start_points(
        bounds,
        1,
        source_x=torch.tensor([[0.2], [0.8]]),
        source_y=torch.tensor([2.0, -1.0]),
        maximize=False,
    )
    assert result.item() == pytest.approx(0.8)


def test_requires_exactly_one_ranking_source():
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="exactly one"):
        select_warm_start_points(bounds, 1, source_x=torch.tensor([[0.2]]))


def test_multiple_sources_use_within_task_ranks():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    result = select_warm_start_points(
        bounds,
        4,
        source_datasets=[
            (
                torch.tensor([[0.1], [0.2]], dtype=torch.double),
                torch.tensor([10.0, 20.0], dtype=torch.double),
            ),
            (
                torch.tensor([[0.3], [0.4]], dtype=torch.double),
                torch.tensor([1000.0, 2000.0], dtype=torch.double),
            ),
        ],
    )
    expected = torch.tensor([[0.2], [0.4], [0.1], [0.3]], dtype=torch.double)
    assert torch.equal(result, expected)


def test_multiple_sources_deduplicate_and_fill():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    result = select_warm_start_points(
        bounds,
        3,
        source_datasets=[
            (torch.tensor([[0.5]]), torch.tensor([1.0])),
            (torch.tensor([[0.5]]), torch.tensor([100.0])),
        ],
        seed=7,
    )
    assert result.shape == (3, 1)
    assert result[0].item() == pytest.approx(0.5)
    assert len(torch.unique(result, dim=0)) == 3


def test_multiple_sources_reject_mixed_inputs():
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="cannot be combined"):
        select_warm_start_points(
            bounds,
            1,
            source_x=torch.tensor([[0.2]]),
            source_datasets=[(torch.tensor([[0.3]]), torch.tensor([1.0]))],
        )
