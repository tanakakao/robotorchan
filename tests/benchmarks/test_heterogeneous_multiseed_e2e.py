"""Phase 15: paired multi-seed benchmark statistics."""

import pytest
import torch

from robotorchan.benchmarks.heterogeneous_multiseed import (
    compare_strategies,
    summarize_trajectories,
)


def test_paired_multi_seed_trajectories() -> None:
    """Collect equal-budget paired trajectories and verify statistical summaries."""
    results = compare_strategies((1501, 1502), initial_points=8, steps=2)
    assert set(results) == {"random", "sobol", "qei"}
    starts = []
    for curves in results.values():
        assert curves.shape == (2, 3)
        assert torch.isfinite(curves).all()
        assert (curves[:, 1:] >= curves[:, :-1]).all()
        mean, std, sem = summarize_trajectories(curves)
        assert mean.shape == std.shape == sem.shape == (3,)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(std).all()
        assert torch.isfinite(sem).all()
        assert (std >= 0).all()
        torch.testing.assert_close(sem, std / 2**0.5)
        starts.append(curves[:, 0])
    for start in starts[1:]:
        torch.testing.assert_close(start, starts[0])


def test_summary_matches_hand_calculation() -> None:
    """Use sample rather than population standard deviation across seeds."""
    curves = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.double)
    mean, std, sem = summarize_trajectories(curves)
    torch.testing.assert_close(mean, torch.tensor([2.0, 4.0], dtype=torch.double))
    torch.testing.assert_close(std, torch.full((2,), 2**0.5, dtype=torch.double))
    torch.testing.assert_close(sem, torch.ones(2, dtype=torch.double))


@pytest.mark.parametrize(
    "curves",
    [
        torch.ones(1, 3),
        torch.ones(2, 0),
        torch.ones(2, 3, 1),
        torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
    ],
)
def test_invalid_summary_inputs(curves: torch.Tensor) -> None:
    """Reject insufficient samples, invalid dimensions, and nonfinite values."""
    with pytest.raises(ValueError):
        summarize_trajectories(curves)


def test_duplicate_seed_rejected() -> None:
    """Paired statistical trials require distinct seeds."""
    with pytest.raises(ValueError, match="distinct"):
        compare_strategies((42, 42))
