"""Verify seeded curve aggregation and paired comparison behavior."""

import pytest
import torch

from robotorchan.benchmarks.comparison import compare_paired_curves, summarize_curves


def test_summary_mean_std_and_confidence_interval() -> None:
    curves = {
        7: torch.tensor([2.0, 4.0], dtype=torch.double),
        3: torch.tensor([0.0, 2.0], dtype=torch.double),
    }
    summary = summarize_curves(curves)
    assert summary.seeds == (3, 7)
    torch.testing.assert_close(summary.mean, torch.tensor([1.0, 3.0], dtype=torch.double))
    torch.testing.assert_close(summary.std, torch.tensor([2.0**0.5, 2.0**0.5], dtype=torch.double))
    assert (summary.lower < summary.mean).all()
    assert (summary.upper > summary.mean).all()


def test_single_seed_has_zero_interval_width() -> None:
    summary = summarize_curves({2: torch.tensor([1.0, 3.0])})
    assert torch.equal(summary.std, torch.zeros(2))
    assert torch.equal(summary.lower, summary.upper)


def test_paired_differences_preserve_seed_alignment() -> None:
    first = {5: torch.tensor([4.0, 5.0]), 1: torch.tensor([2.0, 3.0])}
    second = {1: torch.tensor([1.0, 2.0]), 5: torch.tensor([2.0, 4.0])}
    comparison = compare_paired_curves(first, second)
    assert comparison.seeds == (1, 5)
    torch.testing.assert_close(comparison.difference.mean, torch.tensor([1.5, 1.0]))


@pytest.mark.parametrize(
    "curves",
    [
        {},
        {1: torch.tensor([1.0]), 2: torch.tensor([1.0, 2.0])},
        {1: torch.tensor([float("nan")])},
    ],
)
def test_invalid_curves_rejected(curves) -> None:
    with pytest.raises(ValueError):
        summarize_curves(curves)


def test_paired_seed_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="identical"):
        compare_paired_curves(
            {1: torch.tensor([1.0])},
            {2: torch.tensor([1.0])},
        )


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        summarize_curves({1: torch.tensor([1.0])}, confidence=1.0)
