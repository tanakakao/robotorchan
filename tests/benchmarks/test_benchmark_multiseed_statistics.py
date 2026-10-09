"""Bootstrap reproducibility and paired seed statistical tests."""

import pytest
import torch

from robotorchan.benchmarks.multiseed_statistics import (
    bootstrap_curves,
    compare_paired_bootstrap,
)


def test_bootstrap_determinism_and_shapes() -> None:
    curves = {
        3: torch.tensor([1.0, 3.0], dtype=torch.double),
        1: torch.tensor([3.0, 5.0], dtype=torch.double),
        2: torch.tensor([2.0, 4.0], dtype=torch.double),
    }
    first = bootstrap_curves(curves, n_resamples=100, seed=12)
    second = bootstrap_curves(curves, n_resamples=100, seed=12)
    assert first.seeds == (1, 2, 3)
    torch.testing.assert_close(first.mean, torch.tensor([2.0, 4.0], dtype=torch.double))
    torch.testing.assert_close(first.lower, second.lower)
    torch.testing.assert_close(first.upper, second.upper)
    assert (first.lower <= first.mean).all()
    assert (first.mean <= first.upper).all()


def test_paired_bootstrap_preserves_alignment() -> None:
    first = {
        2: torch.tensor([4.0, 7.0]),
        1: torch.tensor([3.0, 6.0]),
    }
    second = {
        1: torch.tensor([1.0, 2.0]),
        2: torch.tensor([2.0, 3.0]),
    }
    result = compare_paired_bootstrap(first, second, n_resamples=100, seed=9)
    assert result.seeds == (1, 2)
    torch.testing.assert_close(result.difference.mean, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(result.probability_positive, torch.ones(2))


def test_single_seed_bootstrap_is_degenerate() -> None:
    result = bootstrap_curves({1: torch.tensor([2.0, 3.0])}, n_resamples=10)
    torch.testing.assert_close(result.mean, result.lower)
    torch.testing.assert_close(result.mean, result.upper)


@pytest.mark.parametrize("kwargs", [
    {"n_resamples": 0},
    {"n_resamples": True},
    {"seed": -1},
    {"confidence": 1.0},
])
def test_invalid_bootstrap_options(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        bootstrap_curves({1: torch.tensor([1.0])}, **kwargs)


def test_paired_bootstrap_rejects_unmatched_seeds() -> None:
    with pytest.raises(ValueError, match="identical"):
        compare_paired_bootstrap(
            {1: torch.tensor([1.0])},
            {2: torch.tensor([1.0])},
        )
