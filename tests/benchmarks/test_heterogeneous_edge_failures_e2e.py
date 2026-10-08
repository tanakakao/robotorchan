"""Phase 18: benchmark input, failure, and boundary contracts."""

import pytest
import torch

from robotorchan.benchmarks.heterogeneous_baselines import run_strategy
from robotorchan.benchmarks.heterogeneous_multiseed import (
    compare_strategies,
    summarize_trajectories,
)
from robotorchan.benchmarks.heterogeneous_synthetic import (
    evaluate_truth,
    observe,
    reference_front,
)


@pytest.mark.parametrize(
    "X",
    [
        torch.zeros(3),
        torch.zeros(2, 2),
        torch.full((2, 3), -0.01),
        torch.full((2, 3), 1.01),
        torch.full((2, 3), float("nan")),
        torch.full((2, 3), float("inf")),
        torch.zeros(2, 3, dtype=torch.long),
    ],
)
def test_synthetic_truth_rejects_invalid_inputs(X: torch.Tensor) -> None:
    """Invalid dimensions, domain values, and nonfinite inputs must fail clearly."""
    with pytest.raises(ValueError):
        evaluate_truth(X)


@pytest.mark.parametrize("noise_std", [-0.1, float("nan"), float("inf")])
def test_observation_rejects_invalid_noise(noise_std: float) -> None:
    """Reject invalid observation noise rather than propagating invalid labels."""
    with pytest.raises(ValueError):
        observe(torch.zeros(2, 3, dtype=torch.double), seed=18, noise_std=noise_std)


def test_observation_seed_validation_and_reproducibility() -> None:
    """Identical seeds reproduce regression observations and classification labels."""
    X = torch.rand(12, 3, generator=torch.Generator().manual_seed(18), dtype=torch.double)
    with pytest.raises(TypeError):
        observe(X, seed=1.5)
    first = observe(X, seed=18)
    second = observe(X, seed=18)
    torch.testing.assert_close(first.strength, second.strength)
    torch.testing.assert_close(first.conductivity, second.conductivity)
    torch.testing.assert_close(first.passed, second.passed)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_reference_front_rejects_invalid_probability_threshold(threshold: float) -> None:
    """Reject thresholds outside the probability range."""
    with pytest.raises(ValueError):
        reference_front(grid_size=3, probability_threshold=threshold)


def test_reference_front_rejects_degenerate_grid() -> None:
    """The reference grid needs at least two positions per dimension."""
    with pytest.raises(ValueError):
        reference_front(grid_size=1)


def test_reference_front_handles_empty_feasible_region() -> None:
    """An unattainable threshold returns empty tensors with stable dimensions."""
    X, Y = reference_front(grid_size=3, probability_threshold=1.0)
    assert X.shape == (0, 3)
    assert Y.shape == (0, 2)


@pytest.mark.parametrize("strategy", ["unknown", ""])
def test_baseline_rejects_unknown_strategy(strategy: str) -> None:
    """Unknown optimizers fail before any candidate can be appended."""
    initial_X = torch.rand(4, 3, dtype=torch.double)
    with pytest.raises(ValueError, match="Unknown benchmark strategy"):
        run_strategy(strategy, initial_X, seed=18, steps=1)


@pytest.mark.parametrize("seeds", [(), (18,), (18, 18)])
def test_multiseed_rejects_invalid_seed_collections(seeds: tuple[int, ...]) -> None:
    """At least two distinct seeds are required for sample statistics."""
    with pytest.raises(ValueError):
        compare_strategies(seeds)


@pytest.mark.parametrize(
    ("initial_points", "steps"),
    [(1, 2), (2, 0), (2, -1)],
)
def test_multiseed_rejects_invalid_budgets(initial_points: int, steps: int) -> None:
    """Reject insufficient initial designs and nonpositive evaluation budgets."""
    with pytest.raises(ValueError):
        compare_strategies((18, 19), initial_points=initial_points, steps=steps)


@pytest.mark.parametrize(
    "curves",
    [
        torch.ones(2, 2, dtype=torch.long),
        torch.full((2, 2), float("inf")),
        torch.full((2, 2), float("nan")),
    ],
)
def test_summary_rejects_nonfloat_and_nonfinite_curves(curves: torch.Tensor) -> None:
    """Invalid statistical inputs raise instead of producing misleading summaries."""
    with pytest.raises(ValueError):
        summarize_trajectories(curves)
