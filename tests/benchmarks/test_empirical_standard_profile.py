"""Contracts for the empirical standard comparison profile."""

from __future__ import annotations

import pytest

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


@pytest.mark.parametrize("q", (1, 3))
def test_empirical_standard_profile_roundtrip(q: int) -> None:
    """Use ten paired seeds, twenty initial points and fifty new evaluations."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=tuple(range(10)),
        initial_points=20,
        evaluation_budget=50,
        q=q,
        dtype="float64",
        device="cpu",
    )
    cell = ComparativeExperimentCell(tier="empirical_standard", config=config)

    assert ComparativeExperimentCell.from_dict(cell.to_dict()) == cell


@pytest.mark.parametrize(
    ("seeds", "initial_points", "evaluation_budget", "q"),
    (
        (tuple(range(5)), 20, 50, 1),
        (tuple(range(10)), 19, 50, 1),
        (tuple(range(10)), 20, 40, 1),
        (tuple(range(10)), 20, 50, 5),
    ),
)
def test_empirical_standard_rejects_mismatched_settings(
    seeds: tuple[int, ...],
    initial_points: int,
    evaluation_budget: int,
    q: int,
) -> None:
    """Prevent accidental mixing of legacy and empirical comparisons."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=seeds,
        initial_points=initial_points,
        evaluation_budget=evaluation_budget,
        q=q,
    )
    with pytest.raises(ValueError):
        ComparativeExperimentCell(tier="empirical_standard", config=config)


def test_legacy_standard_profile_remains_unchanged() -> None:
    """Keep existing published benchmark configurations loadable."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=tuple(range(5)),
        initial_points=20,
        evaluation_budget=40,
        q=3,
    )
    assert ComparativeExperimentCell(tier="standard", config=config).tier == "standard"


def test_empirical_standard_rejects_blocked_classification_acquisition() -> None:
    """Do not make unsupported classification-label history executable."""
    config = BenchmarkExperimentConfig(
        problem="strength_pass",
        strategy="qEI",
        seeds=tuple(range(10)),
        initial_points=20,
        evaluation_budget=50,
        q=1,
    )
    with pytest.raises(ValueError, match="Classification label history"):
        ComparativeExperimentCell(tier="empirical_standard", config=config)
