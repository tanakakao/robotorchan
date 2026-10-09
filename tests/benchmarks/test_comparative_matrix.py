"""Comparison tier and eligibility validation tests."""

import pytest

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def _cell(
    problem: str = "branin",
    strategy: str = "qEI",
    *,
    tier: str = "standard",
    q: int = 1,
    seeds: tuple[int, ...] = tuple(range(5)),
) -> ComparativeExperimentCell:
    return ComparativeExperimentCell(
        tier=tier,
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy=strategy,
            seeds=seeds,
            initial_points=20,
            evaluation_budget=40,
            q=q,
        ),
    )


def test_valid_cell_roundtrip() -> None:
    cell = _cell(q=3)
    assert ComparativeExperimentCell.from_dict(cell.to_dict()) == cell


@pytest.mark.parametrize(
    ("problem", "strategy"),
    [
        ("strength_pass", "qEI"),
        ("strength_conductivity_pass", "qEHVI"),
        ("dtlz2", "qEHVI"),
        ("noisy_quadratic", "qNEI"),
        ("branin", "qEHVI"),
    ],
)
def test_unsupported_cells_are_rejected(problem: str, strategy: str) -> None:
    with pytest.raises(ValueError):
        _cell(problem, strategy)


@pytest.mark.parametrize(
    ("problem", "strategy"),
    [
        ("strength_pass", "random"),
        ("strength_pass", "sobol"),
        ("strength_conductivity_pass", "random"),
        ("strength_conductivity_pass", "sobol"),
    ],
)
def test_classification_problem_baselines_remain_valid(problem: str, strategy: str) -> None:
    assert _cell(problem, strategy).config.strategy == strategy


def test_invalid_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="seeds"):
        _cell(seeds=(0, 1, 2, 3, 5))


def test_invalid_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="Batch"):
        _cell(q=5)


def test_invalid_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="tier"):
        _cell(tier="unknown")


def test_invalid_budget_is_rejected() -> None:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="qEI",
        seeds=tuple(range(5)),
        initial_points=20,
        evaluation_budget=41,
    )
    with pytest.raises(ValueError, match="budget"):
        ComparativeExperimentCell(tier="standard", config=config)


def test_unknown_serialization_keys_are_rejected() -> None:
    data = _cell().to_dict()
    data["extra"] = True
    with pytest.raises(ValueError, match="exactly"):
        ComparativeExperimentCell.from_dict(data)
