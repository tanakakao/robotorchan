"""Phase 6 comparative qEI integration tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qei import (
    make_comparative_qei_strategy,
    run_comparative_qei,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def _cell(problem: str = "branin", strategy: str = "qEI") -> ComparativeExperimentCell:
    return ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )


@pytest.mark.parametrize("problem", ["branin", "hartmann6"])
def test_qei_smoke_completes_budget_and_respects_bounds(problem: str) -> None:
    trajectory = run_comparative_qei(_cell(problem), num_restarts=1, raw_samples=8)[0]
    assert trajectory.evaluation_count == 4
    assert trajectory.X.shape[0] == 8
    assert trajectory.Y_truth.shape == (8, 1)
    assert torch.isfinite(trajectory.X).all()
    assert torch.isfinite(trajectory.Y_observed).all()


def test_qei_rejects_other_strategy() -> None:
    with pytest.raises(ValueError, match="qEI comparison cell"):
        run_comparative_qei(_cell(strategy="qNEI"))


@pytest.mark.parametrize("name,value", [("num_restarts", 0), ("raw_samples", 0)])
def test_qei_rejects_invalid_optimizer_settings(name: str, value: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_comparative_qei_strategy(**{name: value})
