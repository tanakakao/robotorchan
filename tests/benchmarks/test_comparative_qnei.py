"""Phase 7 qNEI comparative benchmark integration tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qnei import (
    make_comparative_qnei_strategy,
    run_comparative_qnei,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import sobol_initial_design
from robotorchan.benchmarks.standard_problems import branin, hartmann6


def _cell(problem: str = "branin", strategy: str = "qNEI") -> ComparativeExperimentCell:
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
def test_qnei_smoke_completes_budget_and_respects_bounds(problem: str) -> None:
    trajectory = run_comparative_qnei(_cell(problem), num_restarts=1, raw_samples=8)[0]
    assert trajectory.evaluation_count == 4
    assert trajectory.X.shape[0] == 8
    assert trajectory.Y_truth.shape == (8, 1)
    assert torch.isfinite(trajectory.X).all()
    assert torch.isfinite(trajectory.Y_observed).all()


@pytest.mark.parametrize("problem", ["branin", "hartmann6"])
def test_qnei_uses_paired_initial_design(problem: str) -> None:
    trajectory = run_comparative_qnei(_cell(problem), num_restarts=1, raw_samples=8)[0]
    factory = branin if problem == "branin" else hartmann6
    expected = sobol_initial_design(factory(), 4, 0, dtype=torch.double, device=torch.device("cpu"))
    torch.testing.assert_close(trajectory.X[:4], expected)


def test_qnei_rejects_other_strategy() -> None:
    with pytest.raises(ValueError, match="qNEI comparison cell"):
        run_comparative_qnei(_cell(strategy="qEI"))


@pytest.mark.parametrize("name,value", [("num_restarts", 0), ("raw_samples", 0)])
def test_qnei_rejects_invalid_optimizer_settings(name: str, value: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_comparative_qnei_strategy(**{name: value})
