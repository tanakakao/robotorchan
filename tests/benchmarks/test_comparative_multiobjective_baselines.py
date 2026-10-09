"""Phase 9 multiobjective baseline smoke and reproducibility tests."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_multiobjective_baselines import (
    run_multiobjective_baseline,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def _cell(strategy: str, problem: str = "branin_currin") -> ComparativeExperimentCell:
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


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_multiobjective_baseline_hypervolume_and_budget(strategy: str) -> None:
    result = run_multiobjective_baseline(_cell(strategy))
    assert result.evaluations.tolist() == [0, 1, 2, 3, 4]
    assert result.hypervolume.shape == (1, 5)
    assert result.trajectories[0].evaluation_count == 4
    assert result.trajectories[0].Y_truth.shape == (8, 2)
    assert torch.isfinite(result.hypervolume).all()
    assert (result.hypervolume[:, 1:] + 1e-8 >= result.hypervolume[:, :-1]).all()


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_multiobjective_baseline_reproducible(strategy: str) -> None:
    first = run_multiobjective_baseline(_cell(strategy))
    second = run_multiobjective_baseline(_cell(strategy))
    torch.testing.assert_close(first.trajectories[0].X, second.trajectories[0].X)
    torch.testing.assert_close(first.hypervolume, second.hypervolume)


def test_multiobjective_baselines_share_initial_design_and_hypervolume() -> None:
    random = run_multiobjective_baseline(_cell("random"))
    sobol = run_multiobjective_baseline(_cell("sobol"))
    torch.testing.assert_close(random.trajectories[0].X[:4], sobol.trajectories[0].X[:4])
    torch.testing.assert_close(random.hypervolume[:, 0], sobol.hypervolume[:, 0])


def test_multiobjective_baseline_rejects_other_problem() -> None:
    with pytest.raises(ValueError, match="branin_currin"):
        run_multiobjective_baseline(_cell("random", problem="branin"))


def test_multiobjective_baseline_rejects_acquisition_strategy() -> None:
    with pytest.raises(ValueError, match="random or sobol"):
        run_multiobjective_baseline(_cell("qEHVI"))


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_batch_checkpoints_exclude_partial_batches(strategy: str) -> None:
    from dataclasses import replace

    cell = _cell(strategy)
    config = replace(
        cell.config,
        seeds=tuple(range(5)),
        initial_points=20,
        evaluation_budget=40,
        q=3,
    )
    standard = ComparativeExperimentCell(tier="standard", config=config)
    result = run_multiobjective_baseline(standard)
    assert result.evaluations.tolist() == [0, *range(3, 40, 3), 40]
    assert result.hypervolume.shape == (5, len(result.evaluations))
