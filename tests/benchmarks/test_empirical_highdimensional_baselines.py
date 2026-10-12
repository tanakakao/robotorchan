"""Smoke-test high-dimensional baseline comparison eligibility and trajectories."""

import pytest
import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig


@pytest.mark.parametrize(
    "problem,dimension",
    [("sparse_sphere_50", 50), ("rotated_subspace_40", 40), ("interaction_chain_30", 30)],
)
@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_highdimensional_smoke_baselines(problem, dimension, strategy):
    cell = ComparativeExperimentCell(
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
    run = run_comparative_baseline(cell)[0]
    assert run.X.shape == (8, dimension)
    assert torch.isfinite(run.Y_truth).all()
    assert run.completed_batches == (1, 2, 3, 4)


def test_unverified_highdimensional_gp_is_not_eligible():
    with pytest.raises(ValueError, match="not applicable"):
        ComparativeExperimentCell(
            tier="smoke",
            config=BenchmarkExperimentConfig(
                problem="sparse_sphere_50",
                strategy="qEI",
                seeds=(0,),
                initial_points=4,
                evaluation_budget=4,
                q=1,
            ),
        )
