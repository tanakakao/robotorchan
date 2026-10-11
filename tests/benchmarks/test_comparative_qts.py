"""Test qTS empirical eligibility, validation and posterior-sampled candidates."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qts import run_comparative_qts
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.thompson_strategy import thompson_candidates


@pytest.mark.parametrize("problem", ["branin", "hartmann6"])
@pytest.mark.parametrize("q", [1, 3])
def test_qts_empirical_cell(problem, q):
    cell = ComparativeExperimentCell(
        tier="empirical_standard",
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy="qTS",
            seeds=tuple(range(10)),
            initial_points=20,
            evaluation_budget=50,
            q=q,
        ),
    )
    assert cell.config.strategy == "qTS"


def test_qts_real_candidates():
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    problem = registry.create("branin")
    bounds = problem.bounds.to(dtype=torch.float64)
    X = bounds[0] + torch.rand(8, 2, dtype=torch.float64) * (bounds[1] - bounds[0])
    Y = torch.linspace(0, 1, 8, dtype=torch.float64).unsqueeze(-1)
    candidates = thompson_candidates(
        problem, X, Y, 3, torch.Generator().manual_seed(11), pool_size=16
    )
    assert candidates.shape == (3, 2)
    assert torch.isfinite(candidates).all()
    assert (candidates >= bounds[0]).all()
    assert (candidates <= bounds[1]).all()
    assert torch.unique(candidates, dim=0).shape[0] == 3


def test_qts_rejects_wrong_cell():
    cell = ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem="branin",
            strategy="qEI",
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
        ),
    )
    with pytest.raises(ValueError, match="qTS"):
        run_comparative_qts(cell)
