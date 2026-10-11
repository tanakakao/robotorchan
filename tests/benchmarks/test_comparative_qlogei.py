"""Verify qLogEI benchmark eligibility and real BoTorch candidate generation."""

import pytest
import torch

from robotorchan.benchmarks.botorch_strategy import botorch_gp_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qlogei import run_comparative_qlogei
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry


@pytest.mark.parametrize("problem", ["branin", "hartmann6"])
@pytest.mark.parametrize("q", [1, 3])
def test_qlogei_empirical_cell(problem, q):
    cell = ComparativeExperimentCell(
        tier="empirical_standard",
        config=BenchmarkExperimentConfig(
            problem=problem,
            strategy="qLogEI",
            seeds=tuple(range(10)),
            initial_points=20,
            evaluation_budget=50,
            q=q,
        ),
    )
    assert cell.config.strategy == "qLogEI"


def test_qlogei_rejects_wrong_cell():
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
    with pytest.raises(ValueError, match="qLogEI"):
        run_comparative_qlogei(cell)


def test_qlogei_real_candidate_is_finite_and_in_bounds():
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    problem = registry.create("branin")
    bounds = problem.bounds.to(dtype=torch.float64)
    X = bounds[0] + torch.rand(6, 2, dtype=torch.float64) * (bounds[1] - bounds[0])
    Y = torch.zeros(6, 1, dtype=torch.float64)
    candidate = botorch_gp_candidates(
        problem,
        X,
        Y,
        1,
        torch.Generator().manual_seed(7),
        acquisition="qLogEI",
        num_restarts=1,
        raw_samples=8,
    )
    assert candidate.shape == (1, 2)
    assert torch.isfinite(candidate).all()
    assert (candidate >= bounds[0]).all()
    assert (candidate <= bounds[1]).all()
