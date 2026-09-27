"""Tests for the optimizer benchmark harness."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_ga, optimize_acqf_sampling
from robotorchan.optim.benchmark import benchmark_optimizer, benchmark_optimizers
from robotorchan.optim.constraints import CandidateConstraints


class _Quadratic(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.35) ** 2).sum(dim=(-1, -2))


def test_benchmark_optimizer_reports_common_metrics() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    result = benchmark_optimizer(
        "sampling",
        optimize_acqf_sampling,
        _Quadratic(),
        bounds,
        1,
        optimizer_kwargs={"num_samples": 32, "method": "random"},
        seed=7,
    )
    assert result.name == "sampling"
    assert result.candidate.shape == (1, 1)
    assert result.acquisition_value.ndim == 0
    assert result.wall_time_seconds >= 0
    assert result.acquisition_evaluations == 32
    assert result.feasible is None
    assert result.seed == 7


def test_benchmark_optimizer_reports_feasibility() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.2,
            ),
        )
    )
    result = benchmark_optimizer(
        "ga",
        optimize_acqf_ga,
        _Quadratic(),
        bounds,
        1,
        optimizer_kwargs={"population_size": 16, "generations": 3},
        constraints=constraints,
        seed=4,
    )
    assert result.feasible is True
    # GA evaluates 3 populations of 16 plus one final raw-value evaluation.
    assert result.acquisition_evaluations == 49


def test_benchmark_optimizers_runs_multiple_seeds() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    results = benchmark_optimizers(
        {
            "random": (
                optimize_acqf_sampling,
                {"num_samples": 16, "method": "random"},
            ),
        },
        _Quadratic,
        bounds,
        1,
        seeds=(1, 2),
    )
    assert [result.seed for result in results] == [1, 2]
    assert all(result.name == "random" for result in results)
