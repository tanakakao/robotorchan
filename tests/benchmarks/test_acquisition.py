"""Tests for the acquisition-function benchmark harness."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.benchmarks import benchmark_acquisition


class _QuadraticAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.4) ** 2).sum(dim=(-1, -2))


def test_benchmark_acquisition_reports_fixed_batch_metrics() -> None:
    X = torch.linspace(0.0, 1.0, 24, dtype=torch.double).reshape(4, 2, 3)

    result = benchmark_acquisition("quadratic", _QuadraticAcquisition(), X)

    assert result.name == "quadratic"
    assert result.batch_size == 4
    assert result.q == 2
    assert result.input_dim == 3
    assert result.wall_time_seconds >= 0.0
    assert torch.isfinite(torch.tensor(result.value_mean))
    assert result.value_std >= 0.0


def test_benchmark_acquisition_accepts_multiple_batch_dimensions() -> None:
    X = torch.zeros(2, 3, 1, 4, dtype=torch.double)

    result = benchmark_acquisition("quadratic", _QuadraticAcquisition(), X)

    assert result.batch_size == 6
    assert result.q == 1
    assert result.input_dim == 4
