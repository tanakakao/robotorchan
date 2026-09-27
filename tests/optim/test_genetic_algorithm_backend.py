"""Tests for the continuous genetic-algorithm optimizer backend."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_ga
from robotorchan.optim.constraints import CandidateConstraints


class _QuadraticAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.7) ** 2).sum(dim=(-2, -1))


def test_ga_optimizes_continuous_acquisition() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf_ga(
        _QuadraticAcquisition(),
        bounds,
        q=1,
        population_size=64,
        generations=40,
        seed=7,
    )

    assert candidates.shape == torch.Size([1, 1])
    assert candidates.dtype == bounds.dtype
    assert abs(float(candidates[0, 0]) - 0.7) < 0.05
    assert value.ndim == 0


def test_ga_is_reproducible_and_supports_joint_q_batch() -> None:
    bounds = torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double)
    kwargs = {"population_size": 24, "generations": 8, "seed": 11}
    first, first_value = optimize_acqf_ga(_QuadraticAcquisition(), bounds, q=2, **kwargs)
    second, second_value = optimize_acqf_ga(_QuadraticAcquisition(), bounds, q=2, **kwargs)

    assert first.shape == torch.Size([2, 2])
    assert torch.equal(first, second)
    assert torch.equal(first_value, second_value)


def test_ga_validates_configuration() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _QuadraticAcquisition()
    with pytest.raises(ValueError, match="population_size"):
        optimize_acqf_ga(acq, bounds, q=1, population_size=1)
    with pytest.raises(ValueError, match="elite_fraction"):
        optimize_acqf_ga(acq, bounds, q=1, elite_fraction=1.0)
    with pytest.raises(ValueError, match="tournament_size"):
        optimize_acqf_ga(acq, bounds, q=1, population_size=4, tournament_size=5)
