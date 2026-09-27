"""Tests for CMA-ES acquisition optimization."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_cmaes
from robotorchan.optim.constraints import CandidateConstraints


class _QuadraticAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.7) ** 2).sum(dim=(-2, -1))


class _FakeCMA:
    population_size = 2

    def __init__(self, *, mean, sigma, **kwargs):
        self.mean = mean
        self.sigma = sigma
        self.kwargs = kwargs
        self._asked = 0

    def ask(self):
        points = [np.full_like(self.mean, 0.2), np.full_like(self.mean, 0.7)]
        point = points[self._asked % len(points)]
        self._asked += 1
        return point.copy()

    def tell(self, solutions):
        self.solutions = solutions

    def should_stop(self):
        return True


def test_cmaes_selects_best_generated_candidate() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    module = SimpleNamespace(CMA=_FakeCMA)
    with patch.dict(sys.modules, {"cmaes": module}):
        candidates, value = optimize_acqf_cmaes(
            _QuadraticAcquisition(), bounds, q=1, seed=7
        )

    assert candidates.shape == torch.Size([1, 1])
    assert candidates.dtype == bounds.dtype
    assert torch.allclose(candidates, torch.tensor([[0.7]], dtype=torch.double))
    assert value.ndim == 0


def test_cmaes_supports_joint_q_batch() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    module = SimpleNamespace(CMA=_FakeCMA)
    with patch.dict(sys.modules, {"cmaes": module}):
        candidates, _ = optimize_acqf_cmaes(_QuadraticAcquisition(), bounds, q=2)

    assert candidates.shape == torch.Size([2, 2])


def test_cmaes_rejects_candidate_constraints() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.2),)
    )
    with pytest.raises(ValueError, match="does not support candidate constraints yet"):
        optimize_acqf_cmaes(
            _QuadraticAcquisition(), bounds, q=1, constraints=constraints
        )


def test_cmaes_reports_missing_optional_dependency() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with (
        patch.dict(sys.modules, {"cmaes": None}),
        pytest.raises(ImportError, match=r"robotorchan\[cmaes\]"),
    ):
        optimize_acqf_cmaes(_QuadraticAcquisition(), bounds, q=1)
