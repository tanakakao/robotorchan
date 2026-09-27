"""Tests for Differential Evolution acquisition optimization."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from scipy.optimize import OptimizeResult
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_de
from robotorchan.optim.constraints import CandidateConstraints


class _QuadraticAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.7) ** 2).sum(dim=(-2, -1))


def test_de_optimizes_continuous_acquisition() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf_de(
        _QuadraticAcquisition(),
        bounds,
        q=1,
        seed=7,
        options={"maxiter": 40, "popsize": 8, "tol": 1e-7},
    )

    assert candidates.shape == torch.Size([1, 1])
    assert candidates.dtype == bounds.dtype
    assert abs(float(candidates[0, 0]) - 0.7) < 0.03
    assert value.ndim == 0


def test_de_flattens_joint_q_batch() -> None:
    bounds = torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double)
    captured: dict[str, object] = {}

    def fake_de(objective, scipy_bounds, **kwargs):
        captured["bounds"] = scipy_bounds
        point = np.array([0.2, 0.0, 0.8, 1.0])
        captured["objective"] = objective(point)
        return OptimizeResult(x=point, fun=captured["objective"])

    with patch(
        "robotorchan.optim.backends.differential_evolution.differential_evolution",
        side_effect=fake_de,
    ):
        candidates, _ = optimize_acqf_de(_QuadraticAcquisition(), bounds, q=2, seed=3)

    assert len(captured["bounds"]) == 4
    assert candidates.shape == torch.Size([2, 2])


def test_de_rejects_candidate_constraints() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.2),)
    )
    with pytest.raises(ValueError, match="does not support candidate constraints yet"):
        optimize_acqf_de(_QuadraticAcquisition(), bounds, q=1, constraints=constraints)


def test_de_does_not_mutate_options() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    options = {"maxiter": 2, "popsize": 5}
    original = dict(options)

    optimize_acqf_de(_QuadraticAcquisition(), bounds, q=1, options=options, seed=1)

    assert options == original
