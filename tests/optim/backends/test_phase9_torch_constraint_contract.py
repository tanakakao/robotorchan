"""Phase 9 PyTorch optimizer constraint-contract verification."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import CandidateConstraints
from robotorchan.optim.backends.torch import optimize_acqf_torch


class QuadraticAcquisition(AcquisitionFunction):
    """Simple differentiable acquisition for contract checks."""

    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        return -(X - 0.5).square().sum(dim=(-1, -2))


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd"])
def test_torch_nonlinear_penalty_and_strict_mode(optimizer: str, monkeypatch) -> None:
    from unittest.mock import Mock

    from robotorchan.optim.backends import torch as backend

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    candidate = torch.tensor([[0.1]], dtype=torch.float64)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] - 0.2, True),)
    )
    mocked = Mock(return_value=(candidate, torch.tensor(0.0, dtype=bounds.dtype)))
    monkeypatch.setattr(backend, "botorch_optimize_acqf", mocked)
    with pytest.raises(ValueError, match="no feasible candidate"):
        optimize_acqf_torch(
            QuadraticAcquisition(), bounds, q=1, num_restarts=1, raw_samples=4,
            optimizer=optimizer, constraints=constraints,
        )
    returned, _ = optimize_acqf_torch(
        QuadraticAcquisition(), bounds, q=1, num_restarts=1, raw_samples=4,
        optimizer=optimizer, constraints=constraints, constraint_mode="penalty",
    )
    torch.testing.assert_close(returned, candidate)
    penalized = mocked.call_args.kwargs["acq_function"]
    assert penalized(candidate.unsqueeze(0)).item() < -1.0
