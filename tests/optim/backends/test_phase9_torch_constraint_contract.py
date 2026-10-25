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
@pytest.mark.parametrize("kind", ["linear", "equality", "nonlinear"])
def test_torch_optimizer_explicitly_rejects_unsupported_candidate_constraints(
    optimizer: str, kind: str
) -> None:
    """Do not silently ignore candidate constraints during gradient search."""
    linear = (torch.tensor([0]), torch.tensor([1.0]), 0.2)
    if kind == "linear":
        constraints = CandidateConstraints(inequality_constraints=(linear,))
    elif kind == "equality":
        constraints = CandidateConstraints(equality_constraints=(linear,))
    else:
        constraints = CandidateConstraints(
            nonlinear_inequality_constraints=((lambda x: x[0] - 0.2, True),)
        )

    with pytest.raises(ValueError, match="does not support candidate constraints"):
        optimize_acqf_torch(
            QuadraticAcquisition(),
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            q=1,
            num_restarts=1,
            raw_samples=4,
            optimizer=optimizer,
            constraints=constraints,
        )
