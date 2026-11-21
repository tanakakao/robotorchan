"""Phase 4 infeasible q-batch constraint handling across sampling and DE."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import CandidateConstraints
from robotorchan.optim.backends.differential_evolution import optimize_acqf_de
from robotorchan.optim.backends.sampling import optimize_acqf_sampling


class SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        return X[..., 0].sum(dim=-1)


def _impossible_joint_constraint() -> CandidateConstraints:
    # Both candidates are within [0, 1], so their sum cannot exceed 2.
    return CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[:, 0].sum() - 2.5, False),)
    )


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_sampling_rejects_impossible_joint_nonlinear_constraint(method: str) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="no feasible candidate"):
        optimize_acqf_sampling(
            SumAcquisition(),
            bounds,
            q=2,
            num_samples=64,
            method=method,
            seed=17,
            constraints=_impossible_joint_constraint(),
        )


def test_de_rejects_impossible_joint_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="no feasible candidate"):
        optimize_acqf_de(
            SumAcquisition(),
            bounds,
            q=2,
            seed=17,
            options={"maxiter": 3, "popsize": 5},
            constraints=_impossible_joint_constraint(),
        )
