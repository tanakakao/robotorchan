"""Phase 4 cross-backend rejection of impossible joint q-batch constraints."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import CandidateConstraints
from robotorchan.optim.backends.genetic_algorithm import optimize_acqf_ga
from robotorchan.optim.backends.pso import optimize_acqf_pso


class SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        return X[..., 0].sum(dim=-1)


@pytest.mark.parametrize("backend", ["ga", "pso"])
def test_impossible_joint_nonlinear_q2_constraint_is_rejected(backend: str) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    # Every q=2 batch has x0 + x1 <= 2, so the required >= 2.5
    # cannot be satisfied, independently of the acquisition values.
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[:, 0].sum() - 2.5, False),)
    )
    objective = SumAcquisition()

    with pytest.raises(ValueError, match="no feasible candidate"):
        if backend == "ga":
            optimize_acqf_ga(
                objective,
                bounds,
                q=2,
                population_size=24,
                generations=3,
                seed=17,
                constraints=constraints,
            )
        else:
            optimize_acqf_pso(
                objective,
                bounds,
                q=2,
                swarm_size=24,
                iterations=3,
                seed=17,
                constraints=constraints,
            )
