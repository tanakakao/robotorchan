"""Candidate-constraint capability contracts for q-batch optimization."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import CandidateConstraints, optimize_acqf


class _JointAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -(X[..., 0] - 0.7).square().sum(dim=-1)


@pytest.mark.parametrize("q", [1, 2, 4])
def test_torch_linear_constraints_supported(q: int) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0], dtype=torch.double), 0.2),),
    )
    candidates, value = optimize_acqf(
        _JointAcquisition(), bounds, q=q, optimizer="torch_adam",
        constraints=constraints, num_restarts=2, raw_samples=16,
        optimizer_options={"constraint_mode": "penalty"},
        options={"maxiter": 10},
    )
    assert candidates.shape == (q, 1)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("q", [2, 4])
@pytest.mark.parametrize("optimizer", ["ga", "pso"])
def test_interpoint_linear_constraint_joint_batch(optimizer: str, q: int) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, 1.0], dtype=torch.double),
                0.2,
            ),
        ),
    )
    backend_options = (
        {"population_size": 12, "generations": 2}
        if optimizer == "ga"
        else {"swarm_size": 12, "iterations": 2}
    )
    acquisition = _JointAcquisition()
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer=optimizer,
        constraints=constraints,
        optimizer_options=backend_options,
        seed=11,
    )
    assert candidates.shape == (q, 1)
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    assert (candidates[0, 0] + candidates[1, 0]).item() >= 0.2 - 1e-8
    torch.testing.assert_close(value.reshape(()), acquisition(candidates.unsqueeze(0)).reshape(()))
