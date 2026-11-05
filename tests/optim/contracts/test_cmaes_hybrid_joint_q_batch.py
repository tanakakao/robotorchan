"""Joint q-batch contracts for CMA-ES and hybrid optimizers."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import optimize_acqf


class _JointAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        target = torch.linspace(0.2, 0.8, X.shape[-2], dtype=X.dtype, device=X.device)
        return -(X - target.unsqueeze(-1)).square().sum(dim=(-1, -2))


class _DeterministicCMA:
    population_size = 2

    def __init__(self, *, mean, sigma, **kwargs):
        self.mean = mean
        self.asked = 0

    def ask(self):
        fraction = 0.25 if self.asked % 2 == 0 else 0.75
        self.asked += 1
        return np.full_like(self.mean, fraction)

    def tell(self, solutions):
        pass

    def should_stop(self):
        return True


@pytest.mark.parametrize("q", [1, 2, 4])
def test_cmaes_joint_q_batch_contract(q: int) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acquisition = _JointAcquisition()
    with patch.dict(sys.modules, {"cmaes": SimpleNamespace(CMA=_DeterministicCMA)}):
        candidates, value = optimize_acqf(
            acquisition,
            bounds,
            q=q,
            optimizer="cmaes",
            optimizer_options={"max_generations": 2},
        )
    assert candidates.shape == (q, 1)
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    torch.testing.assert_close(value, acquisition(candidates.unsqueeze(0)).reshape(()))


@pytest.mark.parametrize("q", [1, 2, 4])
def test_hybrid_joint_q_batch_contract(q: int) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acquisition = _JointAcquisition()

    def global_optimizer(acq_function, bounds, q, **kwargs):
        candidate = torch.linspace(0.1, 0.9, q, dtype=bounds.dtype).reshape(q, 1)
        return candidate, acq_function(candidate.unsqueeze(0)).reshape(())

    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="hybrid",
        optimizer_options={
            "global_optimizer": global_optimizer,
            "local_options": {"maxiter": 12},
        },
    )
    assert candidates.shape == (q, 1)
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    torch.testing.assert_close(
        value.reshape(()),
        acquisition(candidates.unsqueeze(0)).reshape(()),
        rtol=1e-7,
        atol=1e-7,
    )
