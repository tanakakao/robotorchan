"""Joint q-batch runtime checks for native optimization backends."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import optimize_acqf


class JointQuadratic(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        target = torch.linspace(0.2, 0.8, X.shape[-2], dtype=X.dtype, device=X.device)
        return -(X - target.unsqueeze(-1)).square().sum(dim=(-1, -2))


@pytest.mark.parametrize("q", [1, 2, 4])
@pytest.mark.parametrize("optimizer", ["botorch", "torch_adam", "torch_adamw", "torch_sgd"])
def test_native_joint_q_batch(optimizer: str, q: int) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acquisition = JointQuadratic()
    initial = torch.linspace(0.1, 0.9, q, dtype=bounds.dtype).reshape(1, q, 1)
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer=optimizer,
        num_restarts=1,
        raw_samples=None,
        batch_initial_conditions=initial,
        options={"maxiter": 12},
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
