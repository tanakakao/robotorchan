"""Cross-backend q-batch candidate and acquisition-value contracts."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import optimize_acqf


class _JointQuadratic(AcquisitionFunction):
    """A differentiable acquisition with a nonseparable batch interaction."""

    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        center = torch.linspace(0.2, 0.8, X.shape[-2], dtype=X.dtype, device=X.device)
        target = center.unsqueeze(-1)
        fit = -(X - target).square().sum(dim=(-1, -2))
        interaction = -0.1 * (X[..., 1:, :] - X[..., :-1, :]).square().sum(dim=(-1, -2))
        return fit + interaction


@pytest.mark.parametrize("q", [1, 2, 4])
@pytest.mark.parametrize(
    ("optimizer", "kwargs"),
    [
        ("random", {"seed": 7, "optimizer_options": {"num_samples": 24}}),
        ("sobol", {"seed": 7, "optimizer_options": {"num_samples": 24}}),
        ("ga", {"seed": 7, "optimizer_options": {"population_size": 12, "generations": 2}}),
        ("pso", {"seed": 7, "optimizer_options": {"swarm_size": 12, "iterations": 2}}),
        (
            "de",
            {"seed": 7, "optimizer_options": {"options": {"maxiter": 1, "popsize": 3}}},
        ),
    ],
)
def test_joint_q_batch_contract(optimizer: str, kwargs: dict, q: int) -> None:
    """Each optimizer must return one in-bounds joint batch and its exact score."""
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acquisition = _JointQuadratic()

    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer=optimizer,
        **kwargs,
    )

    assert candidates.shape == (q, 1)
    assert value.numel() == 1
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    expected = acquisition(candidates.unsqueeze(0)).reshape(())
    torch.testing.assert_close(value.reshape(()), expected, rtol=1e-8, atol=1e-8)
