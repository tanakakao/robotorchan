"""Mixed-variable q-batch contracts through the public optimizer dispatcher."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import MixedVariableSpace, optimize_acqf


class _MixedJointAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fit = -(X[..., 0] - 0.6).square().sum(dim=-1)
        interaction = -0.1 * (X[..., 1:, 0] - X[..., :-1, 0]).square().sum(dim=-1)
        return fit + interaction


@pytest.mark.parametrize("q", [1, 2, 4])
@pytest.mark.parametrize(
    ("optimizer", "optimizer_options"),
    [
        ("ga", {"population_size": 12, "generations": 2}),
        ("pso", {"swarm_size": 12, "iterations": 2}),
        ("random", {"num_samples": 24}),
        ("sobol", {"num_samples": 24}),
    ],
)
def test_mixed_joint_q_batch_contract(
    optimizer: str, optimizer_options: dict, q: int
) -> None:
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 20.0, 30.0)},
    )
    acquisition = _MixedJointAcquisition()
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer=optimizer,
        variable_space=space,
        optimizer_options=optimizer_options,
        seed=7,
    )
    assert candidates.shape == (q, 3)
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    assert torch.equal(candidates[:, 1], candidates[:, 1].round())
    assert torch.all(torch.isin(candidates[:, 2], candidates.new_tensor([10.0, 20.0, 30.0])))
    torch.testing.assert_close(
        value.reshape(()),
        acquisition(candidates.unsqueeze(0)).reshape(()),
        rtol=1e-8,
        atol=1e-8,
    )
