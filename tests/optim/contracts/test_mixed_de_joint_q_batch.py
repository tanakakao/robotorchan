"""Mixed differential-evolution q-batch contracts."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import MixedVariableSpace, optimize_acqf


class _JointMixedAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fit = -(X[..., 0] - 0.6).square().sum(dim=-1)
        interaction = -0.1 * (X[..., 1:, 0] - X[..., :-1, 0]).square().sum(dim=-1)
        return fit + interaction


@pytest.mark.parametrize("q", [1, 2, 4])
def test_mixed_de_joint_batch_contract(q: int) -> None:
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 20.0, 30.0)},
    )
    acquisition = _JointMixedAcquisition()
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="de",
        variable_space=space,
        optimizer_options={"options": {"maxiter": 1, "popsize": 3}},
        seed=17,
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
