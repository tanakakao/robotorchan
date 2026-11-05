"""Mixed BoTorch q-batch contracts through the public dispatcher."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import MixedVariableSpace, optimize_acqf


class _JointAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)
        self.X_pending = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fit = -(X[..., 0] - 0.6).square().sum(dim=-1)
        interaction = -0.1 * (X[..., 1:, 0] - X[..., :-1, 0]).square().sum(dim=-1)
        return fit + interaction


@pytest.mark.parametrize("q", [1, 2, 4])
def test_mixed_botorch_q_batch_contract(q: int) -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0)})
    acquisition = _JointAcquisition()
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="botorch",
        variable_space=space,
        optimizer_options={"fixed_features_list": [{1: 0.0}, {1: 1.0}]},
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 12},
    )
    assert candidates.shape == (q, 2)
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    assert torch.all(torch.isin(candidates[:, 1], candidates.new_tensor([0.0, 1.0])))
    torch.testing.assert_close(
        value.reshape(()),
        acquisition(candidates.unsqueeze(0)).reshape(()),
        rtol=1e-6,
        atol=1e-6,
    )
