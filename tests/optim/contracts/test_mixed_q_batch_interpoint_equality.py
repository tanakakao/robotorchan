"""Mixed q-batch interpoint equality contracts through public dispatch."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import CandidateConstraints, MixedVariableSpace, optimize_acqf
from robotorchan.optim.constraints.evaluation import candidate_is_feasible


class _JointAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -(X[..., 0] - 0.7).square().sum(dim=-1)


@pytest.mark.parametrize("q", [2, 4])
def test_mixed_ga_interpoint_equality_contract(q: int) -> None:
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 2.0, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0)},
    )
    constraints = CandidateConstraints(
        equality_constraints=(
            (
                torch.tensor([[i, 1] for i in range(q)]),
                torch.ones(q, dtype=torch.double),
                float(q),
            ),
        ),
    )
    acquisition = _JointAcquisition()
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="ga",
        variable_space=space,
        constraints=constraints,
        optimizer_options={"population_size": 100, "generations": 30},
        seed=23,
    )
    assert candidates.shape == (q, 3)
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    assert torch.equal(candidates[:, 1], candidates[:, 1].round())
    assert torch.all(torch.isin(candidates[:, 2], candidates.new_tensor([10.0, 30.0])))
    assert candidates[:, 1].sum().item() == float(q)
    assert bool(candidate_is_feasible(candidates.unsqueeze(0), constraints).item())
    torch.testing.assert_close(
        value.reshape(()),
        acquisition(candidates.unsqueeze(0)).reshape(()),
    )
