"""Mixed q-batch feasibility contracts through the public dispatcher."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import CandidateConstraints, MixedVariableSpace, optimize_acqf
from robotorchan.optim.constraints.evaluation import candidate_is_feasible


class _JointAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -(X[..., 0] - 0.8).square().sum(dim=-1)


@pytest.mark.parametrize("q", [2, 4])
@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_mixed_ga_interpoint_constraint_contract(q: int, kind: str) -> None:
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 3.0, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0)},
    )
    if kind == "linear":
        constraints = CandidateConstraints(
            inequality_constraints=(
                (
                    torch.tensor([[i, 0] for i in range(q)]),
                    torch.full((q,), -1.0, dtype=torch.double),
                    -0.5,
                ),
            ),
        )
    else:
        constraints = CandidateConstraints(
            nonlinear_inequality_constraints=((lambda x: 0.5 - x[:, 0].sum(), False),)
        )
    acquisition = _JointAcquisition()
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="ga",
        variable_space=space,
        constraints=constraints,
        optimizer_options={"population_size": 80, "generations": 35},
        seed=29,
    )
    assert candidates.shape == (q, 3)
    assert torch.isfinite(candidates).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    assert torch.equal(candidates[:, 1], candidates[:, 1].round())
    assert torch.all(torch.isin(candidates[:, 2], candidates.new_tensor([10.0, 30.0])))
    assert bool(candidate_is_feasible(candidates.unsqueeze(0), constraints).item())
    torch.testing.assert_close(
        value.reshape(()),
        acquisition(candidates.unsqueeze(0)).reshape(()),
    )
