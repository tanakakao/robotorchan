"""Phase 4 independent mixed-variable constrained q-batch Sobol reference."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import CandidateConstraints, MixedVariableSpace
from robotorchan.optim.backends.sampling import optimize_acqf_sampling, sample_candidate_batches


class SumWithIntegerReward(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        return (X[..., 0] + 0.1 * X[..., 1]).sum(dim=-1)


def test_mixed_q2_sobol_constrained_argmax_matches_independent_oracle() -> None:
    bounds = torch.tensor([[0.0, 1.2, 10.0], [1.0, 4.8, 30.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds, integer_dims=(1,), categorical_values={2: (10.0, 20.0, 30.0)}
    )
    constraints = CandidateConstraints(
        inequality_constraints=(
            (torch.tensor([0]), torch.tensor([-1.0]), -0.75),
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([-1.0, -1.0]),
                -1.0,
            ),
        )
    )
    count, seed = 256, 31
    # Independent inverse-CDF mapping of the Sobol coordinates, without
    # invoking robotorchan's mixed-space mapping or feasibility evaluator.
    pool = sample_candidate_batches(bounds, count, q=2, method="sobol", seed=seed)
    integer_unit = (pool[..., 1] - 1.2) / (4.8 - 1.2)
    category_unit = (pool[..., 2] - 10.0) / 20.0
    pool[..., 1] = 2.0 + torch.floor(integer_unit * 3).clamp(max=2)
    pool[..., 2] = 10.0 + 10.0 * torch.floor(category_unit * 3).clamp(max=2)
    pool[..., 2] = 20.0  # Public fixed-feature contract.

    feasible = (pool[..., 0] <= 0.75).all(dim=-1) & (pool[..., 0].sum(dim=-1) <= 1.0)
    assert feasible.any()
    assert (~feasible).any()
    scores = (pool[..., 0] + 0.1 * pool[..., 1]).sum(dim=-1)
    expected_index = torch.where(feasible, scores, torch.full_like(scores, -torch.inf)).argmax()

    candidate, value = optimize_acqf_sampling(
        SumWithIntegerReward(),
        bounds,
        q=2,
        num_samples=count,
        method="sobol",
        seed=seed,
        variable_space=space,
        fixed_features={2: 20.0},
        constraints=constraints,
    )

    assert candidate.shape == (2, 3)
    assert torch.equal(candidate[:, 1], candidate[:, 1].round())
    assert torch.all((candidate[:, 1] >= 2) & (candidate[:, 1] <= 4))
    assert torch.all(candidate[:, 2] == 20.0)
    assert torch.all(candidate[:, 0] <= 0.75)
    assert candidate[:, 0].sum() <= 1.0
    torch.testing.assert_close(candidate, pool[expected_index], atol=0, rtol=0)
    torch.testing.assert_close(value, scores[expected_index], atol=1e-12, rtol=1e-12)
