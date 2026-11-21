"""Phase 4 end-to-end constrained q-batch sampling reference."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import CandidateConstraints
from robotorchan.optim.backends.sampling import optimize_acqf_sampling, sample_candidate_batches


class PreferInfeasibleRegion(AcquisitionFunction):
    """Maximize both coordinates, making unconstrained winners infeasible."""

    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        return X.sum(dim=(-1, -2))


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_constrained_q2_sampling_matches_explicit_feasible_argmax(method: str) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    count, seed = 512, 37
    # Every candidate must lie below 0.8, and the two candidates together
    # must sum to at most 1.1. The acquisition prefers the infeasible region.
    constraints = CandidateConstraints(
        inequality_constraints=(
            (torch.tensor([0]), torch.tensor([-1.0]), -0.8),
            (torch.tensor([[0, 0], [1, 0]]), torch.tensor([-1.0, -1.0]), -1.1),
        )
    )
    pool = sample_candidate_batches(bounds, count, q=2, method=method, seed=seed)
    feasible = (
        (pool[..., 0] <= 0.8).all(dim=-1)
        & (pool[..., 0].sum(dim=-1) <= 1.1)
    )
    assert feasible.any()
    assert (~feasible).any()
    scores = pool[..., 0].sum(dim=-1)
    expected_index = torch.where(
        feasible, scores, torch.full_like(scores, -torch.inf)
    ).argmax()
    expected_candidate = pool[expected_index]
    expected_value = scores[expected_index]

    actual_candidate, actual_value = optimize_acqf_sampling(
        PreferInfeasibleRegion(),
        bounds,
        q=2,
        num_samples=count,
        method=method,
        seed=seed,
        constraints=constraints,
    )

    assert actual_candidate.shape == (2, 1)
    assert (actual_candidate[:, 0] <= 0.8 + 1e-12).all()
    assert actual_candidate[:, 0].sum() <= 1.1 + 1e-12
    torch.testing.assert_close(actual_candidate, expected_candidate, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, expected_value, atol=1e-12, rtol=1e-12)
