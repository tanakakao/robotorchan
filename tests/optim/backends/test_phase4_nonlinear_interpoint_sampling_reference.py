"""Phase 4 independent nonlinear interpoint q-batch sampling reference."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import CandidateConstraints
from robotorchan.optim.backends.sampling import optimize_acqf_sampling, sample_candidate_batches


class SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: Tensor) -> Tensor:
        return X[..., 0].sum(dim=-1)


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_nonlinear_interpoint_q2_matches_independent_feasible_argmax(method: str) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    count, seed = 512, 41
    # The per-point upper bound and joint product limit are independent:
    # neither is implied by the other, and unconstrained maximization violates both.
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (lambda x: 0.8 - x[0], True),
            (lambda x: 0.12 - x[0, 0] * x[1, 0], False),
        )
    )
    pool = sample_candidate_batches(bounds, count, q=2, method=method, seed=seed)
    x0, x1 = pool[:, 0, 0], pool[:, 1, 0]
    feasible = (x0 <= 0.8) & (x1 <= 0.8) & (x0 * x1 <= 0.12)
    assert feasible.any()
    assert (~feasible).any()
    scores = x0 + x1
    expected_index = torch.where(feasible, scores, torch.full_like(scores, -torch.inf)).argmax()

    candidate, value = optimize_acqf_sampling(
        SumAcquisition(),
        bounds,
        q=2,
        num_samples=count,
        method=method,
        seed=seed,
        constraints=constraints,
    )

    assert candidate.shape == (2, 1)
    assert torch.all(candidate[:, 0] <= 0.8)
    assert candidate[0, 0] * candidate[1, 0] <= 0.12
    torch.testing.assert_close(candidate, pool[expected_index], atol=0, rtol=0)
    torch.testing.assert_close(value, scores[expected_index], atol=1e-12, rtol=1e-12)
