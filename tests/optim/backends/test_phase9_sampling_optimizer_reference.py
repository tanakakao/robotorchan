"""Phase 9 sampling-optimizer reference for known acquisition maximizers."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import optimize_acqf
from robotorchan.optim.backends.sampling import sample_candidate_batches


class NegativeQuadratic(AcquisitionFunction):
    """A deterministic q-batch objective with a known global maximum of zero."""

    def __init__(self, target: Tensor) -> None:
        super().__init__(model=None)
        self.target = target

    def forward(self, X: Tensor) -> Tensor:
        return -(X - self.target).square().sum(dim=(-1, -2))


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_sampling_optimizer_selects_independent_reference_argmax(method: str) -> None:
    """Verify selection, bounds and returned score against an explicit candidate pool."""
    bounds = torch.tensor([[-2.0, 1.0], [3.0, 5.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, 2.7], [1.1, 4.2]], dtype=torch.float64)
    objective = NegativeQuadratic(target)
    seed = 29
    count = 128

    candidates = sample_candidate_batches(bounds, count, q=2, method=method, seed=seed)
    independent_scores = -((candidates - target) ** 2).sum(dim=(-1, -2))
    best_index = int(independent_scores.argmax())
    expected_candidate = candidates[best_index]
    expected_score = independent_scores[best_index]

    actual_candidate, actual_score = optimize_acqf(
        objective,
        bounds,
        q=2,
        optimizer=method,
        seed=seed,
        optimizer_options={"num_samples": count},
    )

    assert actual_candidate.shape == (2, 2)
    assert torch.all(actual_candidate >= bounds[0])
    assert torch.all(actual_candidate <= bounds[1])
    torch.testing.assert_close(actual_candidate, expected_candidate, atol=0, rtol=0)
    torch.testing.assert_close(actual_score, expected_score, atol=1e-12, rtol=1e-12)
    assert actual_score <= 0
