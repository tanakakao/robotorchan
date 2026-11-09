"""Focused tests for conservative safe-set expansion."""

import pytest
import torch

from robotorchan.optim.safe_expansion import propose_safe_expansion
from robotorchan.semantics import ContinuousConstraint


class _Posterior:
    def __init__(self, X: torch.Tensor) -> None:
        self.mean = X[..., :1]
        self.variance = (0.01 + 0.04 * X[..., :1]).square()


class _SafetyModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        return _Posterior(X)


def test_expansion_candidate_is_pessimistically_safe() -> None:
    candidate, score = propose_safe_expansion(
        _SafetyModel(),
        torch.tensor([[0.0], [1.0]]),
        (ContinuousConstraint(0, 0.6),),
        num_samples=128,
        seed=5,
    )
    assert candidate.shape == (1, 1)
    assert candidate.item() + 2 * (0.01 + 0.04 * candidate.item()) <= 0.6
    assert score.item() > 0


def test_no_safe_candidate_raises() -> None:
    with pytest.raises(RuntimeError, match="No posterior-safe candidates"):
        propose_safe_expansion(
            _SafetyModel(),
            torch.tensor([[0.0], [1.0]]),
            (ContinuousConstraint(0, -1.0),),
            num_samples=32,
            seed=5,
        )
