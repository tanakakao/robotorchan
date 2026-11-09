"""Focused tests for q=1 safe candidate optimization."""

import pytest
import torch

from robotorchan.optim.safe import optimize_safe_acqf
from robotorchan.semantics import ContinuousConstraint


class _Posterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean
        self.variance = torch.zeros_like(mean)


class _SafetyModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        return _Posterior(X[..., :1])


class _Acquisition:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0, 0]


def test_selects_only_safe_candidates() -> None:
    candidate, value = optimize_safe_acqf(
        _Acquisition(),
        _SafetyModel(),
        torch.tensor([[0.0], [1.0]]),
        (ContinuousConstraint(0, 0.4),),
        num_samples=64,
        seed=7,
    )
    assert candidate.shape == (1, 1)
    assert candidate.item() <= 0.4
    assert value.item() <= 0.4


def test_empty_safe_pool_raises() -> None:
    with pytest.raises(RuntimeError, match="No posterior-safe candidates"):
        optimize_safe_acqf(
            _Acquisition(),
            _SafetyModel(),
            torch.tensor([[0.0], [1.0]]),
            (ContinuousConstraint(0, -1.0),),
            num_samples=16,
            seed=7,
        )
