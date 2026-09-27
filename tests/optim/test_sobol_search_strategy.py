"""Tests for Sobol acquisition-function search."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim import SobolSearchStrategy


class _BatchSumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return X.sum(dim=(-2, -1))


def test_sobol_search_is_reproducible() -> None:
    bounds = torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double)
    acq = _BatchSumAcquisition()
    first = SobolSearchStrategy(bounds, num_samples=64, seed=11).optimize(acq, q=2)
    second = SobolSearchStrategy(bounds, num_samples=64, seed=11).optimize(acq, q=2)

    assert torch.equal(first.candidates, second.candidates)
    assert torch.equal(first.acquisition_value, second.acquisition_value)
    assert first.candidates.shape == torch.Size([2, 2])
    assert first.metadata == {"num_samples": 64, "q": 2, "sampler": "sobol"}


def test_sobol_search_preserves_dtype_and_bounds() -> None:
    bounds = torch.tensor([[-2.0, 1.0], [3.0, 4.0]], dtype=torch.double)
    result = SobolSearchStrategy(bounds, num_samples=32, seed=7).optimize(
        _BatchSumAcquisition()
    )

    assert result.candidates.dtype == bounds.dtype
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])


def test_sobol_search_rejects_invalid_q() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    strategy = SobolSearchStrategy(bounds, num_samples=8)
    try:
        strategy.optimize(_BatchSumAcquisition(), q=0)
    except ValueError as error:
        assert "q must be at least 1" in str(error)
    else:
        raise AssertionError("Expected invalid q to be rejected.")
