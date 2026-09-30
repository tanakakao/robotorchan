"""Tests for random acquisition-function search."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from torch import Tensor

from robotorchan.models import SingleTaskGP
from robotorchan.optim import RandomSearchStrategy


def _make_acquisition(dtype: torch.dtype = torch.double) -> PosteriorMean:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=dtype)
    train_Y = -((train_X - 0.8) ** 2)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    return PosteriorMean(model)


class _BatchSumAcquisition(AcquisitionFunction):
    """Simple acquisition with one scalar value for each q-batch."""

    def __init__(self, dtype: torch.dtype = torch.double) -> None:
        model = _make_acquisition(dtype=dtype).model
        super().__init__(model=model)

    def forward(self, X: Tensor) -> Tensor:
        return X.sum(dim=(-2, -1))


def test_random_search_validates_configuration() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="num_samples"):
        RandomSearchStrategy(bounds, num_samples=0)


def test_random_search_validates_q() -> None:
    strategy = RandomSearchStrategy(torch.tensor([[0.0], [1.0]]), num_samples=4)
    acq = _make_acquisition(dtype=torch.float)

    with pytest.raises(ValueError, match="at least 1"):
        strategy.optimize(acq, q=0)


def test_random_search_returns_best_sampled_point() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _make_acquisition()
    strategy = RandomSearchStrategy(bounds, num_samples=128, seed=17)

    result = strategy.optimize(acq)

    generator = torch.Generator().manual_seed(17)
    samples = torch.rand(128, 1, 1, dtype=torch.double, generator=generator)
    with torch.no_grad():
        scores = acq(samples).reshape(128)
    expected_index = scores.argmax()

    assert torch.equal(result.candidates, samples[expected_index])
    assert torch.equal(result.acquisition_value, scores[expected_index])
    assert result.candidates.shape == torch.Size([1, 1])
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    assert result.metadata == {"num_samples": 128, "q": 1}


def test_random_search_optimizes_joint_q_batches() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    acq = _BatchSumAcquisition()
    strategy = RandomSearchStrategy(bounds, num_samples=64, seed=23)

    result = strategy.optimize(acq, q=3)

    generator = torch.Generator().manual_seed(23)
    batches = torch.rand(64, 3, 2, dtype=torch.double, generator=generator)
    scores = acq(batches)
    expected_index = scores.argmax()

    assert torch.equal(result.candidates, batches[expected_index])
    assert torch.equal(result.acquisition_value, scores[expected_index])
    assert result.candidates.shape == torch.Size([3, 2])
    assert result.acquisition_value.ndim == 0
    assert result.metadata == {"num_samples": 64, "q": 3}


def test_random_search_preserves_float64_dtype_and_device() -> None:
    bounds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    strategy = RandomSearchStrategy(bounds, num_samples=8, seed=5)
    result = strategy.optimize(_BatchSumAcquisition(), q=2)

    assert result.candidates.dtype == bounds.dtype
    assert result.candidates.device == bounds.device
    assert result.acquisition_value is not None
    assert result.acquisition_value.dtype == bounds.dtype
    assert result.acquisition_value.device == bounds.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_search_runs_on_cuda_without_device_transfer() -> None:
    bounds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
        device="cuda",
    )

    class _CudaBatchSum(AcquisitionFunction):
        def __init__(self) -> None:
            model = _make_acquisition().model.to(device=bounds.device)
            super().__init__(model=model)

        def forward(self, X: Tensor) -> Tensor:
            return X.sum(dim=(-2, -1))

    result = RandomSearchStrategy(bounds, num_samples=8, seed=5).optimize(
        _CudaBatchSum(),
        q=2,
    )

    assert result.candidates.device == bounds.device
    assert result.acquisition_value is not None
    assert result.acquisition_value.device == bounds.device


def test_random_search_supports_one_sample() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _make_acquisition()
    strategy = RandomSearchStrategy(bounds, num_samples=1, seed=17)

    result = strategy.optimize(acq)

    assert result.candidates.shape == torch.Size([1, 1])
    assert result.acquisition_value is not None
    assert result.acquisition_value.ndim == 0


def test_random_search_rejects_non_batch_acquisition_output() -> None:
    class _VectorAcquisition(_BatchSumAcquisition):
        def forward(self, X: Tensor) -> Tensor:
            return X.sum(dim=-1)

    strategy = RandomSearchStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        num_samples=8,
        seed=3,
    )

    with pytest.raises(ValueError, match="one scalar value per sampled q-batch"):
        strategy.optimize(_VectorAcquisition(), q=2)


def test_random_search_seed_is_reproducible_without_global_rng_mutation() -> None:
    bounds = torch.tensor([[-2.0, 1.0], [3.0, 4.0]], dtype=torch.double)
    acq = _make_two_dimensional_acquisition()
    first = RandomSearchStrategy(bounds, num_samples=64, seed=9)
    second = RandomSearchStrategy(bounds, num_samples=64, seed=9)

    torch.manual_seed(123)
    expected_global_draw = torch.rand(5)
    torch.manual_seed(123)
    first_result = first.optimize(acq)
    actual_global_draw = torch.rand(5)
    second_result = second.optimize(acq)

    assert torch.equal(actual_global_draw, expected_global_draw)
    assert torch.equal(first_result.candidates, second_result.candidates)
    assert torch.equal(first_result.acquisition_value, second_result.acquisition_value)


def _make_two_dimensional_acquisition() -> PosteriorMean:
    train_X = torch.tensor(
        [[-2.0, 1.0], [0.0, 2.0], [3.0, 4.0]],
        dtype=torch.double,
    )
    train_Y = -(train_X[:, :1] ** 2 + train_X[:, 1:] ** 2)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    return PosteriorMean(model)
