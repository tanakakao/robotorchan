"""Tests for random acquisition-function search."""

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import RandomSearchStrategy


def _make_acquisition(dtype: torch.dtype = torch.double) -> PosteriorMean:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=dtype)
    train_Y = -((train_X - 0.8) ** 2)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    return PosteriorMean(model)


def test_random_search_validates_configuration() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="num_samples"):
        RandomSearchStrategy(bounds, num_samples=0)


def test_random_search_validates_q() -> None:
    strategy = RandomSearchStrategy(torch.tensor([[0.0], [1.0]]), num_samples=4)
    acq = _make_acquisition(dtype=torch.float)

    with pytest.raises(ValueError, match="at least 1"):
        strategy.optimize(acq, q=0)
    with pytest.raises(ValueError, match="num_samples"):
        strategy.optimize(acq, q=5)


def test_random_search_returns_best_sampled_candidates() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _make_acquisition()
    strategy = RandomSearchStrategy(bounds, num_samples=128, seed=17)

    result = strategy.optimize(acq, q=3)

    generator = torch.Generator().manual_seed(17)
    samples = torch.rand(128, 1, dtype=torch.double, generator=generator)
    with torch.no_grad():
        scores = acq(samples.unsqueeze(-2)).squeeze()
    expected_indices = torch.topk(scores, k=3, largest=True, sorted=True).indices

    assert torch.equal(result.candidates, samples[expected_indices])
    assert torch.equal(result.acquisition_value, scores[expected_indices])
    assert result.candidates.shape == torch.Size([3, 1])
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    assert result.metadata == {"num_samples": 128}


def test_random_search_supports_one_sample() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _make_acquisition()
    strategy = RandomSearchStrategy(bounds, num_samples=1, seed=17)

    result = strategy.optimize(acq)

    assert result.candidates.shape == torch.Size([1, 1])
    assert result.acquisition_value is not None
    assert result.acquisition_value.shape == torch.Size([1])


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
