"""Tests for the common search-strategy contracts."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import SearchResult, SearchStrategy


class DummySearchStrategy(SearchStrategy):
    """Minimal concrete strategy used to test the base contract."""

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        del acq_function
        candidates = self.bounds[0].expand(q, -1).clone()
        return SearchResult(
            candidates=candidates,
            acquisition_value=None,
            optimization_time=0.0,
        )


def test_search_strategy_stores_defensive_bounds_copy() -> None:
    bounds = torch.tensor([[0.0, -1.0], [1.0, 2.0]])
    strategy = DummySearchStrategy(bounds)

    bounds[0, 0] = -10.0

    assert torch.equal(strategy.bounds, torch.tensor([[0.0, -1.0], [1.0, 2.0]]))
    assert strategy.input_dim == 2


@pytest.mark.parametrize(
    "bounds",
    [
        torch.zeros(2),
        torch.zeros(3, 2),
        torch.empty(2, 0),
        torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        torch.tensor([[1.0], [0.0]]),
    ],
)
def test_search_strategy_rejects_invalid_bounds(bounds: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        DummySearchStrategy(bounds)


def test_search_result_requires_candidate_matrix() -> None:
    with pytest.raises(ValueError, match="candidates"):
        SearchResult(
            candidates=torch.zeros(3),
            acquisition_value=None,
            optimization_time=0.0,
        )


def test_search_result_rejects_negative_optimization_time() -> None:
    with pytest.raises(ValueError, match="optimization_time"):
        SearchResult(
            candidates=torch.zeros(1, 3),
            acquisition_value=None,
            optimization_time=-1.0,
        )


def test_dummy_strategy_returns_public_space_candidates() -> None:
    strategy = DummySearchStrategy(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    result = strategy.optimize(None, q=3)  # type: ignore[arg-type]

    assert result.candidates.shape == torch.Size([3, 2])
    assert result.optimization_time == 0.0
    assert result.metadata == {}
