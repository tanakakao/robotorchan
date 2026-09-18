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
        )


def test_search_result_accepts_scalar_joint_acquisition_value() -> None:
    result = SearchResult(
        candidates=torch.zeros(3, 2),
        acquisition_value=torch.tensor(1.5),
    )

    assert result.acquisition_value is not None
    assert result.acquisition_value.ndim == 0


def test_search_result_rejects_per_candidate_acquisition_values() -> None:
    with pytest.raises(ValueError, match="scalar"):
        SearchResult(
            candidates=torch.zeros(3, 2),
            acquisition_value=torch.tensor([1.0, 2.0, 3.0]),
        )


def test_search_result_contains_only_search_outputs() -> None:
    result = SearchResult(
        candidates=torch.zeros(1, 3),
        acquisition_value=None,
    )

    assert set(result.__dataclass_fields__) == {
        "candidates",
        "acquisition_value",
        "metadata",
    }


def test_search_strategy_preserves_bounds_dtype_and_device() -> None:
    bounds = torch.tensor(
        [[0.0, -1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    strategy = DummySearchStrategy(bounds)

    assert strategy.bounds.dtype == bounds.dtype
    assert strategy.bounds.device == bounds.device


def test_search_result_preserves_candidate_dtype_and_device() -> None:
    candidates = torch.zeros(2, 3, dtype=torch.float64)
    acquisition_value = torch.tensor(1.0, dtype=torch.float64)
    result = SearchResult(
        candidates=candidates,
        acquisition_value=acquisition_value,
    )

    assert result.candidates.dtype == candidates.dtype
    assert result.candidates.device == candidates.device
    assert result.acquisition_value is not None
    assert result.acquisition_value.dtype == acquisition_value.dtype
    assert result.acquisition_value.device == acquisition_value.device


def test_dummy_strategy_returns_public_space_candidates() -> None:
    strategy = DummySearchStrategy(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    result = strategy.optimize(None, q=3)  # type: ignore[arg-type]

    assert result.candidates.shape == torch.Size([3, 2])
    assert result.metadata == {}


def test_public_optim_exports_are_complete() -> None:
    import robotorchan.optim as optim

    expected = {
        "ALEBOStrategy",
        "SearchResult",
        "SearchStrategy",
        "OriginalSpaceStrategy",
        "RandomSearchStrategy",
        "LatentReconstruction",
        "LatentSpaceStrategy",
        "PCAReconstruction",
        "RandomProjectionReconstruction",
        "REMBOStrategy",
        "HeSBOStrategy",
        "TuRBOState",
        "TuRBOStrategy",
        "update_turbo_state",
        "BAxUSState",
        "BAxUSStrategy",
        "BAxUSThompsonSamplingStrategy",
        "update_baxus_state",
    }

    assert set(optim.__all__) == expected
    assert all(hasattr(optim, name) for name in expected)
