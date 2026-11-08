"""Focused tests for global TuRBO-m Thompson competition."""

import torch

from robotorchan.optim.trust_region import TuRBOMultiRegionState


class _Posterior:
    def __init__(self, value: float, n: int) -> None:
        self.value = value
        self.n = n

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:
        return torch.full((sample_shape[0], self.n, 1), self.value)


class _Model:
    def __init__(self, value: float) -> None:
        self.value = value

    def posterior(self, X: torch.Tensor) -> _Posterior:
        return _Posterior(self.value, X.shape[0])


def test_global_competition_prefers_highest_region_and_returns_ids():
    bounds = torch.tensor([[0.0], [1.0]])
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.2], [0.8]]))
    manager.set_model(0, _Model(0.0))
    manager.set_model(1, _Model(1.0))
    X, region_ids = manager.thompson_sample(q=3, n_candidates=4, seed=123)
    assert X.shape == (3, 1)
    assert region_ids.tolist() == [1, 1, 1]
    assert torch.all((X >= 0) & (X <= 1))


def test_competition_can_allocate_across_regions():
    bounds = torch.tensor([[0.0], [1.0]])
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.2], [0.8]]))
    manager.set_model(0, _Model(2.0))
    manager.set_model(1, _Model(1.0))
    _, region_ids = manager.thompson_sample(q=3, n_candidates=2, seed=123)
    assert region_ids.tolist() == [0, 0, 1]


def test_batch_allocation_tracks_winners_and_respects_region_capacity():
    bounds = torch.tensor([[0.0], [1.0]])
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.2], [0.8]]))
    manager.set_model(0, _Model(2.0))
    manager.set_model(1, _Model(1.0))
    X, region_ids, counts = manager.propose_batch(q=4, n_candidates=4, max_per_region=2, seed=123)
    assert X.shape == (4, 1)
    assert region_ids.tolist() == [0, 0, 1, 1]
    assert counts.tolist() == [2, 2]
    manager.update_regions(X, torch.ones((4, 1)), region_ids)
    assert manager.region(0).train_X.shape[0] == 2
    assert manager.region(1).train_X.shape[0] == 2
