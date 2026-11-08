"""Focused TuRBO-m region update and restart tests."""

from dataclasses import replace

import torch

from robotorchan.optim.trust_region import TuRBOMultiRegionState


def test_updates_only_attributed_regions_and_counts_failed_points():
    bounds = torch.tensor([[0.0], [1.0]])
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.2], [0.8]]))
    manager.update_regions(
        torch.tensor([[0.3], [0.7]]),
        torch.tensor([[2.0], [1.0]]),
        torch.tensor([0, 1]),
    )
    manager.update_regions(
        torch.tensor([[0.4], [0.5]]),
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([0, 0]),
    )
    assert manager.region(0).state.failure_counter == 2
    assert manager.region(1).state.failure_counter == 0
    assert manager.region(0).center.item() == torch.tensor(0.3).item()
    assert manager.region(1).train_X.shape == (1, 1)


def test_restart_clears_only_active_region_and_keeps_history():
    bounds = torch.tensor([[0.0], [1.0]])
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.2], [0.8]]))
    manager.update_regions(
        torch.tensor([[0.2], [0.8]]),
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([0, 1]),
    )
    region = manager.region(0)
    region.state = replace(region.state, restart_triggered=True)
    manager.restart_region(0, center=torch.tensor([0.4]))
    assert region.state.restart_count == 1
    assert region.state.best_value == float("-inf")
    assert region.train_X.shape == (0, 1)
    assert region.history_X.shape == (1, 1)
    assert manager.region(1).train_X.shape == (1, 1)
    assert manager.region(1).state.best_value == 2.0
