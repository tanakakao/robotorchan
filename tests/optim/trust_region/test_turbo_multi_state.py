"""Focused tests for independent TuRBO-m region storage."""

import pytest
import torch

from robotorchan.optim.trust_region import TuRBOMultiRegionState


def test_regions_have_independent_state_data_and_geometry():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    centers = torch.tensor([[0.2, 0.3], [0.8, 0.7]])
    manager = TuRBOMultiRegionState(bounds, centers)
    assert manager.n_trust_regions == 2
    assert not torch.equal(manager.trust_region_bounds(0), manager.trust_region_bounds(1))

    manager.append_observations(0, torch.tensor([[0.2, 0.3]]), torch.tensor([[1.0]]))
    assert manager.region(0).train_X.shape == (1, 2)
    assert manager.region(1).train_X.shape == (0, 2)
    assert manager.region(0).history_Y.item() == 1.0
    assert manager.region(0).state is not manager.region(1).state

    model = object()
    manager.set_model(0, model)
    assert manager.region(0).model is model
    assert manager.region(1).model is None


def test_region_data_replacement_preserves_history():
    bounds = torch.tensor([[0.0], [1.0]])
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.3], [0.7]]))
    manager.append_observations(0, torch.tensor([[0.3]]), torch.tensor([[1.0]]))
    manager.set_region_data(0, torch.tensor([[0.4]]), torch.tensor([[2.0]]))
    assert manager.region(0).train_Y.item() == 2.0
    assert manager.region(0).history_Y.item() == 1.0


def test_invalid_centers_and_training_data():
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="inside bounds"):
        TuRBOMultiRegionState(bounds, torch.tensor([[1.2]]))
    manager = TuRBOMultiRegionState(bounds, torch.tensor([[0.5]]))
    with pytest.raises(ValueError, match="Y must"):
        manager.append_observations(0, torch.tensor([[0.4]]), torch.empty((0, 1)))
