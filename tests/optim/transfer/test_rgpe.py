"""Focused tests for RGPE ranking-based ensemble weights."""

from types import SimpleNamespace

import pytest
import torch

from robotorchan.optim.transfer.rgpe import rgpe_weights


class DummyModel:
    def __init__(self, values):
        self.values = values

    def posterior(self, x):
        return SimpleNamespace(mean=self.values[: x.shape[0], None])


def test_identical_rankings_share_weight():
    x = torch.tensor([[0.0], [0.5], [1.0]])
    y = torch.tensor([0.0, 1.0, 2.0])
    weights = rgpe_weights(
        [DummyModel(y)], x, y, y, n_bootstrap=16, seed=7
    )
    assert torch.allclose(weights, torch.tensor([0.5, 0.5]))


def test_negative_transfer_source_is_excluded():
    x = torch.tensor([[0.0], [0.5], [1.0]])
    y = torch.tensor([0.0, 1.0, 2.0])
    weights = rgpe_weights(
        [DummyModel(-y)], x, y, y, n_bootstrap=16
    )
    assert torch.equal(weights, torch.tensor([0.0, 1.0]))


def test_tied_target_observations_fall_back_to_target():
    x = torch.tensor([[0.0], [0.5]])
    y = torch.tensor([1.0, 1.0])
    weights = rgpe_weights([DummyModel(y)], x, y, y)
    assert torch.equal(weights, torch.tensor([0.0, 1.0]))


def test_rejects_too_few_target_observations():
    with pytest.raises(ValueError, match="n >= 2"):
        rgpe_weights([], torch.tensor([[0.0]]), torch.tensor([1.0]),
                     torch.tensor([1.0]))
