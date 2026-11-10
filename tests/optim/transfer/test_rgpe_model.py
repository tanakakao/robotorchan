"""Focused RGPE posterior model tests."""

import pytest
import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer.rgpe_model import RGPEModel


def test_rgpe_model_matches_single_source_posterior():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    source = SingleTaskGP(x, y)
    target = SingleTaskGP(x, y + 1.0)
    ensemble = RGPEModel([source], target, torch.tensor([1.0, 0.0]))
    query = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    actual = ensemble.posterior(query)
    expected = source.posterior(query)
    assert torch.allclose(actual.mean, expected.mean)
    assert torch.allclose(actual.variance, expected.variance, atol=1e-6)


def test_rgpe_model_rejects_invalid_weights():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = SingleTaskGP(x, x)
    with pytest.raises(ValueError, match="sum to one"):
        RGPEModel([], model, torch.tensor([0.5]))
