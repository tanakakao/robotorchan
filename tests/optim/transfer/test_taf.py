"""Focused tests for transfer acquisition behavior."""

import pytest
import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer.taf import TransferAcquisitionFunction


def test_taf_target_only_matches_expected_improvement():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    target = SingleTaskGP(x, y)
    query = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    acquisition = TransferAcquisitionFunction(
        [target], target, torch.tensor([0.0, 1.0]), best_f=1.0
    )
    assert torch.allclose(acquisition(query), acquisition.target_ei(query))


def test_taf_source_only_matches_positive_mean_improvement():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    source = SingleTaskGP(x, y)
    query = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    acquisition = TransferAcquisitionFunction(
        [source], source, torch.tensor([1.0, 0.0]), best_f=0.2
    )
    expected = (source.posterior(query).mean.squeeze(-1).squeeze(-1) - 0.2).clamp_min(0)
    assert torch.allclose(acquisition(query), expected)


def test_taf_rejects_invalid_weights_and_q():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    model = SingleTaskGP(x, y)
    with pytest.raises(ValueError, match="sum to one"):
        TransferAcquisitionFunction([], model, torch.tensor([0.5]), best_f=0.0)
    acquisition = TransferAcquisitionFunction(
        [], model, torch.tensor([1.0]), best_f=0.0
    )
    with pytest.raises(ValueError, match="q=1"):
        acquisition(torch.tensor([[[0.25], [0.75]]], dtype=torch.double))
