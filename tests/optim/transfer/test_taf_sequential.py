"""Focused sequential TAF tests."""

import pytest
import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer import taf_sequential


def test_taf_sequential_candidate_with_source_model(monkeypatch):
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    source = SingleTaskGP(x, y)
    monkeypatch.setattr(taf_sequential, "fit_gpytorch_mll", lambda *_: None)

    def fake_optimize(acq_function, *, bounds, q, num_restarts, raw_samples):
        assert q == 1
        query = torch.tensor([[[0.25]]], dtype=torch.double)
        assert torch.isfinite(acq_function(query)).all()
        return torch.tensor([[0.25]], dtype=torch.double), torch.tensor(0.0)

    monkeypatch.setattr(taf_sequential, "optimize_acqf", fake_optimize)
    candidate = taf_sequential.propose_taf_candidate(
        [source], x, y, bounds, torch.tensor([0.5, 0.5])
    )
    assert torch.equal(candidate, torch.tensor([[0.25]], dtype=torch.double))


def test_taf_sequential_rejects_invalid_bounds():
    x = torch.tensor([[0.0], [0.5], [1.0]])
    y = torch.tensor([[0.0], [1.0], [0.0]])
    bounds = torch.tensor([[0.0], [0.4]])
    with pytest.raises(ValueError, match="within bounds"):
        taf_sequential.propose_taf_candidate([], x, y, bounds, torch.ones(1))
