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


def test_taf_sequential_mixed_candidate(monkeypatch):
    x = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [0.2, 1.0]],
        dtype=torch.double,
    )
    y = torch.tensor([[0.0], [1.0], [0.2], [0.7]], dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    assignments = [{1: 0.0}, {1: 1.0}]
    monkeypatch.setattr(taf_sequential, "fit_gpytorch_mll", lambda *_: None)

    def fake_optimize(acq_function, *, bounds, q, num_restarts, raw_samples, fixed_features_list):
        assert q == 1
        assert fixed_features_list == assignments
        assert torch.isfinite(acq_function(x[:1].unsqueeze(0))).all()
        return x[1:2], torch.tensor(0.0)

    monkeypatch.setattr(taf_sequential, "optimize_acqf_mixed", fake_optimize)
    candidate = taf_sequential.propose_taf_candidate(
        [], x, y, bounds, torch.ones(1), cat_dims=[1], fixed_features_list=assignments
    )
    assert torch.equal(candidate, x[1:2])


def test_taf_sequential_mixed_rejects_unsupported_category():
    x = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 0.0])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="unsupported categorical"):
        taf_sequential.propose_taf_candidate(
            [],
            x,
            y,
            bounds,
            torch.ones(1),
            cat_dims=[1],
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
        )



def test_taf_sequential_greedy_q_two(monkeypatch):
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    monkeypatch.setattr(taf_sequential, "fit_gpytorch_mll", lambda *_: None)
    calls = []

    def fake_optimize(acq_function, *, bounds, q, num_restarts, raw_samples):
        assert q == 1
        value = 0.25 if not calls else 0.75
        calls.append(value)
        return torch.tensor([[value]], dtype=torch.double), torch.tensor(0.0)

    monkeypatch.setattr(taf_sequential, "optimize_acqf", fake_optimize)
    candidate = taf_sequential.propose_taf_candidate(
        [], x, y, bounds, torch.ones(1), q=2
    )
    assert candidate.shape == (2, 1)
    assert torch.equal(candidate[:, 0], torch.tensor([0.25, 0.75], dtype=torch.double))
