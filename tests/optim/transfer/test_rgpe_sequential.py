"""Focused tests for sequential RGPE candidate proposals."""

import pytest
import torch

from robotorchan.optim.transfer import rgpe_sequential


def test_sequential_rgpe_proposes_one_candidate(monkeypatch):
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    monkeypatch.setattr(rgpe_sequential, "fit_gpytorch_mll", lambda *_: None)
    monkeypatch.setattr(
        rgpe_sequential,
        "target_loo_predictions",
        lambda train_x, train_y: train_y.reshape(-1),
    )

    def fake_optimize(acq_function, *, bounds, q, num_restarts, raw_samples):
        assert q == 1
        query = torch.tensor([[[0.25]]], dtype=torch.double)
        assert torch.isfinite(acq_function(query)).all()
        return torch.tensor([[0.25]], dtype=torch.double), torch.tensor(0.0)

    monkeypatch.setattr(rgpe_sequential, "optimize_acqf", fake_optimize)
    candidate, weights = rgpe_sequential.propose_rgpe_candidate([], x, y, bounds, n_bootstrap=8)
    assert candidate.shape == (1, 1)
    assert torch.equal(candidate, torch.tensor([[0.25]], dtype=torch.double))
    assert torch.equal(weights, torch.ones(1, dtype=torch.double))


def test_sequential_rgpe_rejects_out_of_bounds_observations():
    x = torch.tensor([[0.0], [0.5], [2.0]])
    y = torch.tensor([0.0, 1.0, 0.0])
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="within bounds"):
        rgpe_sequential.propose_rgpe_candidate([], x, y, bounds)


def test_sequential_rgpe_mixed_uses_discrete_optimizer(monkeypatch):
    x = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [0.2, 1.0]],
        dtype=torch.double,
    )
    y = torch.tensor([[0.0], [1.0], [0.2], [0.7]], dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    assignments = [{1: 0.0}, {1: 1.0}]
    monkeypatch.setattr(rgpe_sequential, "fit_gpytorch_mll", lambda *_: None)
    monkeypatch.setattr(
        rgpe_sequential,
        "target_loo_predictions",
        lambda train_x, train_y, *, cat_dims: train_y.reshape(-1),
    )

    def fake_optimize(
        acq_function, *, bounds, q, num_restarts, raw_samples, fixed_features_list
    ):
        assert q == 1
        assert fixed_features_list == assignments
        assert torch.isfinite(acq_function(x[:1].unsqueeze(0))).all()
        return x[1:2], torch.tensor(0.0)

    monkeypatch.setattr(rgpe_sequential, "optimize_acqf_mixed", fake_optimize)
    candidate, weights = rgpe_sequential.propose_rgpe_candidate(
        [], x, y, bounds, cat_dims=[1], fixed_features_list=assignments, n_bootstrap=8
    )
    assert torch.equal(candidate, x[1:2])
    assert torch.equal(weights, torch.ones(1, dtype=torch.double))


def test_sequential_rgpe_mixed_rejects_unsupported_category():
    x = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 0.0])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="unsupported categorical"):
        rgpe_sequential.propose_rgpe_candidate(
            [], x, y, bounds,
            cat_dims=[1],
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
        )
