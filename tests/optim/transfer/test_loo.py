"""Focused leave-one-out tests for transfer RGPE."""

from types import SimpleNamespace

import pytest
import torch

from robotorchan.optim.transfer import loo


def test_loo_excludes_held_out_observation(monkeypatch):
    observed = []

    class FakeGP:
        def __init__(self, x, y):
            observed.append(x.flatten().tolist())
            self.likelihood = object()

        def eval(self):
            return self

        def posterior(self, x):
            return SimpleNamespace(mean=x[:, :1] + 2)

    monkeypatch.setattr(loo, "SingleTaskGP", FakeGP)
    monkeypatch.setattr(loo, "ExactMarginalLogLikelihood", lambda *_: object())
    monkeypatch.setattr(loo, "fit_gpytorch_mll", lambda *_: None)

    x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    y = torch.tensor([2.0, 3.0, 4.0], dtype=torch.double)
    result = loo.target_loo_predictions(x, y)

    assert observed == [[1.0, 2.0], [0.0, 2.0], [0.0, 1.0]]
    assert torch.equal(result, y)


def test_loo_requires_three_observations():
    with pytest.raises(ValueError, match="n >= 3"):
        loo.target_loo_predictions(torch.zeros(2, 1), torch.zeros(2))
