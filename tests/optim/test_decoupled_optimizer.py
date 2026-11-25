"""Candidate and mask enumeration contracts."""

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import create_optimizer, optimize_acqf_decoupled


class MaskedQuadratic(AcquisitionFunction):
    def __init__(self):
        super().__init__(model=SimpleNamespace(num_outputs=2))
        self.X_evaluation_mask = torch.tensor([[True, False]])

    def forward(self, X):
        second = self.X_evaluation_mask[0, 1]
        center = 0.75 if second else 0.25
        return -(X[..., 0, 0] - center).square() + second.to(X)


def test_decoupled_optimizer_selects_mask_and_restores_state():
    acq = MaskedQuadratic()
    original = acq.X_evaluation_mask
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    optimize = create_optimizer("optimize_acqf_decoupled", num_restarts=2, raw_samples=16)
    candidate, selected, value = optimize(acq, bounds, masks)
    torch.testing.assert_close(candidate, bounds.new_tensor([[0.75]]), atol=1e-4, rtol=0)
    assert torch.equal(selected, masks[1:2])
    torch.testing.assert_close(value, bounds.new_tensor(1.0))
    assert acq.X_evaluation_mask is original


def test_decoupled_optimizer_restores_state_on_failure(monkeypatch):
    acq = MaskedQuadratic()
    original = acq.X_evaluation_mask

    def fail(**kwargs):
        raise RuntimeError("optimization failed")

    monkeypatch.setattr("robotorchan.optim.decoupled.optimize_acqf", fail)
    with pytest.raises(RuntimeError, match="optimization failed"):
        optimize_acqf_decoupled(acq, torch.tensor([[0.0], [1.0]]), torch.tensor([[False, True]]))
    assert acq.X_evaluation_mask is original


def test_decoupled_optimizer_rejects_empty_mask():
    with pytest.raises(ValueError, match="nonempty"):
        optimize_acqf_decoupled(
            MaskedQuadratic(), torch.tensor([[0.0], [1.0]]), torch.tensor([[False, False]])
        )
