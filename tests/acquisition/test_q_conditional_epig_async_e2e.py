"""Asynchronous joint qEPIG candidate generation with pending evaluations."""

import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from robotorchan.acquisition import qConditionalExpectedPredictiveInformationGain


def test_conditional_epig_optimizes_with_updated_pending_candidates() -> None:
    train_X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)
    train_Y = torch.sin(6.0 * train_X)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    target_X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    acquisition = qConditionalExpectedPredictiveInformationGain(
        model, target_X, observation_noise=1e-3
    )
    acquisition.set_X_pending(torch.tensor([[0.3]], dtype=torch.double))
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 30},
    )
    assert candidates.shape == (2, 1)
    assert torch.isfinite(value).all()
    assert ((candidates >= 0) & (candidates <= 1)).all()
    assert torch.isfinite(acquisition(candidates.unsqueeze(0))).all()
