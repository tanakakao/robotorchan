"""End-to-end batch EPIG candidate optimization with a fitted Gaussian GP."""

import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from robotorchan.acquisition import (
    qExpectedPredictiveInformationGain,
    qJointExpectedPredictiveInformationGain,
)


def test_joint_epig_optimizes_two_distinct_candidates_with_fixed_feature() -> None:
    train_X = torch.tensor(
        [[0.0, 0.5], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5], [1.0, 0.5]],
        dtype=torch.double,
    )
    train_Y = torch.sin(6.0 * train_X[:, :1])
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    target_X = torch.tensor([[0.2, 0.5], [0.8, 0.5]], dtype=torch.double)
    acquisition = qJointExpectedPredictiveInformationGain(
        model, target_X, observation_noise=1e-3
    )
    bounds = torch.tensor([[0.0, 0.5], [1.0, 0.5]], dtype=torch.double)
    candidates, values = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        fixed_features={1: 0.5},
        options={"maxiter": 30},
    )
    assert candidates.shape == (2, 2)
    assert values.isfinite().all()
    torch.testing.assert_close(candidates[:, 1], torch.full((2,), 0.5, dtype=torch.double))
    assert (candidates[:, 0] >= 0).all() and (candidates[:, 0] <= 1).all()

    marginal = qExpectedPredictiveInformationGain(
        model, target_X, observation_noise=1e-3
    )
    assert marginal(candidates.unsqueeze(0)).isfinite().all()
    assert acquisition(candidates.unsqueeze(0)).isfinite().all()
