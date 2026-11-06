"""Joint qEPIG candidate optimization with BoTorch linear inequality constraints."""

import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from robotorchan.acquisition import qJointExpectedPredictiveInformationGain


def test_joint_epig_respects_linear_candidate_constraints() -> None:
    train_X = torch.tensor(
        [[0.0], [0.25], [0.5], [0.75], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.sin(6.0 * train_X)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    target_X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    acquisition = qJointExpectedPredictiveInformationGain(model, target_X, observation_noise=1e-3)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        inequality_constraints=[
            (torch.tensor([0]), torch.tensor([-1.0], dtype=torch.double), -0.8),
        ],
        options={"maxiter": 30},
    )
    assert candidates.shape == (2, 1)
    assert torch.isfinite(value).all()
    assert (candidates >= 0.0).all()
    assert (candidates <= 0.8 + 1e-6).all()
    assert torch.isfinite(acquisition(candidates.unsqueeze(0))).all()
