"""Mixed continuous/categorical batch candidate generation with joint Gaussian EPIG."""

import torch
from botorch.models import MixedSingleTaskGP
from botorch.optim import optimize_acqf_mixed

from robotorchan.acquisition import qJointExpectedPredictiveInformationGain


def test_joint_epig_optimizes_mixed_batch() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.sin(4.0 * train_X[:, :1]) + 0.3 * train_X[:, 1:]
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    model.eval()
    target_X = torch.tensor([[0.25, 0.0], [0.75, 1.0]], dtype=torch.double)
    acquisition = qJointExpectedPredictiveInformationGain(model, target_X, observation_noise=1e-3)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    candidates, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        sequential=False,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        options={"maxiter": 30},
    )
    assert candidates.shape == (2, 2)
    assert torch.isfinite(value).all()
    assert ((candidates[:, 0] >= 0) & (candidates[:, 0] <= 1)).all()
    assert ((candidates[:, 1] == 0) | (candidates[:, 1] == 1)).all()
    assert torch.isfinite(acquisition(candidates.unsqueeze(0))).all()
