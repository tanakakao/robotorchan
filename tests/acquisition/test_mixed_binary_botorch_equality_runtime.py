"""Equality-constrained BoTorch mixed optimization for binary classification."""

import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification.binary.standard.single_task import (
    MixedBinarySingleTaskGPClassifier,
)
from robotorchan.optim.backends import optimize_acqf_mixed_botorch
from robotorchan.optim.constraints.contracts import CandidateConstraints


def test_mixed_binary_botorch_mixed_linear_equality() -> None:
    torch.manual_seed(0)
    continuous = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    categories = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.double)
    train_x = torch.stack((continuous, categories), dim=-1)
    train_y = (continuous + categories * 0.2 > 0.6).to(dtype=torch.double)
    model = MixedBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        cat_dims=[1],
        inducing_points=4,
    )
    model.eval()
    acquisition = PredictiveEntropy(model)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        equality_constraints=(
            (
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.double),
                0.5,
            ),
        ),
    )

    candidates, value = optimize_acqf_mixed_botorch(
        acquisition,
        bounds,
        q=1,
        num_restarts=2,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        raw_samples=8,
        options={"maxiter": 10},
        constraints=constraints,
    )

    assert candidates.shape == torch.Size([1, 2])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
    assert abs(candidates[0, 0].item() - 0.5) <= 1e-5
    assert candidates[0, 1].item() in (0.0, 1.0)
