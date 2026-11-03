"""Nonlinear constrained mixed BoTorch optimization for binary classification."""

import torch
from torch import Tensor

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification.binary.standard.single_task import (
    MixedBinarySingleTaskGPClassifier,
)
from robotorchan.optim.backends import optimize_acqf_mixed_botorch
from robotorchan.optim.constraints.contracts import CandidateConstraints


def test_mixed_binary_botorch_mixed_nonlinear_inequality() -> None:
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

    def nonlinear_feasibility(x: Tensor) -> Tensor:
        return x[0].square() - 0.25

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((nonlinear_feasibility, True),),
    )
    initial = torch.tensor(
        [[[0.6, 0.0]], [[0.8, 1.0]]],
        dtype=torch.double,
    )
    candidates, value = optimize_acqf_mixed_botorch(
        acquisition,
        bounds,
        q=1,
        num_restarts=2,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        raw_samples=None,
        options={"maxiter": 10},
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert candidates.shape == torch.Size([1, 2])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
    assert nonlinear_feasibility(candidates[0]).item() >= -1e-6
    assert candidates[0, 1].item() in (0.0, 1.0)
