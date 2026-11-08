"""Linear equality candidate constraints for mixed binary GA optimization."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification.binary.standard.single_task import (
    MixedBinarySingleTaskGPClassifier,
)
from robotorchan.optim.backends import optimize_acqf_mixed_ga
from robotorchan.optim.constraints.contracts import CandidateConstraints


def test_mixed_binary_mixed_ga_categorical_equality() -> None:
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
                torch.tensor([1], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.double),
                1.0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="categorical dimensions"):
        optimize_acqf_mixed_ga(
            acquisition,
            bounds,
            q=1,
            categorical_values={1: [0.0, 1.0]},
            constraints=constraints,
            population_size=16,
            generations=3,
            seed=7,
        )

    assert candidates.shape == torch.Size([1, 2])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
    assert candidates[0, 1].item() == 1.0
