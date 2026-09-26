"""Tests for mixed-variable acquisition optimization."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from robotorchan.models.standard.single_task import MixedSingleTaskGP

from robotorchan.optim import CandidateConstraints, MixedSpaceStrategy


def test_mixed_space_strategy_respects_category_and_linear_constraint() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.0], [0.7, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2 * train_X[:, 1:2]).sin()
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = PosteriorMean(model)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([-1.0], dtype=torch.double),
                -0.6,
            ),
        ),
    )
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=16,
        constraints=constraints,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert result.candidates[0, 0] <= 0.6 + 1e-6
    assert result.candidates[0, 1].item() in {0.0, 1.0}
