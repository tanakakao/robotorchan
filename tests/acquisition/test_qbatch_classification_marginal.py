"""q-batch marginal classification active-learning contracts."""

import pytest
import torch

from robotorchan.acquisition import (
    MarginUncertainty,
    PredictiveEntropy,
    qMarginUncertainty,
    qPredictiveEntropy,
)
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.models.classification import BinarySingleTaskGPClassifier


@pytest.mark.parametrize(
    ("batch_type", "point_type"),
    [(qPredictiveEntropy, PredictiveEntropy), (qMarginUncertainty, MarginUncertainty)],
)
@pytest.mark.parametrize("q", [1, 2, 4])
def test_q_classification_scores_are_mean_marginals(
    batch_type: type, point_type: type, q: int
) -> None:
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    X = torch.linspace(0.1, 0.9, q, dtype=torch.double).reshape(1, q, 1)
    expected = point_type(model)(X.transpose(0, 1)).mean()
    actual = batch_type(model)(X)
    assert actual.shape == torch.Size([1])
    torch.testing.assert_close(actual.squeeze(0), expected)
    assert check_acquisition_optimizer_compatibility(
        batch_type.__name__, CandidateOptimizer.CONTINUOUS, q=q
    ).compatible
