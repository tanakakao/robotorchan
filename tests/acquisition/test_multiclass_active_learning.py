"""Multiclass active-learning acquisition integration tests."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
    qBALD,
    qMarginUncertainty,
    qPredictiveEntropy,
    qProbabilityVariance,
)
from robotorchan.models.classification import MulticlassSingleTaskGPClassifier


@pytest.fixture
def model() -> MulticlassSingleTaskGPClassifier:
    torch.manual_seed(23)
    X = torch.rand(15, 2, dtype=torch.double)
    y = (torch.arange(15) % 3).long()
    classifier = MulticlassSingleTaskGPClassifier(X, y, num_classes=3, inducing_points=6)
    classifier.fit(epochs=2, lr=0.01)
    return classifier


def test_multiclass_active_learning_q_one(model: MulticlassSingleTaskGPClassifier) -> None:
    X = model.raw_train_X[:4].unsqueeze(-2)
    for acquisition in (
        PredictiveEntropy(model),
        MarginUncertainty(model),
        ProbabilityVariance(model, num_samples=8),
        BALD(model, num_samples=8),
    ):
        scores = acquisition(X)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()


def test_multiclass_q_acquisitions(model: MulticlassSingleTaskGPClassifier) -> None:
    X = model.raw_train_X[:6].reshape(2, 3, 2)
    for acquisition in (
        qPredictiveEntropy(model),
        qMarginUncertainty(model),
        qProbabilityVariance(model, num_samples=8),
        qBALD(model, num_samples=8),
    ):
        scores = acquisition(X)
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()
    with pytest.raises(ValueError, match="num_classes"):
        qBALD(model, num_samples=8)(model.raw_train_X[:12].unsqueeze(0))
