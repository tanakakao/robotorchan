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
from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)


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


def test_mixed_multiclass_active_learning() -> None:
    torch.manual_seed(29)
    X = torch.rand(15, 2, dtype=torch.double)
    X[:, 1] = torch.arange(15) % 3
    y = (torch.arange(15) % 3).long()
    model = MixedMulticlassSingleTaskGPClassifier(
        X, y, num_classes=3, cat_dims=[1], inducing_points=6
    )
    model.fit(epochs=2, lr=0.01)
    single = X[:4].unsqueeze(-2)
    batch = X[:6].reshape(2, 3, 2)
    for acquisition in (
        PredictiveEntropy(model),
        MarginUncertainty(model),
        ProbabilityVariance(model, num_samples=8),
        BALD(model, num_samples=8),
    ):
        scores = acquisition(single)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()
    for acquisition in (
        qPredictiveEntropy(model),
        qMarginUncertainty(model),
        qProbabilityVariance(model, num_samples=8),
        qBALD(model, num_samples=8),
    ):
        scores = acquisition(batch)
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()


def test_multiclass_uncertainty_sample_contract(
    model: MulticlassSingleTaskGPClassifier,
) -> None:
    X = model.raw_train_X[:3].unsqueeze(0)
    variance = model.probability_variance(X, num_samples=8)
    information = model.mutual_information(X, num_samples=8)
    assert variance.shape == (1, 3, 3)
    assert information.shape == (1, 3)
    assert (variance >= 0).all()
    assert (information >= 0).all()
    with pytest.raises(ValueError, match="at least 2"):
        model.probability_variance(X, num_samples=1)
    with pytest.raises(ValueError, match="at least 2"):
        model.mutual_information(X, num_samples=1)
