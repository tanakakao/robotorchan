"""Tests for ALEBO binary GP classification."""

import pytest
import torch

from robotorchan.models.classification import ALEBOBinarySingleTaskGPClassifier
from robotorchan.optim import ALEBOStrategy


def _problem() -> tuple[ALEBOStrategy, torch.Tensor, torch.Tensor]:
    bounds = torch.stack([torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)])
    strategy = ALEBOStrategy(bounds, embedding_dim=2, seed=17)
    train_Z = strategy.sample_feasible(10, seed=23)
    train_X = strategy.project(train_Z)
    train_Y = (train_X[:, 0] + train_X[:, 1] > 1.0).to(dtype=torch.double)
    return strategy, train_Z, train_Y


def test_alebo_classifier_preserves_binary_prediction_contract() -> None:
    strategy, train_Z, train_Y = _problem()
    model = ALEBOBinarySingleTaskGPClassifier(
        train_Z,
        train_Y,
        projection=strategy.embedding,
    )
    probabilities = model.predict_proba(train_Z[:4])
    assert probabilities.shape == torch.Size([4, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4, dtype=torch.double))
    assert model.metric.shape == torch.Size([2, 2])


def test_alebo_classifier_elbo_reaches_mahalanobis_metric() -> None:
    strategy, train_Z, train_Y = _problem()
    model = ALEBOBinarySingleTaskGPClassifier(
        train_Z,
        train_Y,
        projection=strategy.embedding,
    )
    output = model.model(train_Z)
    loss = -model.make_mll()(output, train_Y)
    loss.backward()
    gradient = model.mahalanobis_kernel.raw_triu.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_alebo_classifier_rejects_projection_dimension_mismatch() -> None:
    _, train_Z, train_Y = _problem()
    with pytest.raises(ValueError, match="projection rows"):
        ALEBOBinarySingleTaskGPClassifier(
            train_Z,
            train_Y,
            projection=torch.randn(3, 6, dtype=torch.double),
        )
