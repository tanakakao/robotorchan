"""Smoke tests for multiclass variational GP classification."""

import pytest
import torch

from robotorchan.models.classification import (
    MulticlassSingleTaskGPClassifier,
    get_classification_model_entry,
)


def test_multiclass_fit_and_prediction() -> None:
    torch.manual_seed(3)
    X = torch.rand(18, 2, dtype=torch.double)
    y = torch.arange(18, dtype=torch.long) % 3
    model = MulticlassSingleTaskGPClassifier(X, y, num_classes=3, inducing_points=8)
    model.fit(epochs=2, lr=0.01)
    test_X = X[:4].unsqueeze(0)
    posterior = model.posterior(test_X)
    assert posterior.mean.shape == (1, 4, 3)
    samples = model.sample_class_probabilities(
        test_X, sample_shape=torch.Size([5])
    )
    assert samples.shape == (5, 1, 4, 3)
    assert torch.allclose(samples.sum(dim=-1), torch.ones_like(samples[..., 0]))
    probs = model.predict_proba(test_X, num_samples=16)
    assert probs.shape == (1, 4, 3)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]))
    assert model.predict_class(test_X, num_samples=16).shape == (1, 4)


def test_multiclass_registry_and_invalid_labels() -> None:
    entry = get_classification_model_entry("multiclass.standard")
    assert entry.model_class is MulticlassSingleTaskGPClassifier
    assert entry.num_classes is None
    X = torch.rand(4, 2)
    with pytest.raises(ValueError, match="Class indices"):
        MulticlassSingleTaskGPClassifier(
            X, torch.tensor([0, 1, 2, 3]), num_classes=3
        )
