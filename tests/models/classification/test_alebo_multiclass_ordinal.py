"""ALEBO classification public-contract and candidate-proposal tests."""

import pytest
import torch

from robotorchan.models.classification import (
    ALEBOMulticlassGPClassifier,
    ALEBOOrdinalGPClassifier,
)


@pytest.mark.parametrize("model_type", [ALEBOMulticlassGPClassifier, ALEBOOrdinalGPClassifier])
def test_alebo_fit_probability_and_entropy_candidate(model_type):
    torch.manual_seed(19)
    x = torch.randn(15, 2, dtype=torch.double)
    y = torch.arange(15) % 3
    projection = torch.eye(2, dtype=torch.double)
    model = model_type(x, y, num_classes=3, projection=projection)
    assert model.metric.shape == (2, 2)
    model.fit(epochs=2, lr=0.005)
    candidates = torch.randn(8, 2, dtype=torch.double)
    with torch.no_grad():
        probabilities = model.predict_proba(candidates)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        candidate = candidates[entropy.argmax()]
    assert probabilities.shape == (8, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(8, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert candidate.shape == (2,)


@pytest.mark.parametrize("model_type", [ALEBOMulticlassGPClassifier, ALEBOOrdinalGPClassifier])
def test_alebo_rejects_projection_dimension_mismatch(model_type):
    x = torch.randn(8, 2, dtype=torch.double)
    y = torch.arange(8) % 3
    with pytest.raises(ValueError, match="projection rows"):
        model_type(x, y, num_classes=3, projection=torch.randn(3, 2))
