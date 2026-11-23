"""Spectral-mixture classification smoke tests and candidate scoring."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    SpectralMixtureMulticlassGPClassifier,
    SpectralMixtureOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_class",
    [SpectralMixtureMulticlassGPClassifier, SpectralMixtureOrdinalGPClassifier],
)
def test_spectral_mixture_fit_and_candidate_scoring(model_class):
    torch.manual_seed(12)
    x = torch.linspace(-1.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    y = torch.bucketize(x.squeeze(-1), torch.tensor([-0.3, 0.3], dtype=x.dtype))
    model = model_class(x, y, num_classes=3, num_mixtures=2)
    model.fit(epochs=2, lr=0.01)
    candidates = torch.linspace(-0.8, 0.8, 7, dtype=x.dtype).unsqueeze(-1)
    probabilities = model.predict_proba(candidates)
    assert probabilities.shape == (7, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(7, dtype=x.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    scores = PredictiveEntropy(model)(candidates.unsqueeze(-2))
    assert scores.shape == (7,)
    assert torch.isfinite(scores).all()
    proposed = candidates[scores.argmax()]
    assert proposed.shape == (1,)


@pytest.mark.parametrize(
    "model_class",
    [SpectralMixtureMulticlassGPClassifier, SpectralMixtureOrdinalGPClassifier],
)
def test_spectral_mixture_rejects_invalid_mixture_count(model_class):
    x = torch.linspace(-1.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="num_mixtures"):
        model_class(x, y, num_classes=3, num_mixtures=0)
