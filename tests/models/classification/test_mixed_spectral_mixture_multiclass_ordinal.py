"""Mixed spectral-mixture multiclass and ordinal classifier tests."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    MixedSpectralMixtureMulticlassGPClassifier,
    MixedSpectralMixtureOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_class",
    [MixedSpectralMixtureMulticlassGPClassifier, MixedSpectralMixtureOrdinalGPClassifier],
)
def test_mixed_spectral_mixture_fit_and_candidate(model_class):
    torch.manual_seed(42)
    continuous = torch.linspace(-1.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    category = (torch.arange(12) % 2).to(continuous).unsqueeze(-1)
    x = torch.cat((continuous, category), dim=-1)
    y = torch.bucketize(
        continuous.squeeze(-1), torch.tensor([-0.3, 0.3], dtype=continuous.dtype)
    )
    model = model_class(x, y, num_classes=3, cat_dims=[1], num_mixtures=2)
    model.fit(epochs=2, lr=0.01)
    candidates = x[:5]
    probabilities = model.predict_proba(candidates)
    assert probabilities.shape == (5, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(5, dtype=x.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    scores = PredictiveEntropy(model)(candidates.unsqueeze(-2))
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()
    assert candidates[scores.argmax()].shape == (2,)


@pytest.mark.parametrize(
    "model_class",
    [MixedSpectralMixtureMulticlassGPClassifier, MixedSpectralMixtureOrdinalGPClassifier],
)
def test_mixed_spectral_mixture_requires_continuous_feature(model_class):
    x = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="continuous feature"):
        model_class(x, y, num_classes=3, cat_dims=[0])
