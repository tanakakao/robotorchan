"""Nonstationary multiclass and ordinal GP smoke tests."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    MixedNonstationaryMulticlassGPClassifier,
    MixedNonstationaryOrdinalGPClassifier,
    NonstationaryMulticlassGPClassifier,
    NonstationaryOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_class",
    [
        NonstationaryMulticlassGPClassifier,
        NonstationaryOrdinalGPClassifier,
        MixedNonstationaryMulticlassGPClassifier,
        MixedNonstationaryOrdinalGPClassifier,
    ],
)
def test_nonstationary_fit_probability_and_candidate(model_class):
    torch.manual_seed(42)
    x = torch.linspace(-1.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    y = torch.bucketize(x.squeeze(-1), torch.tensor([-0.3, 0.3], dtype=x.dtype))
    is_mixed = model_class in (
        MixedNonstationaryMulticlassGPClassifier,
        MixedNonstationaryOrdinalGPClassifier,
    )
    if is_mixed:
        x = torch.cat((x, (torch.arange(12) % 2).to(x).unsqueeze(-1)), dim=-1)
    kwargs = {"cat_dims": [1]} if is_mixed else {}
    model = model_class(x, y, num_classes=3, **kwargs)
    model.fit(epochs=2, lr=0.01)
    candidates = x[:5]
    probs = model.predict_proba(candidates)
    assert probs.shape == (5, 3)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(5, dtype=x.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    scores = PredictiveEntropy(model)(candidates.unsqueeze(-2))
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()
    selected = candidates[scores.argmax()]
    assert selected.shape == (x.shape[-1],)
    if not is_mixed:
        lengthscales = model.local_lengthscale(candidates)
        assert torch.isfinite(lengthscales).all()
        assert (lengthscales > 0).all()
