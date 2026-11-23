"""Targeted tests for jointly trained multiclass / ordinal neural GP classifiers."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    JointEncoderMulticlassGPClassifier,
    JointEncoderOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "cls", [JointEncoderMulticlassGPClassifier, JointEncoderOrdinalGPClassifier]
)
def test_joint_encoder_training_and_probability_candidate(cls):
    torch.manual_seed(4)
    x = torch.linspace(-1.0, 1.0, 15, dtype=torch.double).unsqueeze(-1)
    y = torch.bucketize(x.squeeze(-1), torch.tensor([-0.3, 0.3], dtype=x.dtype))
    model = cls(x, y, num_classes=3, latent_dim=2, hidden_dims=(8,))
    before = [p.detach().clone() for p in model.encoder.parameters()]
    model.fit(epochs=2, lr=0.01)
    assert any(
        not torch.allclose(a, b) for a, b in zip(before, model.encoder.parameters(), strict=True)
    )
    candidates = torch.linspace(-0.9, 0.9, 5, dtype=x.dtype).unsqueeze(-1)
    latent = model.posterior(candidates)
    assert latent.mean.shape[-2:] == (5, 3 if cls is JointEncoderMulticlassGPClassifier else 1)
    probabilities = model.predict_proba(candidates)
    assert probabilities.shape == (5, 3)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(5, dtype=x.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    probability_posterior = model.probability_posterior(candidates)
    assert probability_posterior.mean.shape == (5, 3)
    scores = PredictiveEntropy(model)(candidates.unsqueeze(-2))
    assert torch.isfinite(scores).all()
    assert scores.numel() == 5
