"""Class-transition label-noise GP tests: learning and decision surfaces."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import (
    PredictiveEntropy,
    qPredictiveEntropy,
)
from robotorchan.models.classification import (
    LabelNoiseMulticlassGPClassifier,
    LabelNoiseOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_type", [LabelNoiseMulticlassGPClassifier, LabelNoiseOrdinalGPClassifier]
)
def test_label_noise_fit_clean_observed_probability_and_candidate_proposal(model_type):
    torch.manual_seed(19)
    x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    model = model_type(x, y, num_classes=3, flip_probability=0.2)
    model.fit(epochs=2, lr=0.005)
    pool = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    with torch.no_grad():
        clean = model.predict_clean_proba(pool)
        observed = model.predict_proba(pool)
        posterior = (
            model.probability_posterior(pool, num_samples=16, seed=11)
            if model_type is LabelNoiseMulticlassGPClassifier
            else model.probability_posterior(pool)
        )
        entropy = PredictiveEntropy(model)(pool.unsqueeze(-2))
        q_entropy = qPredictiveEntropy(model)(pool.reshape(3, 2, 1))
    assert clean.shape == observed.shape == (6, 3)
    assert posterior.mean.shape == posterior.variance.shape == (6, 3)
    assert entropy.shape == (6,)
    assert q_entropy.shape == (3,)
    torch.testing.assert_close(observed, clean @ model.transition_matrix, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        observed.sum(dim=-1), torch.ones(6, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert torch.isfinite(posterior.variance).all()
    assert pool[entropy.argmax()].shape == (1,)


@pytest.mark.parametrize(
    "model_type", [LabelNoiseMulticlassGPClassifier, LabelNoiseOrdinalGPClassifier]
)
def test_label_noise_custom_transition_and_validation(model_type):
    x = torch.rand(7, 2, dtype=torch.double)
    y = torch.arange(7) % 3
    transition = torch.tensor(
        [[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.05, 0.05, 0.9]], dtype=torch.double
    )
    model = model_type(x, y, num_classes=3, transition_matrix=transition)
    torch.testing.assert_close(model.transition_matrix, transition)
    with pytest.raises(ValueError, match="rows must sum"):
        model_type(x, y, num_classes=3, transition_matrix=torch.ones(3, 3))
    with pytest.raises(ValueError, match="shape"):
        model_type(x, y, num_classes=3, transition_matrix=torch.eye(2))
    with pytest.raises(ValueError, match="not both"):
        model_type(
            x, y, num_classes=3, transition_matrix=transition, flip_probability=0.1
        )
    with pytest.raises(ValueError, match="flip_probability"):
        model_type(x, y, num_classes=3, flip_probability=-0.1)


def test_ordinal_noisy_likelihood_is_used_in_fit():
    x = torch.linspace(0, 1, 9, dtype=torch.double).unsqueeze(-1)
    y = torch.arange(9) % 3
    model = LabelNoiseOrdinalGPClassifier(x, y, num_classes=3, flip_probability=0.2)
    latent = torch.tensor([0.0, 0.2, 1.0], dtype=torch.double)
    clean = model.likelihood.clean_class_probabilities(latent)
    observed = model.likelihood.class_probabilities(latent)
    torch.testing.assert_close(observed, clean @ model.transition_matrix)
    assert not torch.allclose(clean, observed)
