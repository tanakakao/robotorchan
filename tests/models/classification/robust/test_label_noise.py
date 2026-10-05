"""Tests for binary label-noise classification."""

import pytest
import torch

from robotorchan.models.classification import (
    ClassificationRobustnessType,
    LabelNoiseBernoulliLikelihood,
    LabelNoiseBinarySingleTaskGPClassifier,
)


def test_label_noise_likelihood_applies_asymmetric_flip_formula() -> None:
    likelihood = LabelNoiseBernoulliLikelihood(
        false_positive_rate=0.1,
        false_negative_rate=0.2,
    )
    clean = torch.tensor([0.0, 0.25, 1.0])
    observed = likelihood.corrupt_positive_probability(clean)

    assert torch.allclose(observed, torch.tensor([0.1, 0.275, 0.8]), atol=1e-6)


def test_symmetric_flip_probability_configures_both_rates() -> None:
    train_x = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    model = LabelNoiseBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        flip_probability=0.15,
    )

    assert torch.allclose(model.flip_probabilities, torch.tensor([0.15, 0.15]), atol=1e-6)
    assert model.classification_robustness == frozenset({ClassificationRobustnessType.LABEL_NOISE})
    assert model.models_observed_label_process
    assert model.preserves_latent_classification_posterior
    clean = model.predict_clean_proba(train_x)
    observed = model.predict_proba(train_x)
    expected_positive = 0.15 + 0.7 * clean[..., 1]
    assert torch.allclose(observed[..., 1], expected_positive, atol=1e-5)


def test_learnable_flip_probabilities_receive_gradients() -> None:
    likelihood = LabelNoiseBernoulliLikelihood(
        false_positive_rate=0.1,
        false_negative_rate=0.2,
        learn_flip_probabilities=True,
    )
    loss = likelihood.corrupt_positive_probability(torch.tensor(0.7))
    loss.backward()

    assert likelihood.raw_flip_logits.requires_grad
    assert likelihood.raw_flip_logits.grad is not None


@pytest.mark.parametrize("value", [-0.1, 0.5, float("nan"), float("inf")])
def test_invalid_flip_probability_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        LabelNoiseBernoulliLikelihood(false_positive_rate=value)


def test_symmetric_and_asymmetric_configuration_cannot_be_mixed() -> None:
    train_x = torch.linspace(0.0, 1.0, 4).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError, match="cannot be combined"):
        LabelNoiseBinarySingleTaskGPClassifier(
            train_x,
            train_y,
            flip_probability=0.1,
            false_positive_rate=0.1,
        )
