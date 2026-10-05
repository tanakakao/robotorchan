"""Tests for contaminated binary classification."""

import pytest
import torch

from robotorchan.models.classification import (
    ClassificationRobustnessType,
    ContaminatedBernoulliLikelihood,
    ContaminatedBinarySingleTaskGPClassifier,
)


def test_contamination_mixture_formula() -> None:
    likelihood = ContaminatedBernoulliLikelihood(
        contamination_probability=0.2,
        contaminant_positive_probability=0.75,
    )
    clean = torch.tensor([0.0, 0.25, 1.0])
    observed = likelihood.contaminate_positive_probability(clean)

    assert torch.allclose(observed, torch.tensor([0.15, 0.35, 0.95]), atol=1e-6)


def test_contaminated_classifier_exposes_clean_and_observed_probabilities() -> None:
    train_x = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    model = ContaminatedBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        contamination_probability=0.2,
        contaminant_positive_probability=0.5,
    )

    clean = model.predict_clean_proba(train_x)
    observed = model.predict_proba(train_x)
    expected_positive = 0.8 * clean[..., 1] + 0.1

    assert torch.allclose(observed[..., 1], expected_positive, atol=1e-5)
    assert model.classification_robustness == frozenset({ClassificationRobustnessType.CONTAMINATION})
    assert model.models_observed_label_process
    assert model.preserves_latent_classification_posterior


def test_learnable_contamination_probability_receives_gradient() -> None:
    likelihood = ContaminatedBernoulliLikelihood(
        contamination_probability=0.1,
        learn_contamination_probability=True,
    )
    loss = likelihood.contaminate_positive_probability(torch.tensor(0.8))
    loss.backward()

    assert likelihood.raw_contamination_logit.requires_grad
    assert likelihood.raw_contamination_logit.grad is not None


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, float("nan"), float("inf")])
def test_invalid_contamination_probability_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        ContaminatedBernoulliLikelihood(contamination_probability=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_contaminant_probability_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        ContaminatedBernoulliLikelihood(contaminant_positive_probability=value)
