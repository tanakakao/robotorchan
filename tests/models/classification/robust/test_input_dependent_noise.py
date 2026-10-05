"""Tests for input-dependent binary label noise."""

import pytest
import torch
from torch import nn

from robotorchan.models.classification import (
    ClassificationRobustnessType,
    InputDependentLabelNoiseBinarySingleTaskGPClassifier,
    InputDependentLabelNoiseLikelihood,
)


class LinearNoiseRates(nn.Module):
    """Simple learnable input-dependent rates used by contract tests."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        rate = 0.2 * torch.sigmoid(self.raw_scale) * (0.5 + X[..., 0])
        return torch.stack((rate, 0.5 * rate), dim=-1)


def test_input_dependent_corruption_varies_with_x() -> None:
    train_x = torch.tensor([[0.0], [1.0]])
    likelihood = InputDependentLabelNoiseLikelihood(train_x, LinearNoiseRates())
    clean = torch.tensor([0.25, 0.25])
    observed = likelihood.corrupt_positive_probability(clean, train_x)

    assert observed[0] != observed[1]
    assert torch.all(likelihood.flip_probabilities(train_x) >= 0)


def test_classifier_exposes_clean_and_observed_probabilities() -> None:
    train_x = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    model = InputDependentLabelNoiseBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        noise_rate_model=LinearNoiseRates(),
    )

    clean = model.predict_clean_proba(train_x)
    observed = model.predict_proba(train_x)

    assert clean.shape == observed.shape == torch.Size([6, 2])
    assert not torch.allclose(clean, observed)
    expected = frozenset({ClassificationRobustnessType.INPUT_DEPENDENT_LABEL_NOISE})
    assert model.classification_robustness == expected
    assert model.models_observed_label_process


def test_noise_rate_module_parameters_are_registered() -> None:
    train_x = torch.linspace(0.0, 1.0, 4).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 1, 1])
    model = InputDependentLabelNoiseBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        noise_rate_model=LinearNoiseRates(),
    )

    parameters = dict(model.named_parameters())
    assert "likelihood.noise_rate_model.raw_scale" in parameters


def test_callable_noise_rate_model_is_supported() -> None:
    train_x = torch.tensor([[0.0], [1.0]])

    def rates(X: torch.Tensor) -> torch.Tensor:
        value = 0.1 * torch.ones_like(X[..., 0])
        return torch.stack((value, value), dim=-1)

    likelihood = InputDependentLabelNoiseLikelihood(train_x, rates)
    assert torch.allclose(likelihood.flip_probabilities(train_x), torch.full((2, 2), 0.1))


def test_invalid_noise_rate_shape_is_rejected() -> None:
    train_x = torch.tensor([[0.0], [1.0]])
    likelihood = InputDependentLabelNoiseLikelihood(
        train_x,
        lambda X: torch.zeros_like(X),
    )

    with pytest.raises(ValueError, match="shape"):
        likelihood.flip_probabilities(train_x)


def test_out_of_range_noise_rates_are_rejected() -> None:
    train_x = torch.tensor([[0.0], [1.0]])
    likelihood = InputDependentLabelNoiseLikelihood(
        train_x,
        lambda X: torch.full((*X.shape[:-1], 2), 0.5),
    )

    with pytest.raises(ValueError, match="max_flip_probability"):
        likelihood.flip_probabilities(train_x)
