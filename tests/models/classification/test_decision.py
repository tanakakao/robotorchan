"""Tests for imbalanced and cost-sensitive classification decisions."""

import torch

from robotorchan.models.classification import (
    binary_cost_sensitive_prediction,
    binary_cost_sensitive_threshold,
    binary_expected_decision_cost,
    inverse_frequency_class_weights,
)


def test_equal_costs_recover_standard_binary_threshold() -> None:
    threshold = binary_cost_sensitive_threshold(
        false_positive_cost=1.0,
        false_negative_cost=1.0,
    )
    assert threshold == 0.5


def test_high_false_negative_cost_lowers_positive_threshold() -> None:
    threshold = binary_cost_sensitive_threshold(
        false_positive_cost=1.0,
        false_negative_cost=9.0,
    )
    assert threshold == 0.1


def test_cost_sensitive_prediction_minimizes_expected_cost() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.95, 0.05]],
        dtype=torch.double,
    )
    expected_cost = binary_expected_decision_cost(
        probabilities,
        false_positive_cost=1.0,
        false_negative_cost=9.0,
    )
    predictions = binary_cost_sensitive_prediction(
        probabilities,
        false_positive_cost=1.0,
        false_negative_cost=9.0,
    )

    torch.testing.assert_close(predictions, expected_cost.argmin(dim=-1))
    torch.testing.assert_close(predictions, torch.tensor([1, 0]))


def test_inverse_frequency_weights_upweight_minority_class() -> None:
    targets = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    weights = inverse_frequency_class_weights(targets)

    assert weights[1] > weights[0]
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_cost_sensitive_decision_preserves_probability_calibration() -> None:
    probabilities = torch.tensor([[0.7, 0.3]], dtype=torch.double)
    original = probabilities.clone()

    binary_cost_sensitive_prediction(
        probabilities,
        false_positive_cost=1.0,
        false_negative_cost=4.0,
    )

    torch.testing.assert_close(probabilities, original)
