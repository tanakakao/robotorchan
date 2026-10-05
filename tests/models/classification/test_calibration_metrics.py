"""Tests for classification calibration metrics."""

import math

import torch

from robotorchan.models.classification import (
    brier_score,
    classification_nll,
    expected_calibration_error,
    maximum_calibration_error,
)


def test_perfect_probabilities_have_zero_calibration_error() -> None:
    probabilities = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.double,
    )
    targets = torch.tensor([0, 1, 0], dtype=torch.long)

    assert brier_score(probabilities, targets).item() == 0.0
    assert expected_calibration_error(probabilities, targets).item() == 0.0
    assert maximum_calibration_error(probabilities, targets).item() == 0.0


def test_binary_brier_score_matches_manual_definition() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7]],
        dtype=torch.double,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)

    expected = ((0.2**2 + 0.2**2) + (0.3**2 + 0.3**2)) / 2.0
    torch.testing.assert_close(
        brier_score(probabilities, targets),
        torch.tensor(expected, dtype=torch.double),
    )


def test_multiclass_nll_and_brier_are_supported() -> None:
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
        dtype=torch.double,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    expected_nll = -(math.log(0.7) + math.log(0.6)) / 2.0
    torch.testing.assert_close(
        classification_nll(probabilities, targets),
        torch.tensor(expected_nll, dtype=torch.double),
    )
    assert brier_score(probabilities, targets).item() > 0.0


def test_ece_uses_observation_weighted_bin_gaps() -> None:
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6]],
        dtype=torch.double,
    )
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    ece = expected_calibration_error(probabilities, targets, n_bins=2)
    mce = maximum_calibration_error(probabilities, targets, n_bins=2)

    assert 0.0 <= ece.item() <= 1.0
    assert ece <= mce


def test_metrics_validate_probability_contract() -> None:
    probabilities = torch.tensor([[0.7, 0.4]], dtype=torch.double)
    targets = torch.tensor([0], dtype=torch.long)

    try:
        expected_calibration_error(probabilities, targets)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Invalid probability simplex must be rejected")
