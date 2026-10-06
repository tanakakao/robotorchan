"""Tests for post-hoc classification calibration."""

import torch
from torch.nn import functional as F

from robotorchan.acquisition import BALD, PredictiveEntropy, ProbabilityVariance
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    CalibratedBinaryClassifier,
    TemperatureScalingCalibrator,
)


def test_temperature_one_is_identity() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7]],
        dtype=torch.double,
    )
    calibrator = TemperatureScalingCalibrator(temperature=1.0).double()

    torch.testing.assert_close(calibrator(probabilities), probabilities)


def test_temperature_above_one_softens_probabilities() -> None:
    probabilities = torch.tensor([[0.95, 0.05]], dtype=torch.double)
    calibrator = TemperatureScalingCalibrator(temperature=2.0).double()

    calibrated = calibrator(probabilities)

    assert calibrated[0, 0] < probabilities[0, 0]
    torch.testing.assert_close(calibrated.sum(dim=-1), torch.ones(1, dtype=torch.double))


def test_temperature_fit_reduces_validation_nll() -> None:
    probabilities = torch.tensor(
        [[0.99, 0.01], [0.98, 0.02], [0.02, 0.98], [0.01, 0.99]],
        dtype=torch.double,
    )
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    calibrator = TemperatureScalingCalibrator().double()
    before = F.nll_loss(calibrator(probabilities).log(), targets)

    calibrator.fit(probabilities, targets, max_iter=50)
    after = F.nll_loss(calibrator(probabilities).log(), targets)

    assert after < before
    assert calibrator.temperature.item() > 1.0


def test_calibrated_classifier_routes_active_learning_through_calibration() -> None:
    train_X = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    model = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)
    calibrated = CalibratedBinaryClassifier(
        model,
        TemperatureScalingCalibrator(temperature=2.0).double(),
    )
    X = torch.tensor([[[0.4]], [[0.6]]], dtype=torch.double)

    assert torch.isfinite(PredictiveEntropy(calibrated)(X)).all()
    assert torch.isfinite(ProbabilityVariance(calibrated, num_samples=8)(X)).all()
    assert torch.isfinite(BALD(calibrated, num_samples=8)(X)).all()


def test_calibration_does_not_replace_latent_posterior() -> None:
    train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([0, 1], dtype=torch.long)
    model = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=2)
    calibrated = CalibratedBinaryClassifier(
        model,
        TemperatureScalingCalibrator().double(),
    )
    X = torch.tensor([[0.5]], dtype=torch.double)

    assert calibrated.latent_posterior(X) is not None


def test_calibrated_bald_uses_one_probability_sample_set() -> None:
    train_X = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    model = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)
    calibrated = CalibratedBinaryClassifier(
        model,
        TemperatureScalingCalibrator(temperature=1.5).double(),
    )
    X = torch.tensor([[0.4], [0.6]], dtype=torch.double)

    torch.manual_seed(19)
    probabilities = calibrated.sample_class_probabilities(
        X,
        sample_shape=torch.Size([16]),
    )
    tiny = torch.finfo(probabilities.dtype).tiny
    safe = probabilities.clamp_min(tiny)
    mean_probabilities = probabilities.mean(dim=0)
    expected = (
        -torch.special.xlogy(mean_probabilities, mean_probabilities).sum(dim=-1)
        + torch.special.xlogy(safe, safe).sum(dim=-1).mean(dim=0)
    ).clamp_min(0.0)

    torch.manual_seed(19)
    actual = calibrated.mutual_information(X, num_samples=16)

    torch.testing.assert_close(actual, expected)
