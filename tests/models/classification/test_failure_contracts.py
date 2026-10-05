"""Failure-contract tests for classification models and acquisitions."""

import math

import pytest
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior

from robotorchan.acquisition import BALD, LatentStraddle, PredictiveEntropy, ProbabilityVariance
from robotorchan.models import SingleTaskGP
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
    PCABinarySingleTaskGPClassifier,
    validate_binary_labels,
)


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.double)
    return train_X, train_Y


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor(1.0),
        torch.empty(2, 0),
    ],
)
def test_binary_label_validation_rejects_missing_observation_axis(
    labels: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        validate_binary_labels(labels)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_binary_predict_class_rejects_nonfinite_threshold(threshold: float) -> None:
    train_X, train_Y = _binary_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(ValueError, match="finite"):
        model.predict_class(train_X, threshold=threshold)


@pytest.mark.parametrize("num_samples", [0, 1, -1])
def test_sampling_active_learning_rejects_insufficient_samples(num_samples: int) -> None:
    train_X, train_Y = _binary_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(ValueError, match="at least 2"):
        ProbabilityVariance(model, num_samples=num_samples)
    with pytest.raises(ValueError, match="at least 2"):
        BALD(model, num_samples=num_samples)


def test_classification_acquisition_rejects_regression_model_contract() -> None:
    train_X, train_Y = _binary_data()
    regression = SingleTaskGP(train_X, train_Y.unsqueeze(-1))
    with pytest.raises(TypeError, match="classification prediction contract"):
        PredictiveEntropy(regression)


def test_latent_straddle_rejects_negative_beta() -> None:
    train_X, train_Y = _binary_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(ValueError, match="non-negative"):
        LatentStraddle(model, beta=-1.0)


@pytest.mark.parametrize("beta", [float("nan"), math.inf, -math.inf])
def test_latent_straddle_rejects_nonfinite_beta(beta: float) -> None:
    train_X, train_Y = _binary_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(ValueError, match="finite"):
        LatentStraddle(model, beta=beta)


def test_multitask_rejects_task_feature_outside_input_dimensions() -> None:
    train_X, train_Y = _binary_data()
    long_X = torch.cat((train_X, torch.zeros_like(train_X)), dim=-1)
    with pytest.raises(ValueError):
        MultiTaskBinaryGPClassifier(long_X, train_Y, task_feature=2)


def test_reduced_classifier_rejects_unknown_prediction_dimension() -> None:
    train_X = torch.rand(8, 4, dtype=torch.double)
    train_Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.double)
    model = PCABinarySingleTaskGPClassifier(train_X, train_Y, n_components=2)
    invalid_X = torch.rand(3, 3, dtype=torch.double)
    with pytest.raises(ValueError, match="Expected final input dimension"):
        model.predict_proba(invalid_X)


def test_binary_classifier_rejects_mismatched_training_rows() -> None:
    train_X, train_Y = _binary_data()
    with pytest.raises(ValueError):
        BinarySingleTaskGPClassifier(train_X[:-1], train_Y)


class _LegacyClassificationModel(Model):
    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform: object | None = None,
        **kwargs: object,
    ) -> Posterior:
        raise NotImplementedError

    def predict_proba(self, X: torch.Tensor, **kwargs: object) -> torch.Tensor:
        positive = torch.full(X.shape[:-1], 0.5, dtype=X.dtype, device=X.device)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def sample_class_probabilities(
        self,
        X: torch.Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        probabilities = self.predict_proba(X)
        shape = torch.Size() if sample_shape is None else sample_shape
        return probabilities.expand(shape + probabilities.shape)

    def latent_posterior(self, X: torch.Tensor, **kwargs: object) -> object:
        raise NotImplementedError


def test_uncertainty_acquisitions_reject_incomplete_extended_contract() -> None:
    model = _LegacyClassificationModel()

    with pytest.raises(TypeError, match="probability_variance"):
        ProbabilityVariance(model)
    with pytest.raises(TypeError, match="mutual_information"):
        BALD(model)
