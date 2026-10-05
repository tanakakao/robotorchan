"""Tests for classification model contracts."""

import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.binary.base import BinaryClassificationMixin


class _PredictiveDistribution:
    def __init__(self, probs: Tensor) -> None:
        self.probs = probs


class _BinaryStub(BinaryClassificationMixin):
    def posterior(self, X: Tensor, **kwargs: object) -> GPyTorchPosterior:
        mean = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        covariance = torch.eye(X.shape[-2], dtype=X.dtype, device=X.device)
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        resolved_shape = torch.Size() if sample_shape is None else sample_shape
        return probabilities.expand(resolved_shape + probabilities.shape)

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> object:
        return _PredictiveDistribution(self.predict_proba(X, **kwargs)[..., 1])

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        positive = torch.full(X.shape[:-1], 0.25, dtype=X.dtype, device=X.device)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        return (self.predict_proba(X, **kwargs)[..., 1] >= threshold).long()


def test_classification_mixin_is_abstract() -> None:
    with pytest.raises(TypeError):
        ClassificationModelMixin()


def test_binary_metadata_is_explicit() -> None:
    model = _BinaryStub()
    assert model.num_classes == 2
    assert model.class_labels == (0, 1)


def test_latent_posterior_delegates_to_posterior() -> None:
    model = _BinaryStub()
    posterior = model.latent_posterior(torch.zeros(3, 2))
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == torch.Size([3, 1])


def test_predict_proba_keeps_class_dimension() -> None:
    model = _BinaryStub()
    probabilities = model.predict_proba(torch.zeros(4, 2))
    assert probabilities.shape == torch.Size([4, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))


def test_threshold_is_binary_specific() -> None:
    model = _BinaryStub()
    X = torch.zeros(2, 1)
    assert torch.equal(model.predict_class(X), torch.zeros(2, dtype=torch.long))
    assert torch.equal(
        model.predict_class(X, threshold=0.2),
        torch.ones(2, dtype=torch.long),
    )


def test_binary_classification_metadata_is_explicit() -> None:
    model = _BinaryStub()
    metadata = model.classification_metadata
    assert model.is_classification is True
    assert metadata.num_classes == 2
    assert metadata.class_labels == (0, 1)
    assert metadata.likelihood_family is ClassificationLikelihoodFamily.BERNOULLI
    assert metadata.latent_output_structure is LatentOutputStructure.SINGLE


def test_latent_posterior_exposes_botorch_statistics_and_sampling() -> None:
    model = _BinaryStub()
    posterior = model.latent_posterior(torch.zeros(3, 2))
    assert posterior.mean.shape == torch.Size([3, 1])
    assert posterior.variance.shape == torch.Size([3, 1])
    samples = posterior.rsample(torch.Size([5]))
    assert samples.shape == torch.Size([5, 3, 1])


def test_latent_posterior_is_not_class_probability_output() -> None:
    model = _BinaryStub()
    X = torch.zeros(3, 2)
    posterior = model.latent_posterior(X)
    probabilities = model.predict_proba(X)
    assert posterior.mean.shape[-1] == 1
    assert probabilities.shape[-1] == model.num_classes


def test_predictive_distribution_is_distinct_from_latent_posterior() -> None:
    model = _BinaryStub()
    X = torch.zeros(3, 2)
    latent = model.latent_posterior(X)
    predictive = model.predictive_distribution(X)
    assert isinstance(latent, GPyTorchPosterior)
    assert isinstance(predictive, _PredictiveDistribution)
    torch.testing.assert_close(predictive.probs, torch.full((3,), 0.25))


def test_predict_proba_follows_class_label_order() -> None:
    model = _BinaryStub()
    probabilities = model.predict_proba(torch.zeros(2, 1))
    assert model.class_labels == (0, 1)
    torch.testing.assert_close(probabilities[:, 0], torch.full((2,), 0.75))
    torch.testing.assert_close(probabilities[:, 1], torch.full((2,), 0.25))


def test_uncertainty_contract_separates_observation_and_probability_variance() -> None:
    model = _BinaryStub()
    X = torch.zeros(3, 2)

    observation_variance = model.predictive_variance(X)
    probability_variance = model.probability_variance(X, num_samples=8)

    assert observation_variance.shape == torch.Size([3, 2])
    assert probability_variance.shape == torch.Size([3, 2])
    assert torch.count_nonzero(probability_variance) == 0
    assert torch.count_nonzero(observation_variance) > 0


def test_mutual_information_is_predictive_minus_expected_entropy() -> None:
    model = _BinaryStub()
    X = torch.zeros(3, 2)

    predictive = model.predictive_entropy(X)
    expected = model.expected_class_entropy(X, num_samples=8)
    mutual_information = model.mutual_information(X, num_samples=8)

    torch.testing.assert_close(expected, predictive)
    torch.testing.assert_close(mutual_information, torch.zeros_like(predictive))


@pytest.mark.parametrize("num_samples", [0, 1, -1])
def test_sampling_uncertainty_requires_two_samples(num_samples: int) -> None:
    model = _BinaryStub()
    X = torch.zeros(2, 1)

    with pytest.raises(ValueError, match="at least 2"):
        model.probability_variance(X, num_samples=num_samples)
    with pytest.raises(ValueError, match="at least 2"):
        model.expected_class_entropy(X, num_samples=num_samples)
    with pytest.raises(ValueError, match="at least 2"):
        model.mutual_information(X, num_samples=num_samples)


class _CountingBinaryStub(_BinaryStub):
    def __init__(self) -> None:
        self.probability_sample_calls = 0

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        self.probability_sample_calls += 1
        return super().sample_class_probabilities(
            X,
            sample_shape=sample_shape,
            **kwargs,
        )


def test_mutual_information_uses_one_shared_probability_sample_set() -> None:
    model = _CountingBinaryStub()
    X = torch.zeros(3, 2)

    mutual_information = model.mutual_information(X, num_samples=8)

    assert model.probability_sample_calls == 1
    torch.testing.assert_close(mutual_information, torch.zeros(3))
