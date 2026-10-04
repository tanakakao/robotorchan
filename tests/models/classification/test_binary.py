"""Tests for the binary variational GP classifier."""

import pytest
import torch
from gpytorch.likelihoods import BernoulliLikelihood
from torch.distributions import Bernoulli

from robotorchan.models.classification import BinarySingleTaskGPClassifier


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
    train_Y = (train_X.squeeze(-1) >= 0.5).float()
    return train_X, train_Y


def test_binary_classifier_uses_bernoulli_likelihood() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    assert isinstance(model.likelihood, BernoulliLikelihood)
    assert model.num_outputs == 1
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_binary_classifier_exposes_latent_and_predictive_paths() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]])
    latent = model.latent_posterior(X)
    predictive = model.predictive_distribution(X)
    probabilities = model.predict_proba(X)
    assert latent.mean.shape == torch.Size([2, 1])
    assert isinstance(predictive, Bernoulli)
    assert probabilities.shape == torch.Size([2, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))


def test_binary_classifier_rejects_noncanonical_labels() -> None:
    train_X, _ = _training_data()
    with pytest.raises(ValueError, match="only 0 and 1"):
        BinarySingleTaskGPClassifier(train_X, torch.tensor([0, 1, 2, 0, 1, 0, 1, 0]))


def test_binary_classifier_validates_prediction_threshold() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(ValueError, match="between 0 and 1"):
        model.predict_class(train_X, threshold=1.1)


def test_binary_classifier_make_mll_uses_bernoulli_likelihood() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    mll = model.make_mll()
    assert mll.likelihood is model.likelihood
    assert mll.num_data == train_X.shape[-2]


def test_binary_classifier_mll_evaluates_training_objective() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    model.train()
    model.likelihood.train()
    mll = model.make_mll()
    latent_output = model.model(train_X)
    objective = mll(latent_output, train_Y)
    assert objective.ndim == 0
    assert torch.isfinite(objective)


def test_binary_classifier_mll_accepts_explicit_dataset_size() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    mll = model.make_mll(num_data=32)
    assert mll.num_data == 32


def test_binary_classifier_mll_rejects_nonpositive_dataset_size() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(ValueError, match="num_data must be positive"):
        model.make_mll(num_data=0)


def test_binary_predict_proba_preserves_batch_shape() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.rand(3, 4, 1)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == torch.Size([3, 4, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, 4))


def test_binary_predict_proba_preserves_dtype_and_device() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X.double(), train_Y.double())
    probabilities = model.predict_proba(torch.tensor([[0.5]], dtype=torch.double))
    assert probabilities.dtype == torch.double
    assert probabilities.device == train_X.device


def test_binary_predict_class_returns_canonical_integer_labels() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    predictions = model.predict_class(torch.tensor([[0.25], [0.75]]))
    assert predictions.shape == torch.Size([2])
    assert predictions.dtype == torch.long
    assert set(predictions.tolist()) <= {0, 1}


def test_binary_predict_class_accepts_boundary_thresholds() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]])
    torch.testing.assert_close(model.predict_class(X, threshold=0.0), torch.ones(2).long())
    torch.testing.assert_close(model.predict_class(X, threshold=1.0), torch.zeros(2).long())


def test_binary_predict_class_rejects_non_numeric_threshold() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    with pytest.raises(TypeError, match="real number"):
        model.predict_class(train_X, threshold="0.5")  # type: ignore[arg-type]


def test_binary_uncertainty_components_have_distinct_shapes() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]])
    latent_variance = model.latent_variance(X)
    predictive_variance = model.predictive_variance(X)
    predictive_entropy = model.predictive_entropy(X)
    assert latent_variance.shape == torch.Size([2, 1])
    assert predictive_variance.shape == torch.Size([2, 2])
    assert predictive_entropy.shape == torch.Size([2])


def test_binary_predictive_variance_matches_bernoulli_variance() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]])
    probabilities = model.predict_proba(X)
    expected = probabilities * (1.0 - probabilities)
    torch.testing.assert_close(model.predictive_variance(X), expected)


def test_binary_predictive_entropy_matches_class_probabilities() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]])
    probabilities = model.predict_proba(X)
    expected = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
    torch.testing.assert_close(model.predictive_entropy(X), expected)


def test_binary_predictive_entropy_is_finite() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    entropy = model.predictive_entropy(torch.tensor([[0.0], [0.5], [1.0]]))
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_binary_latent_sampling_uses_botorch_posterior_shape() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    samples = model.sample_latent(torch.tensor([[0.25], [0.75]]), torch.Size([5]))
    assert samples.shape == torch.Size([5, 2, 1])


def test_binary_probability_sampling_has_class_dimension() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    samples = model.sample_class_probabilities(
        torch.tensor([[0.25], [0.75]]),
        torch.Size([7]),
    )
    assert samples.shape == torch.Size([7, 2, 2])
    torch.testing.assert_close(samples.sum(dim=-1), torch.ones(7, 2))
    assert ((samples >= 0.0) & (samples <= 1.0)).all()


def test_binary_probability_samples_are_linked_latent_samples() -> None:
    train_X, train_Y = _training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.5]])
    probabilities = model.sample_class_probabilities(X, torch.Size([4]))
    positive = probabilities[..., 1]
    negative = probabilities[..., 0]
    torch.testing.assert_close(negative, 1.0 - positive)
