"""Tests for the binary variational GP classifier."""

import pytest
import torch
from torch.distributions import Bernoulli
from gpytorch.likelihoods import BernoulliLikelihood

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
