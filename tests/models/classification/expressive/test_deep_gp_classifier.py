"""Tests for binary DeepGP classification."""

import pytest
import torch
from gpytorch.mlls import DeepApproximateMLL

from robotorchan.models.classification import (
    BinaryClassificationMixin,
    BinarySingleTaskDeepGPClassifier,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    X = torch.rand(10, 3, dtype=torch.double)
    Y = (X[:, 0] + X[:, 1] > 1.0).to(dtype=torch.double)
    return X, Y


def test_deep_gp_classifier_preserves_latent_and_probability_contracts() -> None:
    train_X, train_Y = _data()
    model = BinarySingleTaskDeepGPClassifier(
        train_X, train_Y, hidden_dims=(2,), num_inducing=4, posterior_samples=4
    )
    posterior = model.posterior(train_X[:3], num_samples=4)
    probabilities = model.predict_proba(train_X[:3], num_samples=4)
    assert posterior.mean.shape == torch.Size([3, 1])
    assert probabilities.shape == torch.Size([3, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, dtype=torch.double))


def test_deep_gp_classifier_uses_bernoulli_deep_elbo() -> None:
    train_X, train_Y = _data()
    model = BinarySingleTaskDeepGPClassifier(
        train_X, train_Y, hidden_dims=(2,), num_inducing=4, posterior_samples=4
    )
    assert isinstance(model.make_mll(), DeepApproximateMLL)
    loss = model.training_loss(num_likelihood_samples=2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_deep_gp_classifier_validates_labels() -> None:
    train_X, train_Y = _data()
    invalid = train_Y.clone()
    invalid[0] = 2.0
    with pytest.raises(ValueError):
        BinarySingleTaskDeepGPClassifier(
            train_X, invalid, hidden_dims=(2,), num_inducing=4, posterior_samples=4
        )


def test_deep_gp_classifier_preserves_raw_labels() -> None:
    train_X, train_Y = _data()
    model = BinarySingleTaskDeepGPClassifier(
        train_X, train_Y, hidden_dims=(2,), num_inducing=4, posterior_samples=4
    )
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_deep_gp_classifier_implements_common_classification_contract() -> None:
    train_X, train_Y = _data()
    model = BinarySingleTaskDeepGPClassifier(
        train_X,
        train_Y,
        hidden_dims=(2,),
        num_inducing=4,
        posterior_samples=4,
    )

    assert isinstance(model, BinaryClassificationMixin)
    assert model.is_classification is True
    assert model.num_classes == 2
    assert model.class_labels == (0, 1)

    latent = model.latent_posterior(train_X[:3], num_samples=4)
    samples = model.sample_class_probabilities(
        train_X[:3],
        sample_shape=torch.Size([5]),
        num_samples=4,
    )

    assert latent.mean.shape == torch.Size([3, 1])
    assert samples.shape == torch.Size([5, 3, 2])
    torch.testing.assert_close(
        samples.sum(dim=-1),
        torch.ones(5, 3, dtype=train_X.dtype),
    )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_deep_gp_classifier_rejects_nonfinite_threshold(threshold: float) -> None:
    train_X, train_Y = _data()
    model = BinarySingleTaskDeepGPClassifier(
        train_X,
        train_Y,
        hidden_dims=(2,),
        num_inducing=4,
        posterior_samples=4,
    )

    with pytest.raises(ValueError, match="finite"):
        model.predict_class(train_X[:2], threshold=threshold, num_samples=4)
