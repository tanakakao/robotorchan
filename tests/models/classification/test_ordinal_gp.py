"""Focused tests for the ordinal GP core."""

import pytest
import torch

from robotorchan.models.classification import OrdinalSingleTaskGPClassifier
from robotorchan.models.classification.ordinal.likelihood import OrderedProbitLikelihood
from robotorchan.models.classification.registry import get_classification_model_class


def test_ordered_probit_thresholds_and_probabilities() -> None:
    likelihood = OrderedProbitLikelihood(5).double()
    with torch.no_grad():
        likelihood.raw_increments.copy_(torch.tensor([-20.0, 0.0, 10.0]))
    thresholds = likelihood.thresholds
    assert torch.all(thresholds[1:] > thresholds[:-1])
    probabilities = likelihood.class_probabilities(torch.linspace(-3, 3, 7))
    assert probabilities.shape == (7, 5)
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(7, dtype=torch.double))


def test_ordinal_gp_fit_prediction_and_registry() -> None:
    torch.manual_seed(7)
    train_X = torch.linspace(0, 1, 15, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.bucketize(train_X.squeeze(-1), torch.tensor([0.3, 0.65], dtype=torch.double))
    model = OrdinalSingleTaskGPClassifier(train_X, train_Y, num_classes=3)
    assert get_classification_model_class("ordinal.standard") is type(model)
    assert torch.equal(model.raw_train_Y, train_Y)
    mll = model.make_mll()
    model.train()
    output = model.model(train_X)
    loss = -mll(output, train_Y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.likelihood.raw_increments.grad is not None
    assert torch.isfinite(model.likelihood.raw_increments.grad).all()
    model.fit(epochs=2, lr=0.01)
    test_X = train_X[:4]
    posterior = model.posterior(test_X)
    assert posterior.mean.shape == (4, 1)
    probabilities = model.predict_proba(test_X)
    assert probabilities.shape == (4, 3)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4, dtype=torch.double))
    assert model.predict_class(test_X).shape == (4,)
    assert model.predictive_distribution(test_X).probs.shape == (4, 3)
    assert model.sample_class_probabilities(test_X, sample_shape=torch.Size([3])).shape == (3, 4, 3)


def test_ordered_probit_preserves_stochastic_class_order() -> None:
    """Higher latent quality must shift mass toward higher ordered labels."""
    likelihood = OrderedProbitLikelihood(5).double()
    latent = torch.linspace(-6.0, 6.0, 13, dtype=torch.double)
    probabilities = likelihood.class_probabilities(latent)

    assert probabilities.shape == (13, 5)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(13, dtype=torch.double))
    cumulative = probabilities.cumsum(-1)[:, :-1]
    assert (cumulative[1:] <= cumulative[:-1] + 1e-12).all()


def test_ordinal_four_class_gaussian_marginalization_matches_sampling() -> None:
    """Analytic predictive class probabilities integrate latent uncertainty."""
    torch.manual_seed(37)
    X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    Y = torch.bucketize(X[:, 0].contiguous(), torch.tensor([0.25, 0.5, 0.75], dtype=torch.double))
    model = OrdinalSingleTaskGPClassifier(X, Y, num_classes=4, inducing_points=6)
    model.fit(epochs=2, lr=0.01)

    test_X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double)
    analytic = model.predict_proba(test_X)
    torch.manual_seed(91)
    monte_carlo = model.sample_class_probabilities(
        test_X, sample_shape=torch.Size([4096])
    ).mean(dim=0)

    assert model.class_labels == (0, 1, 2, 3)
    assert model.posterior(test_X).mean.shape == (3, 1)
    assert model.predictive_distribution(test_X).probs.shape == (3, 4)
    assert analytic.shape == (3, 4)
    assert torch.isfinite(analytic).all()
    torch.testing.assert_close(analytic.sum(-1), torch.ones(3, dtype=torch.double))
    torch.testing.assert_close(analytic, monte_carlo, atol=0.035, rtol=0.0)


def test_ordinal_rejects_invalid_class_labels() -> None:
    X = torch.linspace(0.0, 1.0, 5, dtype=torch.double).unsqueeze(-1)
    good = torch.tensor([0, 1, 2, 0, 1])

    with pytest.raises(ValueError, match="at least 3"):
        OrdinalSingleTaskGPClassifier(X, good, num_classes=2)
    with pytest.raises(TypeError, match="integer dtype"):
        OrdinalSingleTaskGPClassifier(X, good.double(), num_classes=3)
    with pytest.raises(ValueError, match="Class indices"):
        OrdinalSingleTaskGPClassifier(X, good - 1, num_classes=3)
    with pytest.raises(ValueError, match="Class indices"):
        OrdinalSingleTaskGPClassifier(X, good + 2, num_classes=3)
    with pytest.raises(ValueError, match="shape"):
        OrdinalSingleTaskGPClassifier(X, good[:-1], num_classes=3)
