"""Focused tests for the ordinal GP core."""

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
    train_Y = torch.bucketize(
        train_X.squeeze(-1), torch.tensor([0.3, 0.65], dtype=torch.double)
    )
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
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(4, dtype=torch.double)
    )
    assert model.predict_class(test_X).shape == (4,)
    assert model.predictive_distribution(test_X).probs.shape == (4, 3)
    assert model.sample_class_probabilities(
        test_X, sample_shape=torch.Size([3])
    ).shape == (3, 4, 3)
