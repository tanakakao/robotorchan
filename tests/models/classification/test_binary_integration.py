"""End-to-end contract tests for the standard binary GP classifier."""

import torch

from robotorchan.models.classification import BinarySingleTaskGPClassifier


def test_standard_binary_classifier_end_to_end_contract() -> None:
    train_X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_Y = (train_X.squeeze(-1) >= 0.5).double()
    model = BinarySingleTaskGPClassifier(train_X, train_Y)

    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    latent_output = model.model(train_X)
    loss = -model.make_mll()(latent_output, train_Y)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    model.eval()
    model.likelihood.eval()
    X = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    latent = model.latent_posterior(X)
    probabilities = model.predict_proba(X)
    classes = model.predict_class(X)
    latent_variance = model.latent_variance(X)
    predictive_variance = model.predictive_variance(X)
    entropy = model.predictive_entropy(X)
    latent_samples = model.sample_latent(X, torch.Size([4]))
    probability_samples = model.sample_class_probabilities(X, torch.Size([4]))

    assert latent.mean.shape == torch.Size([3, 1])
    assert latent.variance.shape == torch.Size([3, 1])
    assert probabilities.shape == torch.Size([3, 2])
    assert classes.shape == torch.Size([3])
    assert latent_variance.shape == torch.Size([3, 1])
    assert predictive_variance.shape == torch.Size([3, 2])
    assert entropy.shape == torch.Size([3])
    assert latent_samples.shape == torch.Size([4, 3, 1])
    assert probability_samples.shape == torch.Size([4, 3, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, dtype=torch.double))
    torch.testing.assert_close(
        probability_samples.sum(dim=-1),
        torch.ones(4, 3, dtype=torch.double),
    )
    assert torch.isfinite(latent.mean).all()
    assert torch.isfinite(latent_variance).all()
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(predictive_variance).all()
    assert torch.isfinite(entropy).all()


def test_standard_binary_classifier_preserves_raw_label_contract() -> None:
    train_X = torch.rand(6, 2)
    train_Y = torch.tensor([False, True, False, True, True, False])
    model = BinarySingleTaskGPClassifier(train_X, train_Y)
    assert model.raw_train_Y.dtype == torch.bool
    assert model.raw_train_Y.shape == torch.Size([6])
    torch.testing.assert_close(model.raw_train_Y, train_Y)
