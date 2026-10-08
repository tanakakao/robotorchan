"""Phase 4: binary GP classifier validation on the fixed heterogeneous benchmark."""

import pytest
import torch

from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier


@pytest.mark.parametrize("seed", [14, 28])
def test_binary_classifier_fit_and_predictive_contract(seed: int) -> None:
    """Fit the Bernoulli ELBO and check latent and predictive contracts."""
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(48, 3, generator=generator, dtype=torch.double)
    validation_X = torch.rand(20, 3, generator=generator, dtype=torch.double)
    labels = observe(train_X, seed=seed + 100).passed
    model = BinarySingleTaskGPClassifier(train_X, labels, inducing_points=12)
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, labels)
    assert model.make_mll().num_data == len(train_X)

    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(12):
        optimizer.zero_grad()
        latent_output = model.model(train_X)
        loss = -model.make_mll()(latent_output, labels.to(dtype=train_X.dtype))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        latent = model.latent_posterior(validation_X)
        probabilities = model.predict_proba(validation_X)
        classes = model.predict_class(validation_X)
        probability_samples = model.sample_class_probabilities(
            validation_X, torch.Size([4])
        )

    assert latent.mean.shape == (20, 1)
    assert latent.variance.shape == (20, 1)
    assert torch.isfinite(latent.mean).all()
    assert torch.isfinite(latent.variance).all()
    assert (latent.variance >= 0).all()
    assert probabilities.shape == (20, 2)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(20, dtype=torch.double)
    )
    assert classes.shape == (20,)
    assert set(classes.unique().tolist()).issubset({0, 1})
    assert probability_samples.shape == (4, 20, 2)
    assert torch.isfinite(probability_samples).all()
    torch.testing.assert_close(
        probability_samples.sum(dim=-1),
        torch.ones(4, 20, dtype=torch.double),
    )

    true_probability = evaluate_truth(validation_X)[2]
    brier_to_truth = (probabilities[:, 1] - true_probability).square().mean()
    assert torch.isfinite(brier_to_truth)
    assert 0 <= brier_to_truth <= 1


def test_binary_classifier_batch_q_shape() -> None:
    """Keep batch/q dimensions distinct from the two-class output dimension."""
    generator = torch.Generator().manual_seed(44)
    train_X = torch.rand(24, 3, generator=generator, dtype=torch.double)
    labels = observe(train_X, seed=144).passed
    model = BinarySingleTaskGPClassifier(train_X, labels, inducing_points=8)
    model.eval()
    model.likelihood.eval()

    candidate_X = torch.rand(2, 3, 3, generator=generator, dtype=torch.double)
    with torch.no_grad():
        latent = model.latent_posterior(candidate_X)
        probabilities = model.predict_proba(candidate_X)
        samples = model.sample_class_probabilities(candidate_X, torch.Size([4]))

    assert latent.mean.shape == (2, 3, 1)
    assert probabilities.shape == (2, 3, 2)
    assert samples.shape == (4, 2, 3, 2)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(2, 3, dtype=torch.double)
    )
