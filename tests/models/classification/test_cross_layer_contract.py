"""End-to-end cross-layer contracts for binary GP classification."""

import torch
from botorch.optim import optimize_acqf

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    KroneckerMultiTaskBinaryGPClassifier,
)


def _binary_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.05], [0.20], [0.35], [0.65], [0.80], [0.95]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.double)
    return train_X, train_Y


def test_binary_classifier_cross_layer_pipeline_reaches_candidate() -> None:
    train_X, train_Y = _binary_training_data()
    model = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)
    model.model.variational_strategy.variational_params_initialized.fill_(1)
    model.eval()
    model.model.eval()

    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    latent = model.latent_posterior(X)
    probabilities = model.predict_proba(X)
    probability_samples = model.sample_class_probabilities(
        X,
        sample_shape=torch.Size([8]),
    )
    entropy = model.predictive_entropy(X)

    assert latent.mean.shape[-2:] == (2, 1)
    assert probabilities.shape == (2, 2)
    assert probability_samples.shape == (8, 2, 2)
    assert entropy.shape == (2,)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, dtype=torch.double))

    acquisition = PredictiveEntropy(model)
    candidate, value = optimize_acqf(
        acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=8,
    )

    assert candidate.shape == (1, 1)
    assert value.numel() == 1
    assert 0.0 <= candidate.item() <= 1.0


def test_kronecker_classifier_block_contract_covers_latent_and_sampling() -> None:
    train_X, train_Y = _binary_training_data()
    block_Y = torch.stack((train_Y, 1.0 - train_Y), dim=-1)
    model = KroneckerMultiTaskBinaryGPClassifier(train_X, block_Y)
    model.model.variational_strategy.variational_params_initialized.fill_(1)
    model.eval()
    model.model.eval()

    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    latent = model.latent_posterior(X)
    probabilities = model.predict_proba(X)
    samples = model.sample_class_probabilities(X, sample_shape=torch.Size([4]))

    assert latent.mean.shape == (2, 2, 1)
    assert probabilities.shape == (2, 2, 2)
    assert samples.shape == (4, 2, 2, 2)
