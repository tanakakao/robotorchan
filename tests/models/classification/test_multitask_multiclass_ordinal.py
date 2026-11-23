"""ICM multitask classification smoke tests and candidate proposal."""

import pytest
import torch
from gpytorch.kernels import IndexKernel, ProductKernel

from robotorchan.models.classification import (
    KroneckerMultiTaskMulticlassGPClassifier,
    KroneckerMultiTaskOrdinalGPClassifier,
    MultiTaskMulticlassGPClassifier,
    MultiTaskOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_type", [MultiTaskMulticlassGPClassifier, MultiTaskOrdinalGPClassifier]
)
def test_long_format_task_correlation_fit_probability_candidate(model_type):
    torch.manual_seed(17)
    x = torch.rand(18, 3, dtype=torch.double)
    task = (torch.arange(18) % 2).to(dtype=torch.double).unsqueeze(-1)
    train_X = torch.cat((x, task), dim=-1)
    train_Y = torch.arange(18) % 3
    model = model_type(train_X, train_Y, task_feature=-1, num_classes=3, rank=2)
    kernel = (
        model.class_models[0].model.covar_module
        if isinstance(model, MultiTaskMulticlassGPClassifier)
        else model.model.covar_module
    )
    assert isinstance(kernel, ProductKernel)
    assert any(isinstance(part, IndexKernel) for part in kernel.kernels)
    assert model.num_tasks == 2
    assert model.rank == 2
    model.fit(epochs=2, lr=0.005)
    candidate_X = train_X[:8].clone()
    with torch.no_grad():
        probabilities = model.predict_proba(candidate_X)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        candidate = candidate_X[entropy.argmax()]
    assert probabilities.shape == (8, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(8, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert candidate.shape == (4,)


@pytest.mark.parametrize(
    "model_type", [MultiTaskMulticlassGPClassifier, MultiTaskOrdinalGPClassifier]
)
def test_multitask_rejects_noninteger_and_negative_task_ids(model_type):
    x = torch.rand(9, 3, dtype=torch.double)
    y = torch.arange(9) % 3
    for invalid in (-1.0, 0.5):
        task = torch.zeros(9, 1, dtype=torch.double)
        task[0, 0] = invalid
        with pytest.raises(ValueError, match="task_feature"):
            model_type(torch.cat((x, task), dim=-1), y, task_feature=-1, num_classes=3)


@pytest.mark.parametrize(
    "model_type",
    [KroneckerMultiTaskMulticlassGPClassifier, KroneckerMultiTaskOrdinalGPClassifier],
)
def test_block_design_multitask_fit_predict_proposal(model_type):
    torch.manual_seed(19)
    train_X = torch.rand(12, 2, dtype=torch.double)
    train_Y = torch.stack((torch.arange(12) % 3, (torch.arange(12) + 1) % 3), dim=-1)
    model = model_type(train_X, train_Y, num_classes=3, rank=2)
    assert model.num_tasks == 2
    model.fit(epochs=2, lr=0.005)
    candidates = torch.rand(7, 2, dtype=torch.double)
    with torch.no_grad():
        probabilities = model.predict_proba(candidates)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        next_index = int(entropy.mean(dim=-1).argmax())
        proposed = candidates[next_index]
    assert probabilities.shape == (7, 2, 3)
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(7, 2, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert model.predict_class(candidates).shape == (7, 2)
    assert proposed.shape == (2,)
    # Joint latent posterior is flattened over the task rows.
    assert model.posterior(candidates).mean.shape[-2] == 14
    if isinstance(model, KroneckerMultiTaskOrdinalGPClassifier):
        long_candidates = torch.cat(
            (candidates, torch.zeros(7, 1, dtype=torch.double)), dim=-1
        )
        assert model.predict_proba(long_candidates).shape == (7, 3)
    with pytest.raises(ValueError, match="Expected block-design candidate"):
        model.predict_proba(torch.rand(3, 4, dtype=torch.double))


@pytest.mark.parametrize(
    "model_type",
    [KroneckerMultiTaskMulticlassGPClassifier, KroneckerMultiTaskOrdinalGPClassifier],
)
def test_block_design_requires_matching_observation_grid(model_type):
    x = torch.rand(5, 2, dtype=torch.double)
    with pytest.raises(ValueError, match="train_Y"):
        model_type(x, torch.zeros(4, 2, dtype=torch.long), num_classes=3)


@pytest.mark.parametrize(
    "model_type",
    [KroneckerMultiTaskMulticlassGPClassifier, KroneckerMultiTaskOrdinalGPClassifier],
)
def test_block_design_probability_posterior_and_acquisition_shapes(model_type):
    """Probability posterior uses the same flattened candidate axis as latent GP."""
    torch.manual_seed(31)
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = torch.stack((torch.arange(8) % 3, (torch.arange(8) + 1) % 3), dim=1)
    model = model_type(train_X, train_Y, num_classes=3)
    model.fit(epochs=2, lr=0.005)
    candidates = torch.rand(3, 2, dtype=torch.double)
    with torch.no_grad():
        block_prob = model.predict_proba(candidates)
        probability_posterior = (
            model.probability_posterior(candidates, num_samples=16, seed=11)
            if isinstance(model, KroneckerMultiTaskMulticlassGPClassifier)
            else model.probability_posterior(candidates)
        )
        sampled_prob = model.sample_class_probabilities(candidates, sample_shape=torch.Size([4]))
        entropy = model.predictive_entropy(candidates)
        information = model.mutual_information(candidates, num_samples=4)
    assert block_prob.shape == (3, 2, 3)
    assert probability_posterior.mean.shape == (6, 3)
    assert probability_posterior.variance.shape == (6, 3)
    assert probability_posterior.rsample(torch.Size([2])).shape == (2, 6, 3)
    assert sampled_prob.shape == (4, 3, 2, 3)
    assert entropy.shape == (3, 2)
    assert information.shape == (3, 2)
    if isinstance(model, KroneckerMultiTaskOrdinalGPClassifier):
        torch.testing.assert_close(probability_posterior.mean.reshape(3, 2, 3), block_prob)
