"""ICM multitask classification smoke tests and candidate proposal."""

import pytest
import torch
from gpytorch.kernels import IndexKernel, ProductKernel

from robotorchan.models.classification import (
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
