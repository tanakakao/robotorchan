"""Tests for long-format multi-task binary GP classification."""

import pytest
import torch
from gpytorch.kernels import IndexKernel, ProductKernel

from robotorchan.models.classification import MultiTaskBinaryGPClassifier


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.8, 0.0],
            [0.2, 1.0],
            [0.6, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.double)
    return train_X, train_Y


def test_multitask_binary_classifier_uses_structural_task_kernel() -> None:
    train_X, train_Y = _training_data()
    model = MultiTaskBinaryGPClassifier(train_X, train_Y, task_feature=1)
    covariance = model.model.covar_module
    assert isinstance(covariance, ProductKernel)
    assert any(isinstance(module, IndexKernel) for module in covariance.modules())
    assert model.task_feature == 1
    assert model.num_tasks == 2
    assert model.rank == 1


def test_multitask_binary_classifier_supports_negative_task_feature() -> None:
    train_X, train_Y = _training_data()
    model = MultiTaskBinaryGPClassifier(train_X, train_Y, task_feature=-1)
    assert model.task_feature == 1


def test_multitask_binary_classifier_runs_prediction_contract_for_tasks() -> None:
    train_X, train_Y = _training_data()
    model = MultiTaskBinaryGPClassifier(train_X, train_Y, task_feature=1)
    X = torch.tensor([[0.5, 0.0], [0.5, 1.0]], dtype=torch.double)
    probabilities = model.predict_proba(X)
    samples = model.sample_class_probabilities(X, torch.Size([3]))
    assert probabilities.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([3, 2, 2])
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )


def test_multitask_binary_classifier_validates_task_identifiers() -> None:
    train_X, train_Y = _training_data()
    invalid = train_X.clone()
    invalid[0, 1] = 0.5
    with pytest.raises(ValueError, match="integer task identifiers"):
        MultiTaskBinaryGPClassifier(invalid, train_Y, task_feature=1)

    invalid = train_X.clone()
    invalid[0, 1] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        MultiTaskBinaryGPClassifier(invalid, train_Y, task_feature=1)


def test_multitask_binary_classifier_validates_rank() -> None:
    train_X, train_Y = _training_data()
    with pytest.raises(ValueError, match="rank must be between"):
        MultiTaskBinaryGPClassifier(train_X, train_Y, task_feature=1, rank=3)


def test_multitask_binary_classifier_preserves_raw_long_format_data() -> None:
    train_X, train_Y = _training_data()
    model = MultiTaskBinaryGPClassifier(train_X, train_Y.bool(), task_feature=1)
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y.bool())
