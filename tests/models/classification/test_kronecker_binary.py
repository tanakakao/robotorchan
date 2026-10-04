"""Tests for block-design multi-task binary classification."""

import pytest
import torch

from robotorchan.models.classification import KroneckerMultiTaskBinaryGPClassifier


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0, 0], [0, 1], [1, 1]], dtype=torch.double)
    return train_X, train_Y


def test_kronecker_binary_classifier_preserves_block_design_contract() -> None:
    train_X, train_Y = _training_data()
    model = KroneckerMultiTaskBinaryGPClassifier(train_X, train_Y)
    assert model.num_tasks == 2
    assert model.task_feature == 1
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_kronecker_binary_classifier_predicts_all_tasks() -> None:
    train_X, train_Y = _training_data()
    model = KroneckerMultiTaskBinaryGPClassifier(train_X, train_Y)
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    probabilities = model.predict_proba(X)
    classes = model.predict_class(X)
    assert probabilities.shape == torch.Size([2, 2, 2])
    assert classes.shape == torch.Size([2, 2])
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, 2, dtype=torch.double),
    )


def test_kronecker_binary_classifier_validates_block_design_shapes() -> None:
    train_X, train_Y = _training_data()
    with pytest.raises(ValueError, match="shapes n x d and n x m"):
        KroneckerMultiTaskBinaryGPClassifier(train_X.unsqueeze(0), train_Y)
    with pytest.raises(ValueError, match="same number of rows"):
        KroneckerMultiTaskBinaryGPClassifier(train_X[:-1], train_Y)


def test_kronecker_binary_classifier_preserves_boolean_labels() -> None:
    train_X, train_Y = _training_data()
    model = KroneckerMultiTaskBinaryGPClassifier(train_X, train_Y.bool())
    assert model.raw_train_Y.dtype == torch.bool
    torch.testing.assert_close(model.raw_train_Y, train_Y.bool())
