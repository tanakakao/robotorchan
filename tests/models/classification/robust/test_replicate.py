"""Tests for repeated-label binary classification."""

import pytest
import torch

from robotorchan.models.classification import (
    ClassificationRobustnessType,
    ReplicateLabelBinarySingleTaskGPClassifier,
)


def _replicate_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    train_y = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return train_x, train_y


def test_replicate_labels_are_preserved_as_training_observations() -> None:
    train_x, train_y = _replicate_data()
    model = ReplicateLabelBinarySingleTaskGPClassifier.from_replicates(train_x, train_y)

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.raw_train_X.shape[0] == 6


def test_replicate_statistics_preserve_disagreement() -> None:
    train_x, train_y = _replicate_data()
    model = ReplicateLabelBinarySingleTaskGPClassifier.from_replicates(train_x, train_y)

    assert torch.equal(model.replicate_counts, torch.tensor([3, 3]))
    assert torch.equal(model.replicate_positive_counts, torch.tensor([2.0, 3.0]))
    assert torch.allclose(model.replicate_positive_rate, torch.tensor([2.0 / 3.0, 1.0]))
    assert model.replicate_disagreement[0] > 0
    assert model.replicate_disagreement[1] == 0
    expected = frozenset({ClassificationRobustnessType.REPLICATE_LABELS})
    assert model.classification_robustness == expected
    assert not model.models_observed_label_process


def test_each_design_condition_requires_replicates() -> None:
    train_x = torch.tensor([[0.0], [0.0], [1.0]])
    train_y = torch.tensor([0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="at least two replicate labels"):
        ReplicateLabelBinarySingleTaskGPClassifier.from_replicates(train_x, train_y)


def test_replicate_classifier_accepts_column_labels() -> None:
    train_x, train_y = _replicate_data()
    model = ReplicateLabelBinarySingleTaskGPClassifier.from_replicates(
        train_x,
        train_y.unsqueeze(-1),
    )

    assert torch.equal(model.replicate_counts, torch.tensor([3, 3]))
