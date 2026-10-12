"""Check kernel label predictor probability and benchmark reference metrics."""

import torch

from robotorchan.benchmarks.active_learning_problems import (
    brier_score,
    classification_accuracy,
)
from robotorchan.benchmarks.empirical_active_learning import kernel_label_probabilities


def test_kernel_label_probabilities_are_valid_and_label_dependent():
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    query = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    probabilities = kernel_label_probabilities(X, labels, query)
    assert probabilities.shape == (2, 1)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert probabilities[0, 0] < 0.5 < probabilities[1, 0]
    assert float(classification_accuracy(labels, probabilities)) == 1.0
    assert float(brier_score(labels, probabilities)) < 0.25
