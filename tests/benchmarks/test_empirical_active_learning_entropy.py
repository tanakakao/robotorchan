"""Verify adaptive kernel entropy selection and explicit input validation."""

import pytest
import torch

from robotorchan.benchmarks.empirical_active_learning_entropy import (
    select_kernel_entropy_candidate,
)


def test_kernel_entropy_prefers_uncertain_candidate():
    train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    pool = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float64)
    assert select_kernel_entropy_candidate(train_X, labels, pool) == 1


def test_kernel_entropy_rejects_empty_pool():
    train_X = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    labels = torch.tensor([[1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="nonempty"):
        select_kernel_entropy_candidate(train_X, labels, train_X[:0])
