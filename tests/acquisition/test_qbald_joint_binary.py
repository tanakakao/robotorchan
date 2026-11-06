"""Focused binary joint-label qBALD contracts."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.acquisition import qBALD
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.models.classification import BinarySingleTaskGPClassifier


@pytest.mark.parametrize("q", [1, 2, 4])
def test_qbald_matches_exact_joint_label_information(q: int) -> None:
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    X = torch.linspace(0.1, 0.9, q, dtype=torch.double).reshape(1, q, 1)
    positive = torch.tensor([0.2, 0.8], dtype=torch.double).reshape(2, 1, 1)
    probabilities = torch.stack(
        [(1 - positive).expand(2, 1, q), positive.expand(2, 1, q)], dim=-1
    )
    with patch.object(model, "sample_class_probabilities", return_value=probabilities):
        value = qBALD(model, num_samples=2)(X)
    labels = torch.cartesian_prod(*[torch.arange(2) for _ in range(q)]).reshape(-1, q)
    joint = torch.where(
        labels == 1,
        probabilities[..., 1].unsqueeze(-2),
        probabilities[..., 0].unsqueeze(-2),
    ).prod(dim=-1)
    marginal = joint.mean(dim=0)
    expected = (
        -torch.special.xlogy(marginal, marginal).sum(dim=-1)
        + torch.special.xlogy(joint, joint).sum(dim=-1).mean(dim=0)
    )
    assert value.shape == torch.Size([1])
    torch.testing.assert_close(value, expected)
    assert get_acquisition_registry_entry("qBALD").capabilities.max_q == 8


def test_qbald_rejects_exponential_batch_size() -> None:
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 1.0], dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=2)
    with pytest.raises(ValueError, match="1 <= q <= 8"):
        qBALD(model)(torch.zeros(1, 9, 1, dtype=torch.double))
