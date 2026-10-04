"""Tests for classification label validation."""

import pytest
import torch

from robotorchan.models.classification import validate_binary_labels


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
        torch.tensor([False, True, True, False]),
        torch.tensor([[0.0], [1.0]]),
    ],
)
def test_validate_binary_labels_accepts_canonical_labels(labels: torch.Tensor) -> None:
    validated = validate_binary_labels(labels)
    assert validated is labels


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([-1, 1]),
        torch.tensor([0, 2]),
        torch.tensor([0.0, 0.5, 1.0]),
    ],
)
def test_validate_binary_labels_rejects_noncanonical_values(labels: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_binary_labels(labels)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0.0, float("nan")]),
        torch.tensor([1.0, float("inf")]),
    ],
)
def test_validate_binary_labels_rejects_nonfinite_values(labels: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_binary_labels(labels)


def test_validate_binary_labels_rejects_empty_tensor() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_binary_labels(torch.tensor([]))


def test_validate_binary_labels_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        validate_binary_labels([0, 1])  # type: ignore[arg-type]


def test_validate_binary_labels_rejects_complex_dtype() -> None:
    labels = torch.tensor([0 + 0j, 1 + 0j])
    with pytest.raises(TypeError, match="real-valued or boolean"):
        validate_binary_labels(labels)
