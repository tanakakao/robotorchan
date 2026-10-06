"""Tests for the heterogeneous model container baseline."""

import pytest
import torch
from torch import nn

from robotorchan.models.heterogeneous import HeterogeneousModel


class _ToyModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))


def test_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        HeterogeneousModel()


def test_rejects_non_module_entries() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        HeterogeneousModel(_ToyModel(1.0), object())  # type: ignore[arg-type]


def test_preserves_child_identity_and_order() -> None:
    first = _ToyModel(1.0)
    second = _ToyModel(2.0)

    model = HeterogeneousModel(first, second)

    assert len(model) == 2
    assert model[0] is first
    assert model[1] is second
    assert list(model) == [first, second]
    assert list(model.entries) == [first, second]


def test_module_operations_propagate_to_children() -> None:
    first = _ToyModel(1.0)
    second = _ToyModel(2.0)
    model = HeterogeneousModel(first, second)

    model = model.to(dtype=torch.float64)
    model.eval()

    assert first.weight.dtype is torch.float64
    assert second.weight.dtype is torch.float64
    assert not first.training
    assert not second.training


def test_state_dict_contains_registered_children() -> None:
    model = HeterogeneousModel(_ToyModel(1.0), _ToyModel(2.0))

    assert set(model.state_dict()) == {"models.0.weight", "models.1.weight"}


def test_container_does_not_define_shared_posterior() -> None:
    model = HeterogeneousModel(_ToyModel(1.0))

    assert not hasattr(model, "posterior")
