import pytest
import torch
from torch import nn

from robotorchan.models.base import (
    ModelTrainingMixin,
    RawDataMixin,
    UnsupportedModelOperationError,
)


class DummyRawModel(RawDataMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()


class DummyNonMLLModel(ModelTrainingMixin):
    pass


def test_raw_data_registry_is_empty_before_storage() -> None:
    model = DummyRawModel()

    assert model.raw_data == {}


def test_raw_data_mixin_stores_arbitrary_named_tensors() -> None:
    datapoints = torch.rand(6, 3, dtype=torch.double)
    comparisons = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    model = DummyRawModel()

    model._store_raw_tensor("datapoints", datapoints)
    model._store_raw_tensor("comparisons", comparisons)
    model._store_raw_tensor("optional", None)

    assert set(model.raw_data) == {"datapoints", "comparisons", "optional"}
    assert torch.equal(model.raw_data["datapoints"], datapoints)
    assert torch.equal(model.raw_data["comparisons"], comparisons)
    assert model.raw_data["optional"] is None
    assert model.raw_data["datapoints"].data_ptr() != datapoints.data_ptr()


def test_raw_data_buffers_follow_dtype_and_are_serialized() -> None:
    model = DummyRawModel()
    model._store_raw_tensor("train_X", torch.rand(4, 2, dtype=torch.double))
    model._store_raw_tensor("missing", None)

    model = model.to(dtype=torch.float32)
    state_dict = model.state_dict()

    assert model.raw_data["train_X"].dtype == torch.float32
    assert "_raw_train_X" in state_dict
    assert "_raw_missing" not in state_dict
    assert "missing" in model.raw_data
    assert model.raw_data["missing"] is None


def test_raw_data_tensor_can_be_replaced_without_duplicate_registration() -> None:
    model = DummyRawModel()
    first = torch.zeros(2, 1)
    second = torch.ones(2, 1)

    model._store_raw_tensor("train_Y", first)
    model._store_raw_tensor("train_Y", second)

    assert tuple(model.raw_data) == ("train_Y",)
    assert torch.equal(model.raw_data["train_Y"], second)


def test_raw_data_name_validation() -> None:
    model = DummyRawModel()

    with pytest.raises(ValueError, match="non-empty"):
        model._store_raw_tensor("", torch.ones(1))

    with pytest.raises(ValueError, match="cannot contain"):
        model._store_raw_tensor("bad.name", torch.ones(1))


def test_default_training_capability_rejects_mll() -> None:
    model = DummyNonMLLModel()

    assert model.supports_mll is False
    with pytest.raises(UnsupportedModelOperationError, match="does not support make_mll"):
        model.make_mll()
