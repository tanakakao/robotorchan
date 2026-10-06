"""Tests for the heterogeneous model container baseline."""

import pytest
import torch
from torch import nn

from robotorchan.models import HeterogeneousModel, KroneckerMultiTaskGP, SingleTaskGP
from robotorchan.models.classification import BinarySingleTaskGPClassifier


class _ToyModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))


def test_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        HeterogeneousModel()


def test_rejects_non_module_entries() -> None:
    with pytest.raises(TypeError, match=r"torch\.nn\.Module"):
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


def test_composes_regression_and_classification_models_without_wrapping() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classification = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )

    model = HeterogeneousModel(regression, classification)

    assert model[0] is regression
    assert model[1] is classification
    assert model[0].posterior(train_X[:2]).mean.shape[-1] == 1
    assert model[1].predict_proba(train_X[:2]).shape == torch.Size([2, 2])


def test_optional_names_are_aliases_for_entry_indices() -> None:
    first = _ToyModel(1.0)
    second = _ToyModel(2.0)
    model = HeterogeneousModel(first, second, names=["strength", "pass"])

    assert model.names == ("strength", "pass")
    assert model["strength"] is model[0] is first
    assert model["pass"] is model[1] is second


def test_names_are_optional_per_entry() -> None:
    first = _ToyModel(1.0)
    second = _ToyModel(2.0)
    model = HeterogeneousModel(first, second, names=[None, "pass"])

    assert model.names == (None, "pass")
    assert model[0] is first
    assert model["pass"] is second


def test_names_must_match_entry_count() -> None:
    with pytest.raises(ValueError, match="one entry for each model"):
        HeterogeneousModel(_ToyModel(1.0), _ToyModel(2.0), names=["only-one"])


def test_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        HeterogeneousModel(_ToyModel(1.0), _ToyModel(2.0), names=["same", "same"])


def test_names_must_be_strings_or_none() -> None:
    with pytest.raises(TypeError, match="string or None"):
        HeterogeneousModel(_ToyModel(1.0), names=[1])  # type: ignore[list-item]


def test_unknown_name_raises_clear_key_error() -> None:
    model = HeterogeneousModel(_ToyModel(1.0), names=["strength"])

    with pytest.raises(KeyError, match="Unknown heterogeneous model name"):
        model["missing"]


def test_tracks_multiple_regression_outputs_per_entry() -> None:
    train_X = torch.rand(6, 2)
    multi_output = KroneckerMultiTaskGP(train_X, torch.rand(6, 2))
    single_output = SingleTaskGP(train_X, torch.rand(6, 1))

    model = HeterogeneousModel(multi_output, single_output)

    assert model.entry_num_outputs == (2, 1)
    assert model.num_outputs == 3
    assert model.output_owner(0) == (0, 0)
    assert model.output_owner(1) == (0, 1)
    assert model.output_owner(2) == (1, 0)


def test_output_owner_supports_negative_global_indices() -> None:
    train_X = torch.rand(6, 2)
    model = HeterogeneousModel(
        KroneckerMultiTaskGP(train_X, torch.rand(6, 2)),
        SingleTaskGP(train_X, torch.rand(6, 1)),
    )

    assert model.output_owner(-1) == (1, 0)
    assert model.output_owner(-2) == (0, 1)


def test_output_owner_rejects_invalid_global_indices() -> None:
    model = HeterogeneousModel(_ToyModelWithOutputs(1.0))

    with pytest.raises(IndexError, match="out of range"):
        model.output_owner(1)
    with pytest.raises(IndexError, match="out of range"):
        model.output_owner(-2)
    with pytest.raises(TypeError, match="must be an integer"):
        model.output_owner("0")  # type: ignore[arg-type]


class _ToyModelWithOutputs(_ToyModel):
    @property
    def num_outputs(self) -> int:
        return 1


class _InvalidOutputCountModel(_ToyModel):
    @property
    def num_outputs(self) -> int:
        return 0


def test_requires_declared_positive_output_count() -> None:
    with pytest.raises(ValueError, match="positive integer num_outputs"):
        HeterogeneousModel(_ToyModel(1.0))

    with pytest.raises(ValueError, match="positive integer num_outputs"):
        HeterogeneousModel(_InvalidOutputCountModel(1.0))
