"""Tests for the heterogeneous model container baseline."""

import pytest
import torch
from torch import nn

from robotorchan.models import HeterogeneousModel, KroneckerMultiTaskGP, ModelListGP, SingleTaskGP
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


def test_undeclared_output_count_defaults_to_one() -> None:
    model = HeterogeneousModel(_ToyModel(1.0))

    assert model.entry_num_outputs == (1,)
    assert model.num_outputs == 1


def test_declared_output_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HeterogeneousModel(_InvalidOutputCountModel(1.0))


def test_binary_class_probabilities_remain_one_heterogeneous_output() -> None:
    train_X = torch.rand(6, 2)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(classifier)

    assert classifier.predict_proba(train_X[:2]).shape[-1] == 2
    assert classifier.num_outputs == 1
    assert model.entry_num_outputs == (1,)
    assert model.num_outputs == 1
    assert model.output_owner(0) == (0, 0)


def test_classification_metadata_is_resolved_by_global_output() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)

    assert model.output_classification_metadata(0) is None
    metadata = model.output_classification_metadata(1)
    assert metadata == classifier.classification_metadata
    assert metadata.num_classes == 2


def test_multiple_classification_entries_keep_independent_metadata() -> None:
    train_X = torch.rand(6, 2)
    first = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    second = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([1, 1, 0, 0, 1, 0]),
    )
    model = HeterogeneousModel(first, second)

    assert model.num_outputs == 2
    assert model.output_owner(0) == (0, 0)
    assert model.output_owner(1) == (1, 0)
    assert model.output_classification_metadata(0) == first.classification_metadata
    assert model.output_classification_metadata(1) == second.classification_metadata


def test_mixed_n_by_m_composition_preserves_global_output_layout() -> None:
    train_X = torch.rand(6, 2)
    regression_multi = KroneckerMultiTaskGP(train_X, torch.rand(6, 2))
    classifier_a = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    regression_single = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier_b = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([1, 1, 0, 0, 1, 0]),
    )

    model = HeterogeneousModel(
        regression_multi,
        classifier_a,
        regression_single,
        classifier_b,
        names=["properties", "pass", "cost", "stable"],
    )

    assert model.entry_num_outputs == (2, 1, 1, 1)
    assert model.num_outputs == 5
    assert model.output_owners == ((0, 0), (0, 1), (1, 0), (2, 0), (3, 0))
    assert model.regression_output_indices == (0, 1, 3)
    assert model.classification_output_indices == (2, 4)
    assert model["properties"] is regression_multi
    assert model["pass"] is classifier_a
    assert model["cost"] is regression_single
    assert model["stable"] is classifier_b


def test_mixed_output_metadata_remains_entry_local() -> None:
    train_X = torch.rand(6, 2)
    regression = KroneckerMultiTaskGP(train_X, torch.rand(6, 2))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)

    assert model.output_classification_metadata(0) is None
    assert model.output_classification_metadata(1) is None
    assert model.output_classification_metadata(2) == classifier.classification_metadata


def test_model_list_gp_remains_one_heterogeneous_entry() -> None:
    train_X = torch.rand(6, 2)
    first = SingleTaskGP(train_X, torch.rand(6, 1))
    second = SingleTaskGP(train_X, torch.rand(6, 1))
    model_list = ModelListGP(first, second)

    model = HeterogeneousModel(model_list)

    assert len(model) == 1
    assert model[0] is model_list
    assert model.entry_num_outputs == (2,)
    assert model.num_outputs == 2
    assert model.output_owners == ((0, 0), (0, 1))
    assert model.regression_output_indices == (0, 1)


def test_model_list_gp_native_posterior_is_not_reimplemented() -> None:
    train_X = torch.rand(6, 2)
    first = SingleTaskGP(train_X, torch.rand(6, 1))
    second = SingleTaskGP(train_X, torch.rand(6, 1))
    model_list = ModelListGP(first, second)
    model = HeterogeneousModel(model_list)

    X = torch.rand(2, 2)
    native_posterior = model_list.posterior(X)
    nested_posterior = model[0].posterior(X)

    assert type(nested_posterior) is type(native_posterior)
    assert nested_posterior.mean.shape == native_posterior.mean.shape
    assert nested_posterior.variance.shape == native_posterior.variance.shape
    assert not hasattr(model, "posterior")


def test_model_list_gp_raw_training_data_remains_child_owned() -> None:
    first_X = torch.rand(6, 2)
    second_X = torch.rand(7, 2)
    first = SingleTaskGP(first_X, torch.rand(6, 1))
    second = SingleTaskGP(second_X, torch.rand(7, 1))
    model_list = ModelListGP(first, second)
    model = HeterogeneousModel(model_list)

    assert model[0].raw_train_Xs[0] is first.raw_train_X
    assert model[0].raw_train_Xs[1] is second.raw_train_X
    assert not hasattr(model, "raw_train_X")


def test_model_list_gp_can_coexist_with_classifier_entry() -> None:
    train_X = torch.rand(6, 2)
    first = SingleTaskGP(train_X, torch.rand(6, 1))
    second = SingleTaskGP(train_X, torch.rand(6, 1))
    model_list = ModelListGP(first, second)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )

    model = HeterogeneousModel(model_list, classifier)

    assert model.entry_num_outputs == (2, 1)
    assert model.output_owners == ((0, 0), (0, 1), (1, 0))
    assert model.regression_output_indices == (0, 1)
    assert model.classification_output_indices == (2,)
    assert model.output_classification_metadata(2) == classifier.classification_metadata
