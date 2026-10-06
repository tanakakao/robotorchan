"""Tests for the heterogeneous model container baseline."""

import pytest
import torch
from torch import nn

from robotorchan.models import (
    HeterogeneousModel,
    KroneckerMultiTaskGP,
    ModelListGP,
    MultiTaskGP,
    SingleTaskGP,
)
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.classification.binary.non_gp.sklearn import RandomForestBinaryClassifier


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


def test_long_format_multitask_gp_uses_selected_tasks_as_outputs() -> None:
    data_X = torch.rand(6, 2)
    task_zero = torch.zeros(6, 1)
    task_one = torch.ones(6, 1)
    train_X = torch.cat(
        [
            torch.cat([data_X, task_zero], dim=-1),
            torch.cat([data_X, task_one], dim=-1),
        ],
        dim=0,
    )
    train_Y = torch.rand(12, 1)
    multitask = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        output_tasks=[0, 1],
    )
    model = HeterogeneousModel(multitask)

    assert multitask.num_outputs == 2
    assert model.entry_num_outputs == (2,)
    assert model.num_outputs == 2
    assert model.output_owners == ((0, 0), (0, 1))


def test_long_format_multitask_gp_respects_output_task_subset() -> None:
    data_X = torch.rand(6, 2)
    rows = [torch.cat([data_X, torch.full((6, 1), float(task))], dim=-1) for task in range(3)]
    train_X = torch.cat(rows, dim=0)
    train_Y = torch.rand(18, 1)
    multitask = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        output_tasks=[0, 2],
    )
    model = HeterogeneousModel(multitask)

    assert multitask.num_outputs == 2
    assert model.entry_num_outputs == (2,)
    assert model.output_owners == ((0, 0), (0, 1))


def test_multitask_gp_remains_one_entry_with_native_posterior() -> None:
    data_X = torch.rand(6, 2)
    train_X = torch.cat(
        [
            torch.cat([data_X, torch.zeros(6, 1)], dim=-1),
            torch.cat([data_X, torch.ones(6, 1)], dim=-1),
        ],
        dim=0,
    )
    train_Y = torch.rand(12, 1)
    multitask = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        output_tasks=[0, 1],
    )
    model = HeterogeneousModel(multitask)

    X = torch.rand(3, 2)
    native_posterior = multitask.posterior(X)
    nested_posterior = model[0].posterior(X)

    assert len(model) == 1
    assert model[0] is multitask
    assert type(nested_posterior) is type(native_posterior)
    assert nested_posterior.mean.shape == native_posterior.mean.shape
    assert nested_posterior.mean.shape[-1] == 2
    assert not hasattr(model, "posterior")


def test_multitask_gp_and_classifier_keep_task_and_observation_semantics_separate() -> None:
    data_X = torch.rand(6, 2)
    train_X = torch.cat(
        [
            torch.cat([data_X, torch.zeros(6, 1)], dim=-1),
            torch.cat([data_X, torch.ones(6, 1)], dim=-1),
        ],
        dim=0,
    )
    multitask = MultiTaskGP(
        train_X,
        torch.rand(12, 1),
        task_feature=2,
        output_tasks=[0, 1],
    )
    classifier = BinarySingleTaskGPClassifier(
        data_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(multitask, classifier)

    assert model.entry_num_outputs == (2, 1)
    assert model.output_owners == ((0, 0), (0, 1), (1, 0))
    assert model.regression_output_indices == (0, 1)
    assert model.classification_output_indices == (2,)
    assert model.output_classification_metadata(0) is None
    assert model.output_classification_metadata(1) is None
    assert model.output_classification_metadata(2) == classifier.classification_metadata


def test_entry_posterior_delegates_without_changing_native_semantics() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    model = HeterogeneousModel(regression, names=["strength"])
    X = torch.rand(3, 2)

    native = regression.posterior(X)
    delegated = model.entry_posterior("strength", X)

    assert type(delegated) is type(native)
    assert delegated.mean.shape == native.mean.shape
    assert delegated.variance.shape == native.variance.shape


def test_entry_predict_proba_delegates_classification_probability() -> None:
    train_X = torch.rand(6, 2)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(classifier, names=["pass"])
    X = torch.rand(3, 2)

    native = classifier.predict_proba(X)
    delegated = model.entry_predict_proba("pass", X)

    assert torch.equal(delegated, native)
    assert delegated.shape == (3, 2)


def test_entry_predict_proba_rejects_regression_entry() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    model = HeterogeneousModel(regression, names=["strength"])

    with pytest.raises(TypeError, match="not a classification model"):
        model.entry_predict_proba("strength", torch.rand(3, 2))


def test_entry_posterior_rejects_entry_without_posterior_contract() -> None:
    model = HeterogeneousModel(_ToyModel(1.0), names=["custom"])

    with pytest.raises(TypeError, match=r"does not provide posterior\(X\)"):
        model.entry_posterior("custom", torch.rand(3, 2))


def test_prediction_access_keeps_posterior_and_probability_distinct() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(
        regression,
        classifier,
        names=["strength", "pass"],
    )
    X = torch.rand(3, 2)

    regression_posterior = model.entry_posterior("strength", X)
    classification_probability = model.entry_predict_proba("pass", X)

    assert regression_posterior.mean.shape == (3, 1)
    assert classification_probability.shape == (3, 2)
    assert not hasattr(model, "posterior")
    assert not hasattr(model, "predict_proba")


def test_classification_latent_posterior_is_explicitly_distinct_from_probability() -> None:
    train_X = torch.rand(6, 2)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(classifier, names=["pass"])
    X = torch.rand(3, 2)

    latent = model.entry_latent_posterior("pass", X)
    probability = model.entry_predict_proba("pass", X)

    assert type(latent) is type(classifier.latent_posterior(X))
    assert latent.mean.shape[-1] == 1
    assert probability.shape[-1] == classifier.num_classes
    assert probability.shape[-1] == 2


def test_classification_entry_posterior_preserves_native_latent_semantics() -> None:
    train_X = torch.rand(6, 2)
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(classifier)
    X = torch.rand(3, 2)

    native = classifier.posterior(X)
    generic = model.entry_posterior(0, X)
    explicit = model.entry_latent_posterior(0, X)

    assert type(generic) is type(native)
    assert type(explicit) is type(native)
    assert generic.mean.shape == explicit.mean.shape == native.mean.shape
    assert generic.variance.shape == explicit.variance.shape == native.variance.shape


def test_entry_latent_posterior_rejects_regression_entry() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    model = HeterogeneousModel(regression, names=["strength"])

    with pytest.raises(TypeError, match="not a classification model"):
        model.entry_latent_posterior("strength", torch.rand(3, 2))


def test_multitask_regression_posterior_keeps_native_output_dimension() -> None:
    train_X = torch.rand(6, 2)
    multitask = KroneckerMultiTaskGP(train_X, torch.rand(6, 2))
    model = HeterogeneousModel(multitask, names=["properties"])
    X = torch.rand(3, 2)

    native = multitask.posterior(X)
    delegated = model.entry_posterior("properties", X)

    assert model.entry_num_outputs == (2,)
    assert type(delegated) is type(native)
    assert delegated.mean.shape == native.mean.shape
    assert delegated.mean.shape[-1] == 2


def test_entry_make_mll_preserves_exact_gp_training_objective() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    model = HeterogeneousModel(regression, names=["strength"])

    native = regression.make_mll()
    delegated = model.entry_make_mll("strength")

    assert type(delegated) is type(native)
    assert delegated.model is regression


def test_entry_make_mll_rejects_entry_without_mll_contract() -> None:
    model = HeterogeneousModel(_ToyModel(1.0), names=["custom"])

    with pytest.raises(TypeError, match=r"does not provide make_mll\(\)"):
        model.entry_make_mll("custom")


def test_entry_fit_delegates_explicit_non_gp_training() -> None:
    train_X = torch.rand(8, 2)
    classifier = RandomForestBinaryClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        n_estimators=4,
        random_state=0,
    )
    model = HeterogeneousModel(classifier, names=["pass"])

    assert classifier.is_fitted is False
    model.entry_fit("pass")
    assert classifier.is_fitted is True


def test_entry_fit_rejects_entry_without_explicit_fit_contract() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    model = HeterogeneousModel(regression, names=["strength"])

    with pytest.raises(TypeError, match=r"does not provide fit\(\)"):
        model.entry_fit("strength")


def test_heterogeneous_model_does_not_define_global_training_objective() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)

    assert not hasattr(model, "make_mll")
    assert not hasattr(model, "fit")


def test_training_data_remains_owned_by_each_entry() -> None:
    regression_X = torch.rand(7, 2)
    regression_Y = torch.rand(7, 1)
    classification_X = torch.rand(5, 2)
    classification_Y = torch.tensor([0, 1, 0, 1, 1])
    regression = SingleTaskGP(regression_X, regression_Y)
    classifier = BinarySingleTaskGPClassifier(classification_X, classification_Y)
    model = HeterogeneousModel(regression, classifier)

    assert torch.equal(model.raw_train_Xs[0], regression_X)
    assert torch.equal(model.raw_train_Ys[0], regression_Y)
    assert torch.equal(model.raw_train_Xs[1], classification_X)
    assert torch.equal(model.raw_train_Ys[1], classification_Y)


def test_training_data_contract_allows_partial_observation_rows() -> None:
    strength_X = torch.rand(8, 2)
    pass_X = torch.rand(3, 2)
    strength = SingleTaskGP(strength_X, torch.rand(8, 1))
    classifier = BinarySingleTaskGPClassifier(
        pass_X,
        torch.tensor([0, 1, 1]),
    )
    model = HeterogeneousModel(strength, classifier)

    assert model.raw_train_Xs[0].shape[0] == 8
    assert model.raw_train_Xs[1].shape[0] == 3
    assert not hasattr(model, "raw_train_X")
    assert not hasattr(model, "raw_train_Y")


def test_training_data_contract_reports_missing_optional_data_per_entry() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    custom = _ToyModel(1.0)
    model = HeterogeneousModel(regression, custom)

    assert torch.equal(model.raw_train_Xs[0], train_X)
    assert model.raw_train_Xs[1] is None
    assert model.raw_train_Ys[1] is None
    assert model.raw_train_Yvars[1] is None


def test_training_data_views_follow_child_owned_buffers() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    model = HeterogeneousModel(regression)

    assert model.raw_train_Xs[0] is regression.raw_train_X
    assert model.raw_train_Ys[0] is regression.raw_train_Y


def test_to_dtype_propagates_to_registered_entries_and_raw_data() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)

    returned = model.to(dtype=torch.float64)

    assert returned is model
    assert next(regression.parameters()).dtype == torch.float64
    assert next(classifier.parameters()).dtype == torch.float64
    assert model.raw_train_Xs[0].dtype == torch.float64
    assert model.raw_train_Xs[1].dtype == torch.float64


def test_prediction_access_preserves_native_batched_input_shape() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)
    X = torch.rand(4, 3, 2)

    posterior = model.entry_posterior(0, X)
    probabilities = model.entry_predict_proba(1, X)

    assert posterior.mean.shape == (4, 3, 1)
    assert probabilities.shape == (4, 3, 2)


def test_entries_may_retain_different_dtypes_until_container_is_moved() -> None:
    regression_X = torch.rand(6, 2, dtype=torch.float64)
    classification_X = torch.rand(6, 2, dtype=torch.float32)
    regression = SingleTaskGP(regression_X, torch.rand(6, 1, dtype=torch.float64))
    classifier = BinarySingleTaskGPClassifier(
        classification_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)

    assert model.raw_train_Xs[0].dtype == torch.float64
    assert model.raw_train_Xs[1].dtype == torch.float32


def test_train_and_eval_modes_propagate_to_all_entries() -> None:
    train_X = torch.rand(6, 2)
    regression = SingleTaskGP(train_X, torch.rand(6, 1))
    classifier = BinarySingleTaskGPClassifier(
        train_X,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    model = HeterogeneousModel(regression, classifier)

    model.eval()
    assert model.training is False
    assert regression.training is False
    assert classifier.training is False

    model.train()
    assert model.training is True
    assert regression.training is True
    assert classifier.training is True
